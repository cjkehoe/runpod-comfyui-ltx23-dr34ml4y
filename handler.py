import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set
from urllib.parse import urlparse

import requests
import runpod

from workflow_builder import (
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    WorkflowInputError,
    build_ltx23_dr34ml4y_job_input,
    is_high_level_ltx23_request,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_HANDLER_PATH = Path(os.getenv("BASE_HANDLER_PATH", "/handler_base.py"))
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("asset-manifest.json")
MANIFEST_PATH = Path(
    os.getenv(
        "ASSET_MANIFEST_PATH",
        str(DEFAULT_MANIFEST_PATH if DEFAULT_MANIFEST_PATH.exists() else Path("/workspace/asset-manifest.json")),
    )
)
DEFAULT_MODELS_RELATIVE_PATH = "models/loras"
NETWORK_VOLUME_ROOT = Path(os.getenv("NETWORK_VOLUME_ROOT", "/runpod-volume"))
COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/comfyui"))
OUTPUT_PUBLIC_BASE = os.getenv("OUTPUT_PUBLIC_BASE", "").strip().rstrip("/")
OUTPUT_BUCKET_NAME = os.getenv("OUTPUT_BUCKET_NAME", "").strip().strip("/")


def _load_base_handler():
    spec = importlib.util.spec_from_file_location("handler_base", BASE_HANDLER_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load base handler from {BASE_HANDLER_PATH}")

    module = importlib.util.module_from_spec(spec)
    base_handler_dir = str(BASE_HANDLER_PATH.parent)
    added_to_sys_path = False

    original_start = runpod.serverless.start
    runpod.serverless.start = lambda *args, **kwargs: None
    if base_handler_dir and base_handler_dir not in sys.path:
        sys.path.insert(0, base_handler_dir)
        added_to_sys_path = True
    try:
        spec.loader.exec_module(module)
    finally:
        runpod.serverless.start = original_start
        if added_to_sys_path:
            try:
                sys.path.remove(base_handler_dir)
            except ValueError:
                pass

    handler = getattr(module, "handler", None)
    if not callable(handler):
        raise RuntimeError("Base handler did not expose a callable handler(job)")

    return handler


BASE_HANDLER = _load_base_handler()


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _resolve_job_input(job: Dict[str, Any]) -> Dict[str, Any]:
    payload = job.get("input") or {}

    if isinstance(payload, str):
        try:
            payload = json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            raise ValueError("Job input is not valid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("Job input must be an object")

    return payload


def _normalize_relative_path(value: Any) -> str:
    raw = str(value or DEFAULT_MODELS_RELATIVE_PATH).strip().replace("\\", "/").strip("/")
    parts = [part for part in raw.split("/") if part not in {"", ".", ".."}]
    return "/".join(parts) or DEFAULT_MODELS_RELATIVE_PATH


def _iter_model_downloads(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    downloads = payload.pop("model_downloads", None)

    if downloads is None:
        return []

    if not isinstance(downloads, list):
        raise ValueError("model_downloads must be a list")

    return downloads


def _download_file(url: str, destination: Path) -> None:
    tmp_destination = destination.with_suffix(f"{destination.suffix}.partial")

    if tmp_destination.exists():
        tmp_destination.unlink()

    try:
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()

            with tmp_destination.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output_file.write(chunk)

        os.replace(tmp_destination, destination)
    except Exception:
        if tmp_destination.exists():
            tmp_destination.unlink()
        raise


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _get_expected_file_size(url: str) -> Optional[int]:
    try:
        response = requests.head(url, allow_redirects=True, timeout=120)
        response.raise_for_status()
    except Exception:
        logger.warning("Unable to determine remote size for %s; continuing without size validation", url)
        return None

    raw_size = response.headers.get("Content-Length") or response.headers.get("content-length")
    if not raw_size:
        return None

    try:
        return int(raw_size)
    except (TypeError, ValueError):
        return None


def _has_expected_size(path: Path, expected_size: Optional[int]) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    if expected_size is None:
        return True

    return path.stat().st_size == expected_size


def _resolve_cache_destination(relative_path: str, filename: str) -> Optional[Path]:
    if not NETWORK_VOLUME_ROOT.exists() or not NETWORK_VOLUME_ROOT.is_dir():
        return None

    return NETWORK_VOLUME_ROOT / "comfyui" / relative_path / filename


def _ensure_core_model(model: Dict[str, Any]) -> Optional[str]:
    relative_path = _normalize_relative_path(model.get("relative_path"))
    filename = Path(str(model.get("filename") or "").strip()).name
    if not filename:
        return None

    runtime_destination = COMFY_ROOT / relative_path / filename
    runtime_destination.parent.mkdir(parents=True, exist_ok=True)
    cache_destination = _resolve_cache_destination(relative_path, filename)
    url = os.getenv(str(model.get("url_env") or ""), "").strip() or str(model.get("default_url") or "").strip()
    expected_size = _get_expected_file_size(url) if url else None

    if _has_expected_size(runtime_destination, expected_size):
        logger.info("Using existing core model: %s", runtime_destination)
        return relative_path
    if runtime_destination.exists():
        logger.warning("Discarding incomplete core model at %s", runtime_destination)
        runtime_destination.unlink(missing_ok=True)

    if cache_destination and _has_expected_size(cache_destination, expected_size):
        logger.info("Copying cached core model into Comfy path: %s -> %s", cache_destination, runtime_destination)
        _copy_file(cache_destination, runtime_destination)
        return relative_path
    if cache_destination and cache_destination.exists():
        logger.warning("Discarding incomplete cached core model at %s", cache_destination)
        cache_destination.unlink(missing_ok=True)

    if not url:
        logger.warning(
            "Skipping missing core model %s because %s is unset and no default_url is configured",
            filename,
            model.get("url_env"),
        )
        return None

    logger.info("Downloading core model %s -> %s", filename, runtime_destination)
    _download_file(url, runtime_destination)
    if expected_size is not None and runtime_destination.stat().st_size != expected_size:
        actual_size = runtime_destination.stat().st_size
        runtime_destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded core model {filename} had size {actual_size} but expected {expected_size}"
        )

    if cache_destination:
        try:
            _copy_file(runtime_destination, cache_destination)
            logger.info("Cached core model at %s", cache_destination)
        except Exception:
            logger.exception("Failed to cache core model at %s", cache_destination)

    return relative_path


def _ensure_core_models() -> Set[str]:
    try:
        manifest = _load_manifest()
    except FileNotFoundError:
        logger.warning("Asset manifest not found at %s; skipping core model ensure", MANIFEST_PATH)
        return set()

    downloaded_relative_paths: Set[str] = set()
    for model in manifest.get("core_models", []):
        relative_path = _ensure_core_model(model)
        if relative_path:
            downloaded_relative_paths.add(relative_path)

    return downloaded_relative_paths


def _process_model_downloads(payload: Dict[str, Any]) -> Set[str]:
    downloaded_relative_paths: Set[str] = set()

    for index, item in enumerate(_iter_model_downloads(payload)):
        if not isinstance(item, dict):
            raise ValueError(f"model_downloads[{index}] must be an object")

        url = str(item.get("url") or "").strip()
        filename = Path(str(item.get("filename") or "").strip()).name
        relative_path = _normalize_relative_path(item.get("relative_path"))

        if not url:
            raise ValueError(f"model_downloads[{index}] is missing url")

        if not filename:
            raise ValueError(f"model_downloads[{index}] is missing filename")

        runtime_destination = COMFY_ROOT / relative_path / filename
        runtime_destination.parent.mkdir(parents=True, exist_ok=True)
        cache_destination = _resolve_cache_destination(relative_path, filename)

        if runtime_destination.exists() and runtime_destination.stat().st_size > 0:
            logger.info("Using existing runtime asset: %s", runtime_destination)
        elif cache_destination and cache_destination.exists() and cache_destination.stat().st_size > 0:
            logger.info("Copying cached asset into Comfy path: %s -> %s", cache_destination, runtime_destination)
            _copy_file(cache_destination, runtime_destination)
        else:
            logger.info("Downloading runtime asset %s -> %s", filename, runtime_destination)
            _download_file(url, runtime_destination)

            if cache_destination:
                try:
                    _copy_file(runtime_destination, cache_destination)
                    logger.info("Cached runtime asset at %s", cache_destination)
                except Exception:
                    logger.exception("Failed to cache runtime asset at %s", cache_destination)

        downloaded_relative_paths.add(relative_path)

    return downloaded_relative_paths


def _refresh_comfy_file_cache(downloaded_relative_paths: Set[str]) -> None:
    folder_keys = set()

    for relative_path in downloaded_relative_paths:
        if relative_path == "models/loras":
            folder_keys.add("loras")
        elif relative_path == "models/checkpoints":
            folder_keys.add("checkpoints")
        elif relative_path in {"models/clip", "models/text_encoders"}:
            folder_keys.add("clip")
        elif relative_path == "models/latent_upscale_models":
            folder_keys.add("latent_upscale_models")

    if not folder_keys:
        return

    try:
        import folder_paths

        for folder_key in folder_keys:
            folder_paths.filename_list_cache.pop(folder_key, None)
            visible_files = folder_paths.get_filename_list(folder_key)
            logger.info("Refreshed ComfyUI cache for %s. Visible files: %s", folder_key, visible_files[:20])
    except Exception:
        logger.exception("Failed to refresh ComfyUI file cache")


def _rewrite_public_output_url(url: str) -> str:
    if not OUTPUT_PUBLIC_BASE:
        return url

    parsed = urlparse(url)
    path = parsed.path.lstrip("/")

    if OUTPUT_BUCKET_NAME and path.startswith(f"{OUTPUT_BUCKET_NAME}/"):
        path = path[len(OUTPUT_BUCKET_NAME) + 1 :]

    if not path:
        return url

    return f"{OUTPUT_PUBLIC_BASE}/{path}"


def _select_primary_video_url(images: Iterable[Any]) -> Optional[str]:
    fallback_url: Optional[str] = None

    for item in images:
        if not isinstance(item, dict):
            continue

        data = item.get("data")
        if not isinstance(data, str) or not data.strip():
            continue

        filename = str(item.get("filename") or "").strip().lower()
        if filename.endswith("-audio.mp4"):
            return data

        if fallback_url is None:
            fallback_url = data

    return fallback_url


def _normalize_video_output(result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    images = result.get("images")
    if not isinstance(images, list) or not images:
        return result

    normalized_images = []

    for item in images:
        if not isinstance(item, dict):
            normalized_images.append(item)
            continue

        normalized_item = dict(item)
        data = normalized_item.get("data")
        if isinstance(data, str) and data.strip():
            normalized_item["data"] = _rewrite_public_output_url(data)

        normalized_images.append(normalized_item)

    normalized_result = dict(result)
    normalized_result["images"] = normalized_images

    primary_video_url = _select_primary_video_url(normalized_images)
    if primary_video_url:
        normalized_result["video_url"] = primary_video_url

    return normalized_result


def _merge_metadata(result: Any, payload: Dict[str, Any]) -> Any:
    if not isinstance(result, dict):
        return result

    if payload.get("workflow_id") != WORKFLOW_ID:
        return result

    merged = dict(result)
    merged["workflow_id"] = WORKFLOW_ID
    merged["workflow_version"] = WORKFLOW_VERSION
    merged["action"] = payload.get("action")
    merged["seed"] = payload.get("seed")
    merged["settings"] = payload.get("settings") or {}
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        merged["metadata"] = metadata
    return merged


def _is_prewarm_request(payload: Dict[str, Any]) -> bool:
    action = str(payload.get("action") or "").strip().lower()
    if action == "prewarm_core_models":
        return True

    return bool(payload.get("prewarm_core_models"))


def handler(job: Dict[str, Any]) -> Any:
    payload = _resolve_job_input(job)
    downloaded_relative_paths = _ensure_core_models()

    if _is_prewarm_request(payload):
        _refresh_comfy_file_cache(downloaded_relative_paths)
        return {
            "ok": True,
            "mode": "prewarm_core_models",
            "downloaded_relative_paths": sorted(downloaded_relative_paths),
            "network_volume_attached": NETWORK_VOLUME_ROOT.exists() and NETWORK_VOLUME_ROOT.is_dir(),
            "network_volume_root": str(NETWORK_VOLUME_ROOT),
        }

    try:
        if is_high_level_ltx23_request(payload):
            payload = build_ltx23_dr34ml4y_job_input(payload)
    except WorkflowInputError as exc:
        return {"error": str(exc), "workflow_id": WORKFLOW_ID, "workflow_version": WORKFLOW_VERSION}

    downloaded_relative_paths.update(_process_model_downloads(payload))
    _refresh_comfy_file_cache(downloaded_relative_paths)

    job["input"] = payload
    base_result = BASE_HANDLER(job)
    return _merge_metadata(_normalize_video_output(base_result), payload)


runpod.serverless.start({"handler": handler})
