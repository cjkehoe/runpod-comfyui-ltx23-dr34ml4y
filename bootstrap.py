import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import requests

MANIFEST_PATH = Path(__file__).with_name("asset-manifest.json")
DEFAULT_COMFY_ROOT = Path(os.getenv("COMFY_ROOT", "/workspace/ComfyUI"))
DEFAULT_NETWORK_VOLUME_ROOT = Path(os.getenv("NETWORK_VOLUME_ROOT", "/runpod-volume"))


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _resolve_cache_destination(
    relative_path: str, filename: str, network_volume_root: Path
) -> Optional[Path]:
    if not network_volume_root.exists() or not network_volume_root.is_dir():
        return None

    return network_volume_root / "comfyui" / relative_path / filename


def _ensure_core_model(model: Dict[str, Any], comfy_root: Path, network_volume_root: Path) -> None:
    relative_path = str(model["relative_path"]).strip("/")
    destination = comfy_root / relative_path / str(model["filename"])
    if destination.exists() and destination.stat().st_size > 0:
        return

    cache_destination = _resolve_cache_destination(relative_path, str(model["filename"]), network_volume_root)
    if cache_destination and cache_destination.exists() and cache_destination.stat().st_size > 0:
        print(f"Copying cached core model {cache_destination} -> {destination}")
        _copy_file(cache_destination, destination)
        return

    url = os.getenv(str(model["url_env"]), "").strip()
    if not url:
        print(f"Skipping {model['filename']} - {model['url_env']} is not set")
        return

    print(f"Downloading {model['filename']} -> {destination}")
    _download_file(url, destination)

    if cache_destination:
        _copy_file(destination, cache_destination)


def main() -> None:
    manifest = _load_manifest()
    for model in manifest.get("core_models", []):
        _ensure_core_model(model, DEFAULT_COMFY_ROOT, DEFAULT_NETWORK_VOLUME_ROOT)


if __name__ == "__main__":
    main()
