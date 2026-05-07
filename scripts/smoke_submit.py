#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, subtitles, watermark, text, logo, distorted anatomy, "
    "extra limbs, bad hands, bad face, distorted sound, saturated sound, loud sound"
)

ACTION_PROMPTS = {
    "missionary": (
        "A realistic cinematic adult scene continues from the starting frame with steady missionary motion, "
        "subtle body movement and natural skin contact over time. The camera stays mostly static with a gentle "
        "handheld feel, preserving the subject identity and composition. Native audio should include quiet room "
        "ambience, soft breathing, bedsheet movement, and natural intimate sounds with no added dialogue."
    ),
    "doggy": (
        "A realistic cinematic adult scene continues from the starting frame with doggy-style motion that starts "
        "slowly and becomes more rhythmic while preserving anatomy and identity. The camera remains mostly static "
        "from the same angle with minimal shake. Native audio should include quiet room ambience, soft breathing, "
        "subtle movement sounds, and natural intimate sound without speech."
    ),
    "cowgirl": (
        "A realistic cinematic adult scene continues from the starting frame with cowgirl motion, visible rhythmic "
        "up-and-down movement, and consistent body positioning. The camera is fixed at a flattering portrait angle "
        "with only slight natural handheld motion. Native audio should include room tone, soft breathing, fabric "
        "movement, and restrained intimate sounds with no dialogue."
    ),
    "reverse-cowgirl": (
        "A realistic cinematic adult scene continues from the starting frame with reverse cowgirl motion, gradual "
        "rhythmic movement, and stable anatomy throughout the shot. The camera holds a static portrait composition "
        "with minimal motion. Native audio should include subtle ambience, breathing, bedsheet movement, and natural "
        "intimate sound without music or dialogue."
    ),
    "blowjob": (
        "A realistic cinematic adult scene continues from the starting frame with oral-sex motion that begins slowly "
        "and stays controlled, preserving facial identity and natural head movement. The camera remains mostly static "
        "with a close portrait-friendly framing. Native audio should include soft breathing, subtle mouth movement "
        "sounds, quiet room ambience, and no dialogue unless supplied by the caller."
    ),
}


def _read_runpod_api_key() -> str:
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if key:
        return key

    config_path = Path.home() / ".runpod" / "config.toml"
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("apikey"):
                _, raw = line.split("=", 1)
                key = raw.strip().strip("'\"")
                if key:
                    return key

    raise SystemExit("RUNPOD_API_KEY is not set and ~/.runpod/config.toml has no apikey")


def _request(method: str, url: str, api_key: str, **kwargs: Any) -> Dict[str, Any]:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {api_key}"
    response = requests.request(method, url, headers=headers, timeout=120, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"RunPod returned non-object JSON: {data!r}")
    return data


def _build_payload(args: argparse.Namespace, action: str) -> Dict[str, Any]:
    if args.prewarm:
        return {"input": {"action": "prewarm_core_models"}}

    prompt = args.prompt or ACTION_PROMPTS[action]
    return {
        "input": {
            "workflow_id": "ltx23_dr34ml4y_i2v_v1",
            "image": args.image_url,
            "prompt": prompt,
            "action": action,
            "duration": args.duration,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "seed": args.seed,
            "generate_audio": not args.no_audio,
            "negative_prompt": args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            "settings": {
                "distill_lora_scale": args.distill_lora_scale,
                "dr34ml4y_lora_scale": args.dr34ml4y_lora_scale,
                "video_cfg": args.video_cfg,
                "audio_cfg": args.audio_cfg,
                "stage_one_steps": args.stage_one_steps,
                "stage_two_steps": args.stage_two_steps,
            },
        }
    }


def _submit(endpoint_id: str, api_key: str, payload: Dict[str, Any]) -> str:
    data = _request("POST", f"https://api.runpod.ai/v2/{endpoint_id}/run", api_key, json=payload)
    job_id = str(data.get("id") or "").strip()
    if not job_id:
        raise RuntimeError(f"RunPod did not return a job id: {data}")
    return job_id


def _poll(endpoint_id: str, api_key: str, job_id: str, timeout_seconds: int, poll_seconds: int) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        data = _request("GET", f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}", api_key)
        status = str(data.get("status") or "").upper()
        print(json.dumps({"job_id": job_id, "status": status, "delayTime": data.get("delayTime"), "executionTime": data.get("executionTime")}))
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return data
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for {job_id}; last status={status}")
        time.sleep(poll_seconds)


def _extract_video_url(status: Dict[str, Any]) -> Optional[str]:
    output = status.get("output")
    if isinstance(output, dict):
        video_url = output.get("video_url")
        if isinstance(video_url, str) and video_url.startswith("https://"):
            return video_url
    return None


def _ffprobe_has_audio(video_url: str) -> Optional[bool]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                video_url,
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except FileNotFoundError:
        return None
    except subprocess.CalledProcessError:
        return False

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False
    streams = data.get("streams")
    return isinstance(streams, list) and len(streams) > 0


def _iter_actions(value: str) -> Iterable[str]:
    if value == "all":
        return ACTION_PROMPTS.keys()
    if value not in ACTION_PROMPTS:
        raise SystemExit(f"--action must be one of {', '.join(ACTION_PROMPTS)} or all")
    return [value]


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit and poll CEL-180 LTX-2.3 DR34ML4Y RunPod smoke jobs.")
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--image-url", help="HTTPS URL for a consenting/synthetic adult starting image.")
    parser.add_argument("--action", default="missionary", help="One action id or 'all'.")
    parser.add_argument("--prewarm", action="store_true", help="Only prewarm core models and return.")
    parser.add_argument("--prompt", help="Override the built-in prompt for a single action run.")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--distill-lora-scale", type=float, default=0.30)
    parser.add_argument("--dr34ml4y-lora-scale", type=float, default=1.0)
    parser.add_argument("--video-cfg", type=float, default=2.5)
    parser.add_argument("--audio-cfg", type=float, default=7.0)
    parser.add_argument("--stage-one-steps", type=int, default=18)
    parser.add_argument("--stage-two-steps", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--verify-audio", action="store_true", help="Run ffprobe against completed MP4 URLs when available.")
    parser.add_argument("--output-jsonl", type=Path)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    if not args.prewarm and not args.image_url:
        raise SystemExit("--image-url is required unless --prewarm is set")
    api_key = _read_runpod_api_key()
    output_file = args.output_jsonl.open("a", encoding="utf-8") if args.output_jsonl else None

    try:
        actions = ["prewarm"] if args.prewarm else list(_iter_actions(args.action))
        for action in actions:
            payload = _build_payload(args, action)
            print(json.dumps({"action": action, "event": "submit"}))
            job_id = _submit(args.endpoint_id, api_key, payload)
            status = _poll(args.endpoint_id, api_key, job_id, args.timeout_seconds, args.poll_seconds)
            video_url = _extract_video_url(status)
            record: Dict[str, Any] = {
                "action": action,
                "job_id": job_id,
                "status": status.get("status"),
                "video_url": video_url,
                "raw_status": status,
            }
            if args.verify_audio and video_url:
                record["has_audio"] = _ffprobe_has_audio(video_url)
            print(json.dumps(record, sort_keys=True))
            if output_file:
                output_file.write(json.dumps(record, sort_keys=True) + "\n")
                output_file.flush()
            if status.get("status") != "COMPLETED":
                return 1
    finally:
        if output_file:
            output_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
