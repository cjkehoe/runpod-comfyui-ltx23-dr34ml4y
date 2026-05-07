import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import unquote, urlparse

WORKFLOW_ID = "ltx23_dr34ml4y_i2v_v1"
WORKFLOW_VERSION = "v1"

LTX_CHECKPOINT_NAME = "ltx-2.3-22b-dev.safetensors"
LTX_TEXT_ENCODER_NAME = "gemma_3_12B_it.safetensors"
LTX_DISTILLED_LORA_NAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
LTX_UPSCALER_NAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DR34ML4Y_LORA_NAME = "DR34ML4Y_LTXXX_V1.safetensors"

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, subtitles, watermark, text, logo, distorted anatomy, "
    "extra limbs, bad hands, bad face, distorted sound, saturated sound, loud sound"
)

ACTION_TRIGGERS = {
    "missionary": "m15510n4ry",
    "doggy": "d0gg1e",
    "cowgirl": "c0wg1rl",
    "reverse-cowgirl": "r3v3rs3_c0wg1rl",
    "blowjob": "bl0wj0b",
}

EXPERIMENTAL_TRIGGERS = {
    "double-blowjob": "d0ubl3_bj",
}

SETTING_DEFAULTS = {
    "distill_lora_scale": 0.30,
    "dr34ml4y_lora_scale": 1.0,
    "video_cfg": 2.5,
    "audio_cfg": 7.0,
    "stage_one_steps": 18,
    "stage_two_steps": 4,
    "img_compression": 18,
    "i2v_strength": 0.7,
}

LTX_STAGE_TWO_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


class WorkflowInputError(ValueError):
    pass


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def _normalize_dimension(value: Any, default: int) -> int:
    dimension = _clamp_int(value, default, 320, 1536)
    return max(320, min(1536, round(dimension / 32) * 32))


def _normalize_duration(value: Any) -> int:
    return _clamp_int(value, 5, 1, 20)


def _normalize_fps(value: Any) -> int:
    return _clamp_int(value, 24, 8, 24)


def _resolve_length(duration: int, fps: int) -> int:
    # LTX frame counts should be 8n + 1. duration * fps + 1 is valid for
    # 24fps five-second tests and we round other durations down to 8n + 1.
    requested = max(9, duration * fps + 1)
    return ((requested - 1) // 8) * 8 + 1


def _resolve_seed(value: Any) -> int:
    try:
        seed = int(value)
    except (TypeError, ValueError):
        seed = -1
    if seed >= 0:
        return seed
    return random.randint(0, 2**31 - 1)


def _infer_filename_from_url(url: str, fallback: str) -> str:
    try:
        parsed = urlparse(url)
        raw_name = parsed.path.split("/")[-1]
        decoded = unquote(raw_name)
        if decoded:
            name = Path(decoded).name
            if name:
                return name
    except Exception:
        pass
    return fallback


def _infer_image_filename(url: str) -> str:
    filename = _infer_filename_from_url(url, "ltx23-start-image.png")
    if "." not in Path(filename).name:
        return f"{filename}.png"
    return filename


def _normalize_action(action: Any, *, allow_experimental: bool = False) -> Tuple[str, str]:
    value = str(action or "").strip().lower()
    if value in ACTION_TRIGGERS:
        return value, ACTION_TRIGGERS[value]
    if allow_experimental and value in EXPERIMENTAL_TRIGGERS:
        return value, EXPERIMENTAL_TRIGGERS[value]
    supported = ", ".join(sorted(ACTION_TRIGGERS))
    raise WorkflowInputError(f"action must be one of: {supported}")


def _normalize_settings(settings: Any) -> Dict[str, Any]:
    if not isinstance(settings, dict):
        settings = {}
    return {
        "distill_lora_scale": _clamp_float(
            settings.get("distill_lora_scale"),
            SETTING_DEFAULTS["distill_lora_scale"],
            0.25,
            0.35,
        ),
        "dr34ml4y_lora_scale": _clamp_float(
            settings.get("dr34ml4y_lora_scale"),
            SETTING_DEFAULTS["dr34ml4y_lora_scale"],
            0.0,
            1.5,
        ),
        "video_cfg": _clamp_float(settings.get("video_cfg"), SETTING_DEFAULTS["video_cfg"], 1.0, 3.0),
        "audio_cfg": _clamp_float(settings.get("audio_cfg"), SETTING_DEFAULTS["audio_cfg"], 1.0, 10.0),
        "stage_one_steps": _clamp_int(settings.get("stage_one_steps"), SETTING_DEFAULTS["stage_one_steps"], 8, 30),
        "stage_two_steps": _clamp_int(settings.get("stage_two_steps"), SETTING_DEFAULTS["stage_two_steps"], 0, 8),
        "img_compression": _clamp_int(settings.get("img_compression"), SETTING_DEFAULTS["img_compression"], 1, 30),
        "i2v_strength": _clamp_float(settings.get("i2v_strength"), SETTING_DEFAULTS["i2v_strength"], 0.1, 1.0),
    }


def _stage_one_sigmas(steps: int) -> str:
    if steps <= 8:
        return "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    values = []
    for index in range(steps):
        t = index / max(1, steps - 1)
        value = (1.0 - t) ** 1.6
        values.append(max(0.0, min(1.0, value)))
    values[-1] = 0.0
    return ", ".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in values)


def _build_prompt(trigger: str, prompt: str) -> str:
    body = str(prompt or "").strip()
    if not body:
        raise WorkflowInputError("prompt is required")
    if body.lower().startswith(trigger.lower()):
        return body
    return f"{trigger}. {body}"


def _workflow_node(class_type: str, inputs: Dict[str, Any], title: Optional[str] = None) -> Dict[str, Any]:
    node: Dict[str, Any] = {"class_type": class_type, "inputs": inputs}
    if title:
        node["_meta"] = {"title": title}
    return node


def _create_core_workflow(args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    workflow: Dict[str, Dict[str, Any]] = {
        "1": _workflow_node("LoadImage", {"image": args["input_image_filename"], "upload": "image"}),
        "2": _workflow_node("CheckpointLoaderSimple", {"ckpt_name": LTX_CHECKPOINT_NAME}),
        "3": _workflow_node(
            "LTXAVTextEncoderLoader",
            {
                "text_encoder": LTX_TEXT_ENCODER_NAME,
                "ckpt_name": LTX_CHECKPOINT_NAME,
                "device": "default",
            },
        ),
        "4": _workflow_node("CLIPTextEncode", {"text": args["prompt"], "clip": ["3", 0]}),
        "5": _workflow_node("CLIPTextEncode", {"text": args["negative_prompt"], "clip": ["3", 0]}),
        "6": _workflow_node(
            "LTXVConditioning",
            {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": args["fps"]},
        ),
        "7": _workflow_node("LTXVAudioVAELoader", {"ckpt_name": LTX_CHECKPOINT_NAME}),
        "8": _workflow_node("LatentUpscaleModelLoader", {"model_name": LTX_UPSCALER_NAME}),
        "9": _workflow_node(
            "LoraLoaderModelOnly",
            {
                "model": ["2", 0],
                "lora_name": LTX_DISTILLED_LORA_NAME,
                "strength_model": args["settings"]["distill_lora_scale"],
            },
        ),
        "10": _workflow_node(
            "LoraLoaderModelOnly",
            {
                "model": ["9", 0],
                "lora_name": DR34ML4Y_LORA_NAME,
                "strength_model": args["settings"]["dr34ml4y_lora_scale"],
            },
        ),
        "30": _workflow_node(
            "EmptyLTXVLatentVideo",
            {
                "width": args["width"],
                "height": args["height"],
                "length": args["length"],
                "batch_size": 1,
            },
        ),
        "31": _workflow_node(
            "LTXVEmptyLatentAudio",
            {
                "audio_vae": ["7", 0],
                "frames_number": args["length"],
                "frame_rate": args["fps"],
                "batch_size": 1,
            },
        ),
        "32": _workflow_node(
            "LTXVPreprocess",
            {"image": ["1", 0], "img_compression": args["settings"]["img_compression"]},
        ),
        "33": _workflow_node(
            "LTXVImgToVideoConditionOnly",
            {
                "vae": ["2", 2],
                "image": ["32", 0],
                "latent": ["30", 0],
                "strength": args["settings"]["i2v_strength"],
                "bypass": False,
            },
        ),
        "34": _workflow_node("LTXVConcatAVLatent", {"video_latent": ["33", 0], "audio_latent": ["31", 0]}),
        "35": _workflow_node("RandomNoise", {"noise_seed": args["seed"]}),
        "36": _workflow_node(
            "CFGGuider",
            {
                "model": ["10", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "cfg": args["settings"]["video_cfg"],
            },
        ),
        "37": _workflow_node("KSamplerSelect", {"sampler_name": "euler_ancestral_cfg_pp"}),
        "38": _workflow_node("ManualSigmas", {"sigmas": _stage_one_sigmas(args["settings"]["stage_one_steps"])}),
        "39": _workflow_node(
            "SamplerCustomAdvanced",
            {
                "noise": ["35", 0],
                "guider": ["36", 0],
                "sampler": ["37", 0],
                "sigmas": ["38", 0],
                "latent_image": ["34", 0],
            },
        ),
        "40": _workflow_node("LTXVSeparateAVLatent", {"av_latent": ["39", 0]}),
    }

    decoded_video_samples: List[Any] = ["40", 0]
    decoded_audio_samples: List[Any] = ["40", 1]

    if args["settings"]["stage_two_steps"] > 0:
        workflow.update(
            {
                "41": _workflow_node(
                    "LTXVLatentUpsampler",
                    {"samples": ["40", 0], "upscale_model": ["8", 0], "vae": ["2", 2]},
                ),
                "42": _workflow_node(
                    "LTXVImgToVideoConditionOnly",
                    {
                        "vae": ["2", 2],
                        "image": ["1", 0],
                        "latent": ["41", 0],
                        "strength": 1,
                        "bypass": False,
                    },
                ),
                "43": _workflow_node(
                    "LTXVConcatAVLatent",
                    {"video_latent": ["42", 0], "audio_latent": ["40", 1]},
                ),
                "44": _workflow_node("RandomNoise", {"noise_seed": args["seed"]}),
                "45": _workflow_node(
                    "CFGGuider",
                    {
                        "model": ["10", 0],
                        "positive": ["6", 0],
                        "negative": ["6", 1],
                        "cfg": args["settings"]["video_cfg"],
                    },
                ),
                "46": _workflow_node("KSamplerSelect", {"sampler_name": "euler_cfg_pp"}),
                "47": _workflow_node("ManualSigmas", {"sigmas": LTX_STAGE_TWO_SIGMAS}),
                "48": _workflow_node(
                    "SamplerCustomAdvanced",
                    {
                        "noise": ["44", 0],
                        "guider": ["45", 0],
                        "sampler": ["46", 0],
                        "sigmas": ["47", 0],
                        "latent_image": ["43", 0],
                    },
                ),
                "49": _workflow_node("LTXVSeparateAVLatent", {"av_latent": ["48", 0]}),
            }
        )
        decoded_video_samples = ["49", 0]
        decoded_audio_samples = ["49", 1]

    workflow["50"] = _workflow_node(
        "VAEDecodeTiled",
        {
            "samples": decoded_video_samples,
            "vae": ["2", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 512,
            "temporal_overlap": 8,
        },
    )
    workflow["51"] = _workflow_node("LTXVAudioVAEDecode", {"samples": decoded_audio_samples, "audio_vae": ["7", 0]})
    workflow["52"] = _workflow_node(
        "VHS_VideoCombine",
        {
            "images": ["50", 0],
            "frame_rate": args["fps"],
            "filename_prefix": f"{WORKFLOW_ID}_{args['action']}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "loop_count": 0,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
    )
    workflow["53"] = _workflow_node(
        "VideoOutputBridge",
        {"filenames": ["52", 0], "label": WORKFLOW_ID},
    )
    if args["generate_audio"]:
        workflow["52"]["inputs"]["audio"] = ["51", 0]

    return workflow


def build_ltx23_dr34ml4y_job_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    workflow_id = str(payload.get("workflow_id") or WORKFLOW_ID).strip()
    if workflow_id != WORKFLOW_ID:
        raise WorkflowInputError(f"workflow_id must be {WORKFLOW_ID}")

    image_url = str(payload.get("image") or "").strip()
    if not image_url.startswith(("https://", "http://")):
        raise WorkflowInputError("image must be an HTTP(S) URL")

    action, trigger = _normalize_action(
        payload.get("action"),
        allow_experimental=bool(payload.get("allow_experimental_actions")),
    )
    settings = _normalize_settings(payload.get("settings"))
    fps = _normalize_fps(payload.get("fps"))
    duration = _normalize_duration(payload.get("duration"))
    width = _normalize_dimension(payload.get("width"), 704)
    height = _normalize_dimension(payload.get("height"), 1280)
    length = _resolve_length(duration, fps)
    seed = _resolve_seed(payload.get("seed"))
    input_image_filename = _infer_image_filename(image_url)
    prompt = _build_prompt(trigger, str(payload.get("prompt") or ""))
    negative_prompt = str(payload.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT).strip()
    generate_audio = bool(payload.get("generate_audio", True))

    normalized_args = {
        "workflow_id": WORKFLOW_ID,
        "workflow_version": WORKFLOW_VERSION,
        "action": action,
        "trigger": trigger,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps,
        "length": length,
        "seed": seed,
        "generate_audio": generate_audio,
        "settings": settings,
        "input_image_filename": input_image_filename,
    }
    workflow = _create_core_workflow(normalized_args)

    return {
        "workflow": workflow,
        "model_downloads": [
            {
                "url": image_url,
                "filename": input_image_filename,
                "relative_path": "input",
            }
        ],
        "workflow_id": WORKFLOW_ID,
        "workflow_version": WORKFLOW_VERSION,
        "action": action,
        "seed": seed,
        "settings": {
            **settings,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "length": length,
            "generate_audio": generate_audio,
            "trigger": trigger,
            "model": "ltx-2.3-22b-dev",
            "dr34ml4y_lora": DR34ML4Y_LORA_NAME,
        },
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "input_image_filename": input_image_filename,
        },
    }


def is_high_level_ltx23_request(payload: Dict[str, Any]) -> bool:
    if "workflow" in payload:
        return False
    workflow_id = str(payload.get("workflow_id") or WORKFLOW_ID).strip()
    return workflow_id == WORKFLOW_ID

