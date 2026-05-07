# RunPod ComfyUI LTX-2.3 DR34ML4Y Worker

RunPod Serverless worker for CEL-180 manual NSFW image-to-video testing with LTX-2.3, native LTX audio, and the `DR34ML4Y_LT3X_v1` action LoRA.

This endpoint is intentionally staging/manual-test only. Do not route production CelebMaker WAN traffic to it.

## Contract

Submit with RunPod `/run`:

```json
{
  "input": {
    "workflow_id": "ltx23_dr34ml4y_i2v_v1",
    "image": "https://.../input.png",
    "prompt": "expanded action/audio prompt",
    "action": "missionary",
    "duration": 5,
    "width": 704,
    "height": 1280,
    "fps": 24,
    "seed": -1,
    "generate_audio": true,
    "negative_prompt": "low quality, blurry, subtitles, watermark, text, logo, distorted anatomy, extra limbs, bad hands, bad face, distorted sound, saturated sound, loud sound",
    "settings": {
      "distill_lora_scale": 0.3,
      "dr34ml4y_lora_scale": 1.0,
      "video_cfg": 2.5,
      "audio_cfg": 7.0
    }
  }
}
```

Poll with `/status/{job_id}`. Completed output is normalized to:

```json
{
  "video_url": "https://.../output.mp4",
  "seed": 123456789,
  "workflow_id": "ltx23_dr34ml4y_i2v_v1",
  "workflow_version": "v1",
  "action": "missionary",
  "settings": { "actual": "settings used" }
}
```

## Supported Actions

| action id | trigger |
| --- | --- |
| `missionary` | `m15510n4ry` |
| `doggy` | `d0gg1e` |
| `cowgirl` | `c0wg1rl` |
| `reverse-cowgirl` | `r3v3rs3_c0wg1rl` |
| `blowjob` | `bl0wj0b` |

`d0ubl3_bj` is kept out of the public v1 action surface.

## Workflow Settings

- Model path: LTX-2.3 dev checkpoint, not a distilled-only checkpoint.
- Native audio: enabled by default through LTX audio latent generation and audio VAE decode.
- Prompt enhancer: not used. The caller sends expanded prompts.
- Default dimensions: `704x1280`, divisible by 32 and portrait-friendly with lower VRAM pressure than `768x1280`.
- Default frame count: `duration * fps + 1`, rounded to `8n + 1`; 5 seconds at 24 fps is 121 frames.
- Default distill LoRA: `0.30`, clamped to `0.25-0.35`.
- Default DR34ML4Y LoRA: `1.0`.
- Default video CFG: `2.5`, clamped to `1.0-3.0`.
- Default audio CFG is accepted and recorded for the endpoint contract, but the current all-in-one Comfy graph uses the stable `CFGGuider` path from the deployed LTX worker. A follow-up split-loader/RuneXX graph migration should wire separate multimodal audio/video guider parameters.

## Model Cache

Attach a RunPod network volume at `/runpod-volume`. The handler checks `/runpod-volume/comfyui/...` before downloading and mirrors downloaded model files back into that cache.

Core files:

| file | Comfy path | source |
| --- | --- | --- |
| `ltx-2.3-22b-dev.safetensors` | `models/checkpoints/` | `https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors` |
| `gemma_3_12B_it.safetensors` | `models/text_encoders/` | `https://huggingface.co/GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn/resolve/main/gemma_3_12B_it_fp8_e4m3fn.safetensors` |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `models/loras/` | `https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | `models/latent_upscale_models/` | `https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| `DR34ML4Y_LTXXX_V1.safetensors` | `models/loras/` | `https://civitai.com/api/download/models/2913022` |

Kijai split LTX-2.3 files remain the preferred future production graph target:

- `diffusion_models/ltx-2.3-22b-dev_transformer_only_bf16.safetensors` or `diffusion_models/ltx-2-3-22b-dev_transformer_only_fp8_input_scaled.safetensors`
- `vae/LTX23_video_vae_bf16.safetensors`
- `vae/LTX23_audio_vae_bf16.safetensors`
- `text_encoders/ltx-2.3_text_projection_bf16.safetensors`

The v1 worker keeps the proven all-in-one checkpoint path to get a callable endpoint quickly without losing native audio support.

## Endpoint Settings

Initial RunPod Serverless settings:

- GPU: H100/A100 80GB class preferred for the dev checkpoint and portrait tests.
- Network volume: `nextmedia-ltx-core-cache-eur-is-3` (`36khvc3na6`) or a dedicated replacement with the same Comfy paths.
- Workers min/max: `0/1` for first testing to control spend.
- Idle timeout: 1800 seconds so model cache stays warm during a test batch.
- Execution timeout/TTL: allow at least 3600 seconds for cold prewarm and first canary.

## Smoke Commands

Prewarm:

```bash
python scripts/smoke_submit.py \
  --endpoint-id "$RUNPOD_LTX23_VIDEO_ENDPOINT_ID" \
  --prewarm \
  --timeout-seconds 3600
```

Run all five action tests:

```bash
python scripts/smoke_submit.py \
  --endpoint-id "$RUNPOD_LTX23_VIDEO_ENDPOINT_ID" \
  --image-url "https://example.com/synthetic-adult-start.png" \
  --action all \
  --verify-audio \
  --output-jsonl benchmarks/ltx23_dr34ml4y_first5.jsonl
```

The script reads `RUNPOD_API_KEY` or the saved `~/.runpod/config.toml` credential, submits via `/run`, polls `/status/{job_id}`, prints each `video_url`, and uses `ffprobe` for audio-track verification when available.

## Local Checks

```bash
python -m unittest -v
python -m compileall .
```

## Out Of Scope

- V2 custom/reference audio.
- Production CelebMaker routing.
- Replacing WaveSpeed WAN or existing RunPod WAN behavior.
