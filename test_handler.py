import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_handler_module(base_handler_path: Path):
    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = types.SimpleNamespace(start=lambda *args, **kwargs: None)
    sys.modules["runpod"] = fake_runpod

    os.environ["BASE_HANDLER_PATH"] = str(base_handler_path)

    module_path = Path(__file__).with_name("handler.py")
    spec = importlib.util.spec_from_file_location("ltx_i2v_handler_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_stub_base_handler(temp_dir: Path) -> Path:
    base_handler_path = temp_dir / "handler_base_stub.py"
    base_handler_path.write_text(
        "def handler(job):\n"
        "    return {'images': []}\n",
        encoding="utf-8",
    )
    return base_handler_path


class LtxI2vHandlerTests(unittest.TestCase):
    @staticmethod
    def empty_core_model_env():
        return {
            "LTX_CHECKPOINT_URL": "",
            "LTX_TEXT_ENCODER_URL": "",
            "LTX_DISTILLED_LORA_URL": "",
            "LTX_SPATIAL_UPSCALER_URL": "",
            "DR34ML4Y_LTX_LORA_URL": "",
        }

    @staticmethod
    def single_checkpoint_manifest():
        return {
            "core_models": [
                {
                    "filename": "ltx-2.3-22b-dev.safetensors",
                    "relative_path": "models/checkpoints",
                    "url_env": "LTX_CHECKPOINT_URL",
                }
            ]
        }

    def test_load_base_handler_can_import_sibling_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sibling_module = temp_path / "network_volume.py"
            sibling_module.write_text(
                "def is_network_volume_debug_enabled():\n"
                "    return False\n"
                "\n"
                "def run_network_volume_diagnostics():\n"
                "    return {}\n",
                encoding="utf-8",
            )

            base_handler_path = temp_path / "handler_base_stub.py"
            base_handler_path.write_text(
                "from network_volume import is_network_volume_debug_enabled\n"
                "\n"
                "def handler(job):\n"
                "    return {'network_volume_debug': is_network_volume_debug_enabled()}\n",
                encoding="utf-8",
            )

            handler = load_handler_module(base_handler_path)

            self.assertTrue(callable(handler.BASE_HANDLER))
            self.assertEqual(handler.BASE_HANDLER({"input": {}}), {"network_volume_debug": False})

    def test_handler_downloads_runtime_assets_and_rewrites_public_video_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            handler = load_handler_module(write_stub_base_handler(temp_path))
            handler.COMFY_ROOT = temp_path / "comfyui"
            handler.NETWORK_VOLUME_ROOT = temp_path / "runpod-volume"
            handler.OUTPUT_PUBLIC_BASE = "https://cdn.example.com/videos"
            handler.OUTPUT_BUCKET_NAME = "videos"

            captured_job = {}

            def fake_base_handler(job):
                captured_job["value"] = job
                return {
                    "images": [
                        {
                            "filename": "ltx_i2v_00001.mp4",
                            "type": "s3_url",
                            "data": "https://account.r2.cloudflarestorage.com/videos/job-123/ltx_i2v_00001.mp4",
                        }
                    ]
                }

            def fake_download(url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(f"downloaded:{url}".encode("utf-8"))

            handler.BASE_HANDLER = fake_base_handler

            with patch.dict(os.environ, self.empty_core_model_env(), clear=False), patch.object(
                handler, "_ensure_core_models", return_value=set(), create=True
            ), patch.object(
                handler, "_download_file", side_effect=fake_download, create=True
            ), patch.object(handler, "_refresh_comfy_file_cache", return_value=None, create=True):
                result = handler.handler(
                    {
                        "input": {
                            "workflow": {"52": {"class_type": "VHS_VideoCombine"}},
                            "model_downloads": [
                                {
                                    "url": "https://cdn.example.com/start-frame.png",
                                    "filename": "start-frame.png",
                                    "relative_path": "input",
                                },
                                {
                                    "url": "https://cdn.example.com/motion.safetensors",
                                    "filename": "motion.safetensors",
                                    "relative_path": "models/loras",
                                },
                            ],
                        }
                    }
                )

            self.assertTrue((handler.COMFY_ROOT / "input" / "start-frame.png").exists())
            self.assertTrue((handler.COMFY_ROOT / "models" / "loras" / "motion.safetensors").exists())
            self.assertNotIn("model_downloads", captured_job["value"]["input"])
            self.assertEqual(result["video_url"], "https://cdn.example.com/videos/job-123/ltx_i2v_00001.mp4")
            self.assertEqual(
                result["images"][0]["data"],
                "https://cdn.example.com/videos/job-123/ltx_i2v_00001.mp4",
            )

    def test_normalize_video_output_prefers_audio_variant_as_primary_video_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            handler = load_handler_module(write_stub_base_handler(temp_path))
            handler.OUTPUT_PUBLIC_BASE = "https://cdn.example.com/videos"
            handler.OUTPUT_BUCKET_NAME = "videos"

            result = handler._normalize_video_output(
                {
                    "images": [
                        {
                            "filename": "ltx_i2v_00001.mp4",
                            "type": "s3_url",
                            "data": "https://account.r2.cloudflarestorage.com/videos/job-123/ltx_i2v_00001.mp4",
                        },
                        {
                            "filename": "ltx_i2v_00001-audio.mp4",
                            "type": "s3_url",
                            "data": "https://account.r2.cloudflarestorage.com/videos/job-123/ltx_i2v_00001-audio.mp4",
                        },
                    ]
                }
            )

            self.assertEqual(
                result["video_url"],
                "https://cdn.example.com/videos/job-123/ltx_i2v_00001-audio.mp4",
            )

    def test_handler_reuses_cached_runtime_assets_from_network_volume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            handler = load_handler_module(write_stub_base_handler(temp_path))
            handler.COMFY_ROOT = temp_path / "comfyui"
            handler.NETWORK_VOLUME_ROOT = temp_path / "runpod-volume"

            cached_model = (
                handler.NETWORK_VOLUME_ROOT / "comfyui" / "models" / "loras" / "cached-motion.safetensors"
            )
            cached_model.parent.mkdir(parents=True, exist_ok=True)
            cached_model.write_bytes(b"cached-model")

            handler.BASE_HANDLER = lambda job: {"images": []}

            with patch.dict(os.environ, self.empty_core_model_env(), clear=False), patch.object(
                handler, "_ensure_core_models", return_value=set(), create=True
            ), patch.object(
                handler,
                "_download_file",
                side_effect=AssertionError("cached files should not redownload"),
                create=True,
            ), patch.object(handler, "_refresh_comfy_file_cache", return_value=None, create=True):
                handler.handler(
                    {
                        "input": {
                            "workflow": {"52": {"class_type": "VHS_VideoCombine"}},
                            "model_downloads": [
                                {
                                    "url": "https://cdn.example.com/cached-motion.safetensors",
                                    "filename": "cached-motion.safetensors",
                                    "relative_path": "models/loras",
                                }
                            ],
                        }
                    }
                )

            runtime_model = handler.COMFY_ROOT / "models" / "loras" / "cached-motion.safetensors"
            self.assertTrue(runtime_model.exists())
            self.assertEqual(runtime_model.read_bytes(), b"cached-model")

    def test_handler_downloads_missing_core_models_before_base_handler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            handler = load_handler_module(write_stub_base_handler(temp_path))
            handler.COMFY_ROOT = temp_path / "comfyui"
            handler.NETWORK_VOLUME_ROOT = temp_path / "runpod-volume"
            handler.BASE_HANDLER = lambda job: {"images": []}

            def fake_download(url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(f"downloaded:{url}".encode("utf-8"))

            with patch.dict(
                os.environ,
                {"LTX_CHECKPOINT_URL": "https://example.com/ltx-2.3-22b-dev.safetensors"},
                clear=False,
            ), patch.object(handler, "_load_manifest", return_value=self.single_checkpoint_manifest(), create=True), patch.object(
                handler, "_download_file", side_effect=fake_download, create=True
            ), patch.object(
                handler, "_refresh_comfy_file_cache", return_value=None, create=True
            ):
                handler.handler({"input": {"workflow": {"52": {"class_type": "VHS_VideoCombine"}}}})

            runtime_model = handler.COMFY_ROOT / "models" / "checkpoints" / "ltx-2.3-22b-dev.safetensors"
            self.assertTrue(runtime_model.exists())
            self.assertEqual(
                runtime_model.read_bytes(),
                b"downloaded:https://example.com/ltx-2.3-22b-dev.safetensors",
            )

    def test_handler_supports_explicit_prewarm_without_invoking_base_handler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            handler = load_handler_module(write_stub_base_handler(temp_path))
            handler.COMFY_ROOT = temp_path / "comfyui"
            handler.NETWORK_VOLUME_ROOT = temp_path / "runpod-volume"

            def fake_download(url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(f"downloaded:{url}".encode("utf-8"))

            handler.BASE_HANDLER = lambda job: (_ for _ in ()).throw(
                AssertionError("prewarm requests should not invoke the base handler")
            )

            with patch.dict(
                os.environ,
                {"LTX_CHECKPOINT_URL": "https://example.com/ltx-2.3-22b-dev.safetensors"},
                clear=False,
            ), patch.object(handler, "_load_manifest", return_value=self.single_checkpoint_manifest(), create=True), patch.object(
                handler, "_download_file", side_effect=fake_download, create=True
            ), patch.object(
                handler, "_refresh_comfy_file_cache", return_value=None, create=True
            ):
                result = handler.handler({"input": {"action": "prewarm_core_models"}})

            self.assertTrue(result["ok"])
            self.assertEqual(result["mode"], "prewarm_core_models")
            self.assertIn("models/checkpoints", result["downloaded_relative_paths"])
            self.assertTrue(
                (handler.COMFY_ROOT / "models" / "checkpoints" / "ltx-2.3-22b-dev.safetensors").exists()
            )

    def test_handler_redownloads_corrupt_cached_core_model_when_size_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            handler = load_handler_module(write_stub_base_handler(temp_path))
            handler.COMFY_ROOT = temp_path / "comfyui"
            handler.NETWORK_VOLUME_ROOT = temp_path / "runpod-volume"
            handler.BASE_HANDLER = lambda job: {"images": []}

            cached_model = (
                handler.NETWORK_VOLUME_ROOT / "comfyui" / "models" / "checkpoints" / "ltx-2.3-22b-dev.safetensors"
            )
            cached_model.parent.mkdir(parents=True, exist_ok=True)
            cached_model.write_bytes(b"bad-cache")

            def fake_download(url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"fresh-good-cache")

            with patch.dict(
                os.environ,
                {"LTX_CHECKPOINT_URL": "https://example.com/ltx-2.3-22b-dev.safetensors"},
                clear=False,
            ), patch.object(handler, "_load_manifest", return_value=self.single_checkpoint_manifest(), create=True), patch.object(
                handler, "_get_expected_file_size", return_value=len(b"fresh-good-cache"), create=True
            ), patch.object(
                handler, "_download_file", side_effect=fake_download, create=True
            ), patch.object(handler, "_refresh_comfy_file_cache", return_value=None, create=True):
                handler.handler({"input": {"action": "prewarm_core_models"}})

            runtime_model = handler.COMFY_ROOT / "models" / "checkpoints" / "ltx-2.3-22b-dev.safetensors"
            self.assertEqual(runtime_model.read_bytes(), b"fresh-good-cache")
            self.assertEqual(cached_model.read_bytes(), b"fresh-good-cache")


if __name__ == "__main__":
    unittest.main()
