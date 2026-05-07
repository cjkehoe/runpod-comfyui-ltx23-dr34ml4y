import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load_bootstrap_module():
    module_path = Path(__file__).with_name("bootstrap.py")
    spec = importlib.util.spec_from_file_location("ltx_i2v_bootstrap_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


bootstrap = load_bootstrap_module()


class LtxI2vBootstrapTests(unittest.TestCase):
    def test_ensure_core_model_downloads_once_and_caches_to_network_volume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            comfy_root = temp_path / "comfyui"
            network_volume_root = temp_path / "runpod-volume"
            network_volume_root.mkdir(parents=True, exist_ok=True)
            os.environ["LTX_CHECKPOINT_URL"] = "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors"

            def fake_download(url: str, destination: Path) -> None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"core-model")

            with patch.object(bootstrap, "_download_file", side_effect=fake_download):
                bootstrap._ensure_core_model(
                    {
                        "filename": "ltx-2.3-22b-dev.safetensors",
                        "relative_path": "models/checkpoints",
                        "url_env": "LTX_CHECKPOINT_URL",
                    },
                    comfy_root,
                    network_volume_root,
                )

            runtime_model = comfy_root / "models" / "checkpoints" / "ltx-2.3-22b-dev.safetensors"
            cached_model = (
                network_volume_root / "comfyui" / "models" / "checkpoints" / "ltx-2.3-22b-dev.safetensors"
            )

            self.assertTrue(runtime_model.exists())
            self.assertTrue(cached_model.exists())
            self.assertEqual(runtime_model.read_bytes(), b"core-model")
            self.assertEqual(cached_model.read_bytes(), b"core-model")


if __name__ == "__main__":
    unittest.main()
