import importlib.util
import unittest
from pathlib import Path


def load_bridge_module():
    module_path = Path(__file__).parent / "vendor" / "ComfyUI-VideoOutputBridge" / "__init__.py"
    spec = importlib.util.spec_from_file_location("video_output_bridge_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


bridge_module = load_bridge_module()


class VideoOutputBridgeTests(unittest.TestCase):
    def test_absolute_output_root_path_uses_empty_subfolder(self):
        bridge = bridge_module.VideoOutputBridge()

        result = bridge.forward("/comfyui/output/ltx_i2v_00001.mp4", "ltx_i2v")

        self.assertEqual(
            result["ui"]["images"],
            [
                {
                    "filename": "ltx_i2v_00001.mp4",
                    "subfolder": "",
                    "type": "output",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
