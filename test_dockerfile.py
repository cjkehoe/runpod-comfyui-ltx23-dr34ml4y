from pathlib import Path
import unittest


class DockerfileStartupTests(unittest.TestCase):
    def test_dockerfile_executes_start_sh_directly(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            'CMD ["/start.sh"]',
            dockerfile,
        )

    def test_dockerfile_pins_supported_comfyui_release_for_ltx_2_3(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("git checkout v0.16.1", dockerfile)

    def test_dockerfile_vendors_video_output_bridge_node(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text(encoding="utf-8")

        self.assertIn(
            "COPY vendor/ComfyUI-VideoOutputBridge /comfyui/custom_nodes/ComfyUI-VideoOutputBridge",
            dockerfile,
        )

    def test_dockerfile_installs_kjnodes_for_runexx_ltx_reference_support(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("https://github.com/kijai/ComfyUI-KJNodes.git", dockerfile)

    def test_dockerfile_installs_comfy_aimdo_for_newer_comfyui(self):
        dockerfile = Path(__file__).with_name("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("pip install --no-cache-dir --break-system-packages comfy-aimdo", dockerfile)


if __name__ == "__main__":
    unittest.main()
