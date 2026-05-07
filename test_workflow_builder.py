import unittest

from workflow_builder import (
    ACTION_TRIGGERS,
    DR34ML4Y_LORA_NAME,
    WORKFLOW_ID,
    WORKFLOW_VERSION,
    WorkflowInputError,
    build_ltx23_dr34ml4y_job_input,
)


class WorkflowBuilderTests(unittest.TestCase):
    def test_builds_high_level_ltx23_dr34ml4y_payload(self):
        payload = build_ltx23_dr34ml4y_job_input(
            {
                "workflow_id": WORKFLOW_ID,
                "image": "https://cdn.example.com/start.png",
                "prompt": "realistic motion with quiet room audio",
                "action": "missionary",
                "duration": 5,
                "width": 704,
                "height": 1280,
                "fps": 24,
                "seed": -1,
                "generate_audio": True,
                "settings": {
                    "distill_lora_scale": 0.9,
                    "video_cfg": 9,
                },
            }
        )

        self.assertEqual(payload["workflow_id"], WORKFLOW_ID)
        self.assertEqual(payload["workflow_version"], WORKFLOW_VERSION)
        self.assertEqual(payload["action"], "missionary")
        self.assertEqual(payload["settings"]["trigger"], "m15510n4ry")
        self.assertEqual(payload["settings"]["length"], 121)
        self.assertEqual(payload["settings"]["distill_lora_scale"], 0.35)
        self.assertEqual(payload["settings"]["video_cfg"], 3.0)
        self.assertIn("audio", payload["workflow"]["52"]["inputs"])
        self.assertEqual(payload["workflow"]["10"]["inputs"]["lora_name"], DR34ML4Y_LORA_NAME)
        self.assertTrue(payload["metadata"]["prompt"].startswith("m15510n4ry. "))

    def test_supports_exact_required_action_surface(self):
        self.assertEqual(
            set(ACTION_TRIGGERS),
            {"missionary", "doggy", "cowgirl", "reverse-cowgirl", "blowjob"},
        )

    def test_rejects_unknown_actions(self):
        with self.assertRaises(WorkflowInputError):
            build_ltx23_dr34ml4y_job_input(
                {
                    "image": "https://cdn.example.com/start.png",
                    "prompt": "motion",
                    "action": "unsupported",
                }
            )


if __name__ == "__main__":
    unittest.main()
