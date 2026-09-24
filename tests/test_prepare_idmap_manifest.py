import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_idmap_manifest.py"
SPEC = importlib.util.spec_from_file_location("prepare_idmap_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareManifestTest(unittest.TestCase):
    def test_session_identity_reused_and_distinct_speakers_separated(self) -> None:
        rows = [
            {"utterance_id": "a1", "text": "One", "session_id": "s", "local_speaker_id": "A"},
            {"utterance_id": "b1", "text": "Two", "session_id": "s", "local_speaker_id": "B"},
            {"utterance_id": "a2", "text": "Three", "session_id": "s", "local_speaker_id": "A"},
        ]
        first = MODULE.build_manifest(rows, mode="session", seed=7, pool_size=100)
        second = MODULE.build_manifest(rows, mode="session", seed=7, pool_size=100)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["anonymous_index"], first[2]["anonymous_index"])
        self.assertNotEqual(first[0]["anonymous_index"], first[1]["anonymous_index"])

    def test_utterance_mode_rotates_and_bad_output_is_rejected(self) -> None:
        rows = [
            {"utterance_id": "a", "text": "One"},
            {"utterance_id": "b", "text": "Two"},
        ]
        result = MODULE.build_manifest(rows, mode="utterance", seed=1, pool_size=2)
        self.assertNotEqual(result[0]["anonymous_index"], result[1]["anonymous_index"])
        rows[1]["output_relative_path"] = "../escape.wav"
        with self.assertRaisesRegex(ValueError, "unsafe output"):
            MODULE.build_manifest(rows, mode="utterance", seed=1, pool_size=2)


if __name__ == "__main__":
    unittest.main()
