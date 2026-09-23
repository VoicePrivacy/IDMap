import tempfile
from pathlib import Path
import unittest

try:
    import numpy as np
    from voice_anon.idmap.data import SpeakerEmbeddingCorpus
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "research extras (numpy) are not installed")
class SpeakerEmbeddingCorpusTest(unittest.TestCase):
    def test_dimension_is_inferred_from_backend_archive(self) -> None:
        embeddings = np.zeros((6, 192), dtype=np.float32)
        speaker_ids = np.repeat(["a", "b", "c"], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            np.savez(path, embeddings=embeddings, speaker_ids=speaker_ids)
            corpus = SpeakerEmbeddingCorpus.load(path)
        self.assertEqual(corpus.embeddings.shape, (6, 192))
        self.assertEqual(corpus.speaker_means.shape, (3, 192))

    def test_paper_batch_has_256_distinct_speaker_triplets(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(20 * 3, 512)).astype(np.float32)
        speaker_ids = np.repeat([f"speaker-{i}" for i in range(20)], 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            np.savez(path, embeddings=embeddings, speaker_ids=speaker_ids)
            corpus = SpeakerEmbeddingCorpus.load(path)
        identities, targets, auxiliaries = corpus.sample_paper_batch(
            rng, speakers_per_batch=16, auxiliary_utterances_per_speaker=16
        )
        self.assertEqual(identities.shape, (256,))
        self.assertEqual(targets.shape, (256, 512))
        self.assertEqual(auxiliaries.shape, (256, 512))
        for identity, auxiliary in zip(identities, auxiliaries, strict=True):
            own_indices = corpus.utterance_indices_by_speaker[int(identity)]
            self.assertFalse(any(np.array_equal(auxiliary, embeddings[i]) for i in own_indices))


if __name__ == "__main__":
    unittest.main()
