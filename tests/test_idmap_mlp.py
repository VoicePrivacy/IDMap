import unittest

try:
    import torch
    from voice_anon.idmap.mlp import IDMapMLP, IDMapMLPLoss, IdentityVectorSampler
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "research extras (torch/numpy) are not installed")
class IdentityVectorSamplerTest(unittest.TestCase):
    def test_pcg64_mapping_is_deterministic_and_index_specific(self) -> None:
        sampler = IdentityVectorSampler(distribution="normal")
        first = sampler.sample(1234)
        repeated = sampler.sample(1234)
        different = sampler.sample(1235)
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, different))

    def test_mvn_has_zero_mean_and_unit_variance(self) -> None:
        vector = IdentityVectorSampler().sample(7)[0]
        self.assertAlmostEqual(vector.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(vector.std(unbiased=False).item(), 1.0, places=5)


@unittest.skipIf(torch is None, "research extras (torch/numpy) are not installed")
class IDMapMLPTest(unittest.TestCase):
    def test_non_512_backend_space(self) -> None:
        dimension = 192
        model = IDMapMLP(dimension=dimension).train()
        identity = IdentityVectorSampler(dimension=dimension).sample(range(4))
        auxiliary = torch.randn(4, dimension)
        prediction = model(identity, auxiliary)
        self.assertEqual(tuple(prediction.shape), (4, dimension))

    def test_forward_shape_and_finite_gradient(self) -> None:
        model = IDMapMLP(dimension=512).train()
        identity = IdentityVectorSampler().sample(range(4))
        auxiliary = torch.randn(4, 512)
        target = torch.randn(4, 512)
        prediction = model(identity, auxiliary)
        loss = IDMapMLPLoss(alpha=0.5)(prediction, target)
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (4, 512))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_equation_three_is_zero_for_identical_nonzero_vectors(self) -> None:
        vector = torch.randn(8, 512)
        loss = IDMapMLPLoss(alpha=0.5)(vector, vector)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_fixed_auxiliary_inference_is_reproducible(self) -> None:
        model = IDMapMLP().eval()
        sampler = IdentityVectorSampler(distribution="uniform")
        auxiliary = torch.randn(512)
        with torch.no_grad():
            first = model.generate_from_indices([1, 2, 3], auxiliary, sampler)
            second = model.generate_from_indices([1, 2, 3], auxiliary, sampler)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
