import unittest

try:
    import torch
    from voice_anon.idmap.diffusion import IDMapDiff, IDMapDiffEDM
    from voice_anon.idmap.mlp import IdentityVectorSampler
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "research extras (torch/numpy) are not installed")
class IDMapDiffTest(unittest.TestCase):
    def test_non_512_backend_space(self) -> None:
        dimension = 192
        model = IDMapDiff(dimension=dimension).train()
        identity = IdentityVectorSampler(dimension=dimension).sample(range(2))
        auxiliary = torch.randn(2, dimension)
        target = torch.randn(2, dimension)
        loss, _ = model.training_loss(identity, auxiliary, target)
        self.assertTrue(torch.isfinite(loss))

    def test_paper_schedule_endpoints(self) -> None:
        model = IDMapDiff()
        times = torch.tensor([0.0, 1.0])
        beta = model.beta_min + (model.beta_max - model.beta_min) * times
        self.assertTrue(torch.allclose(beta, torch.tensor([0.05, 20.0])))

    def test_weighted_score_loss_has_finite_gradient(self) -> None:
        model = IDMapDiff().train()
        identity = IdentityVectorSampler().sample(range(4))
        auxiliary = torch.randn(4, 512)
        target = torch.randn(4, 512)
        loss, metrics = model.training_loss(identity, auxiliary, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["unweighted_score_mse"]))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_interval_gamma_matches_integrated_linear_beta(self) -> None:
        model = IDMapDiff()
        start = torch.tensor([0.4])
        end = torch.tensor([0.6])
        expected_integral = (
            model.beta_min
            + 0.5 * (model.beta_max - model.beta_min) * (start + end)
        ) * (end - start)
        self.assertTrue(
            torch.allclose(
                model.gamma_interval(start, end),
                torch.exp(-0.5 * expected_integral),
            )
        )

    def test_sde_ml_coefficients_are_finite_at_all_five_steps(self) -> None:
        model = IDMapDiff()
        times = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
        coefficients = model.sde_ml_coefficients(times, 0.2)
        self.assertTrue(all(torch.isfinite(value).all() for value in coefficients))
        self.assertTrue((coefficients[2] >= 0.0).all())

    def test_five_step_generation_is_index_deterministic(self) -> None:
        model = IDMapDiff().eval()
        auxiliary = torch.randn(512)
        sampler = IdentityVectorSampler()
        first = model.generate_from_indices([7, 8], auxiliary, sampler, steps=5)
        second = model.generate_from_indices([7, 8], auxiliary, sampler, steps=5)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(tuple(first.shape), (2, 512))

    def test_edm_backend_normalization_round_trip(self) -> None:
        mean = torch.linspace(-2.0, 2.0, 32)
        std = torch.linspace(0.5, 2.0, 32)
        model = IDMapDiffEDM(dimension=32, data_mean=mean, data_std=std)
        values = torch.randn(3, 32) * std + mean
        self.assertTrue(torch.allclose(model.denormalize(model.normalize(values)), values))

    def test_edm_weighted_denoising_loss_has_finite_gradient(self) -> None:
        dimension = 32
        model = IDMapDiffEDM(dimension=dimension, geometry_weight=0.1).train()
        identity = IdentityVectorSampler(dimension=dimension).sample(range(4))
        auxiliary = torch.randn(4, dimension)
        target = torch.randn(4, dimension)
        sigma = torch.tensor([0.01, 0.1, 1.0, 10.0])
        loss, metrics = model.training_loss(
            identity,
            auxiliary,
            target,
            sigma=sigma,
            noise=torch.randn_like(target),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["unweighted_denoising_mse"]))
        self.assertTrue(torch.isfinite(metrics["native_geometry_loss"]))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_edm_heun_generation_is_deterministic(self) -> None:
        dimension = 32
        mean = torch.linspace(-1.0, 1.0, dimension)
        std = torch.linspace(0.5, 1.5, dimension)
        model = IDMapDiffEDM(
            dimension=dimension,
            data_mean=mean,
            data_std=std,
        ).eval()
        auxiliary = torch.randn(dimension)
        sampler = IdentityVectorSampler(dimension=dimension)
        first = model.generate_from_indices([7, 8], auxiliary, sampler, steps=6)
        second = model.generate_from_indices([7, 8], auxiliary, sampler, steps=6)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertEqual(tuple(first.shape), (2, dimension))


if __name__ == "__main__":
    unittest.main()
