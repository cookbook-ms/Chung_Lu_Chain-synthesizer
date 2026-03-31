import pytest
import numpy as np
from powergrid_synth.deg_dist_optimizer import DegreeDistributionOptimizer


class TestDegreeDistributionOptimizer:

    @pytest.fixture()
    def optimizer(self):
        return DegreeDistributionOptimizer(verbose=False)

    def test_dgln_pdf_sums_to_one(self, optimizer):
        pdf = optimizer._dgln_pdf(100, alpha=1.5, beta=2.0)
        assert pdf.shape == (100,)
        assert abs(pdf.sum() - 1.0) < 1e-10

    def test_dgln_pdf_all_nonneg(self, optimizer):
        pdf = optimizer._dgln_pdf(50, alpha=1.0, beta=1.5)
        assert np.all(pdf >= 0)

    def test_dgln_pdf_invalid_params_returns_zeros(self, optimizer):
        pdf = optimizer._dgln_pdf(10, alpha=-1.0, beta=2.0)
        assert np.all(pdf == 0)

    def test_dpl_pdf_sums_to_one(self, optimizer):
        pdf = optimizer._dpl_pdf(100, gamma=2.5)
        assert pdf.shape == (100,)
        assert abs(pdf.sum() - 1.0) < 1e-10

    def test_dpl_pdf_all_nonneg(self, optimizer):
        pdf = optimizer._dpl_pdf(50, gamma=3.0)
        assert np.all(pdf >= 0)

    def test_optimize_dgln_returns_correct_structure(self, optimizer):
        params, pdf = optimizer.optimize(
            target_avg=3.5, max_deg=30, dist_type="dgln"
        )
        assert len(params) == 2  # alpha, beta
        assert isinstance(pdf, np.ndarray)
        assert abs(pdf.sum() - 1.0) < 1e-6

    def test_optimize_dpl_returns_correct_structure(self, optimizer):
        params, pdf = optimizer.optimize(
            target_avg=2.5, max_deg=20, dist_type="dpl"
        )
        assert len(params) == 1  # gamma
        assert isinstance(pdf, np.ndarray)

    def test_optimize_achieves_target_avg(self, optimizer):
        target = 3.0
        _, pdf = optimizer.optimize(
            target_avg=target, max_deg=40, dist_type="dgln"
        )
        degrees = np.arange(1, len(pdf) + 1)
        achieved_avg = np.dot(degrees, pdf)
        assert abs(achieved_avg - target) < 0.5  # generous tolerance
