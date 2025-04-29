import unittest
import numpy as np
from prism.permutation_inference import permutation_analysis
import re


class TestPermutationAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n_samples = 36
        self.n_elements_per_sample = 1000 # Reduced for faster tests
        self.n_features = 3
        self.n_permutations = 100 # Reduced for faster tests
        self.n_exchangeability_groups = 2
        self.random_state = 42

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Generate test data (null hypothesis - no real effect)
        self.simulated_data = np.random.randn(self.n_samples, self.n_elements_per_sample)
        # Generate a simple design matrix (e.g., intercept + 2 group indicators)
        # Ensure it's not rank deficient for F/G tests if needed
        self.simulated_design = np.random.randn(self.n_samples, self.n_features)
        # Add intercept
        # self.simulated_design = np.column_stack([
        #     np.ones(self.n_samples),
        #     np.random.randint(0, 2, size=(self.n_samples, self.n_features - 1))
        # ])


        # --- Contrasts ---
        # Case 1: 1D contrast
        self.contrast_1d = np.zeros(self.n_features)
        self.contrast_1d[0] = 1

        # Case 2: Pseudo-1D contrast
        self.contrast_pseudo_1d = self.contrast_1d.reshape(1, -1) # [[1., 0., 0.]]

        # Case 3 & 5: 2D contrast (ensure it's compatible with design rank)
        self.contrast_2d = np.zeros((2, self.n_features))
        self.contrast_2d[0, 0] = 1
        self.contrast_2d[1, 1] = 1

        # --- Exchangeability Matrix ---
        # Ensure consistent group sizes for simplicity in setup
        group_size = self.n_samples // self.n_exchangeability_groups
        remainder = self.n_samples % self.n_exchangeability_groups
        groups = []
        for i in range(self.n_exchangeability_groups):
            size = group_size + (1 if i < remainder else 0)
            groups.append(np.ones(size) * (i + 1))
        self.simulated_exchangeability_matrix = np.concatenate(groups)
        np.random.shuffle(self.simulated_exchangeability_matrix) # Shuffle to make it less trivial

        # --- Common parameters ---
        self.common_params = {
            "data": self.simulated_data,
            "design": self.simulated_design,
            "n_permutations": self.n_permutations,
            "random_state": self.random_state,
            "two_tailed": True, # Adjust as needed per stat function
            "within": False, # Default, override when needed
            "whole": False, # Default, override when needed
            "accel_tail": True,
            "exchangeability_matrix": None # Default, override when needed
        }

    def _check_outputs(self, unc_p, fdr_p, fwe_p, test_name):
        """Helper to check output shapes and basic significance levels."""
        expected_shape = (self.n_elements_per_sample,)
        self.assertEqual(unc_p.shape, expected_shape, f"{test_name}: Incorrect shape for uncorrected p-values")
        self.assertEqual(fdr_p.shape, expected_shape, f"{test_name}: Incorrect shape for FDR p-values")
        self.assertEqual(fwe_p.shape, expected_shape, f"{test_name}: Incorrect shape for FWE p-values")

        # Optional: Basic sanity check for significance under null hypothesis
        # Since data is random, expect ~5% significance at alpha=0.05 for uncorrected p-values
        # FWE/FDR should be much lower. Check that it's not excessively high.
        significance_threshold = 0.05
        # Allow slightly higher proportion due to randomness, especially with fewer elements/perms
        max_proportion = 0.10
        max_significant = int(max_proportion * self.n_elements_per_sample)

        for p_values, p_type in [(unc_p, "uncorrected"),
                                (fdr_p, "FDR-corrected"),
                                (fwe_p, "FWE-corrected")]:
            n_significant = np.sum(p_values < significance_threshold)
            # For corrected p-values, we expect very few significant results under the null
            adjusted_max_significant = max_significant if p_type == "uncorrected" else max(10, int(0.02 * self.n_elements_per_sample)) # Allow a few spurious hits

            self.assertLessEqual(
                n_significant,
                adjusted_max_significant,
                f"{test_name}: Too many significant results for {p_type} p-values: "
                f"{n_significant} > {adjusted_max_significant} (threshold={significance_threshold})"
            )
            # Check p-values are in valid range [0, 1]
            self.assertTrue(np.all(p_values >= 0) and np.all(p_values <= 1),
                            f"{test_name}: {p_type} p-values out of range [0, 1]")


    # --- Test Cases ---

    def test_case1_1d_contrast_t_stat(self):
        """1: Test 1D contrast with t-statistic."""
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_1d,
            "stat_function": 'auto',
        })
        results = permutation_analysis(**params)
        unc_p, fdr_p, fwe_p = results.tstat_uncp, results.tstat_fdrp, results.tstat_fwep
        self._check_outputs(unc_p, fdr_p, fwe_p, "Case 1 (1D t-stat)")

    def test_case2_pseudo_1d_contrast_t_stat(self):
        """2: Test pseudo-1D contrast ([[1,0,0]]) with t-statistic."""
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_pseudo_1d,
            "stat_function": 'auto',
        })
        # Expect this to work like the 1D case
        results = permutation_analysis(**params)
        unc_p, fdr_p, fwe_p = results.tstat_uncp, results.tstat_fdrp, results.tstat_fwep
        self._check_outputs(unc_p, fdr_p, fwe_p, "Case 2 (Pseudo-1D t-stat)")

    def test_case3_2d_contrast_F_stat(self):
        """3: Test 2D contrast (ANOVA-like) with F-statistic."""
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_2d,
            "stat_function": 'auto',
            "two_tailed": False, # F-stat is typically one-tailed
            "f_contrast_indices": np.ones(self.contrast_2d.shape[0]),
            "f_only": True
        })
        results = permutation_analysis(**params)
        unc_p, fdr_p, fwe_p = results.fstat_uncp_f, results.fstat_fdrp_f, results.fstat_fwep_f
        self._check_outputs(unc_p, fdr_p, fwe_p, "Case 3 (2D F-stat)")

    def test_case4_1d_contrast_aspinwelch_exchangeability(self):
        """4: Test 1D contrast with Aspin-Welch V and exchangeability blocks."""
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_1d,
            "stat_function": 'auto',
            "exchangeability_matrix": self.simulated_exchangeability_matrix,
            "within": True,
            "vg_auto": True
        })
        results = permutation_analysis(**params)
        unc_p, fdr_p, fwe_p = results.vstat_uncp, results.vstat_fdrp, results.vstat_fwep
        self._check_outputs(unc_p, fdr_p, fwe_p, "Case 4 (1D Aspin-Welch Exch.)")

    def test_case5_2d_contrast_G_stat_exchangeability(self):
        """5: Test 2D contrast with G-statistic and exchangeability blocks."""
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_2d,
            "stat_function": 'auto',
            "exchangeability_matrix": self.simulated_exchangeability_matrix,
            "within": True, # Crucial for using exchangeability blocks
            "two_tailed": False, # Assume G is one-tailed like F
            "vg_auto": True,
            "f_contrast_indices": np.ones(self.contrast_2d.shape[0]),
            "f_only": True
        })
        results = permutation_analysis(**params)
        unc_p, fdr_p, fwe_p = results.gstat_uncp_f, results.gstat_fdrp_f, results.gstat_fwep_f
        self._check_outputs(unc_p, fdr_p, fwe_p, "Case 5 (2D G-stat Exch.)")

    # --- Error Cases ---

    def test_error_contrast_mismatch_design(self):
        """6a: Error when contrast columns don't match design columns."""
        bad_contrast = np.array([1, 0]) # Only 2 elements, design has 3 features
        params = self.common_params.copy()
        params.update({
            "contrast": bad_contrast,
            "stat_function": 'auto',
        })
        expected_pattern = re.compile(r"Contrast dimensions.*design matrix", re.IGNORECASE)
        with self.assertRaisesRegex(ValueError, expected_pattern):
             permutation_analysis(**params)

    def test_error_exchangeability_mismatch_samples(self):
        """6b: Error when exchangeability matrix length doesn't match samples."""
        bad_exchangeability = self.simulated_exchangeability_matrix[:-1] # One element short
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_1d,
            "stat_function": 'auto', # Or any using exchangeability
            "exchangeability_matrix": bad_exchangeability,
            "within": True
        })
        with self.assertRaisesRegex(ValueError, r"Exchangeability matrix length.*number of samples"):
             permutation_analysis(**params)

    def test_error_invalid_n_permutations(self):
        """6c: Error when n_permutations is non-positive."""
        params = self.common_params.copy()
        params.update({
            "contrast": self.contrast_1d,
            "stat_function": 'auto',
            "n_permutations": 0
        })
        with self.assertRaisesRegex(ValueError, r"Number of permutations must be positive"):
             permutation_analysis(**params)

        params["n_permutations"] = -100
        with self.assertRaisesRegex(ValueError, r"Number of permutations must be positive"):
             permutation_analysis(**params)


if __name__ == '__main__':
    unittest.main()
