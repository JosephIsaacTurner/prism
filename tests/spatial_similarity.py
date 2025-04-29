import unittest
import numpy as np
from prism.datasets import Dataset
from prism.spatial_similarity import spatial_similarity_permutation_analysis
from prism.stats import t
import numpy.testing as npt
from sklearn.utils import Bunch


class TestSpatialSimilarityAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up common simulation data for all tests."""
        np.random.seed(42)
        cls.n_samples = 36
        cls.n_elements = 100 # Features
        cls.n_features_design = 3 # Regressors in design matrix
        cls.n_perms = 250

        # Data/Design/Contrast 1
        cls.data1 = np.random.randn(cls.n_samples, cls.n_elements)
        cls.design1 = np.random.randn(cls.n_samples, cls.n_features_design)
        cls.design1[:, 0] = 1 # Intercept
        cls.contrast1 = np.zeros(cls.n_features_design); cls.contrast1[1] = 1 # Contrast for 2nd regressor

        # Data/Design/Contrast 2
        cls.data2 = np.random.randn(cls.n_samples, cls.n_elements)
        cls.design2 = cls.design1.copy()
        cls.contrast2 = cls.contrast1.copy()

        # Data/Design/Contrast 3
        cls.data3 = np.random.randn(cls.n_samples, cls.n_elements)
        cls.design3 = cls.design1.copy()
        cls.contrast3 = cls.contrast1.copy()

        # Reference Maps (1D arrays)
        cls.ref1 = np.random.randn(cls.n_elements)
        cls.ref2 = np.random.randn(cls.n_elements)

        # Dataset instances (using dummy stat function 't')
        cls.ds1 = Dataset(data=cls.data1, design=cls.design1, contrast=cls.contrast1,
                          stat_function=t, n_permutations=cls.n_perms, random_state=42)
        cls.ds2 = Dataset(data=cls.data2, design=cls.design2, contrast=cls.contrast2,
                          stat_function=t, n_permutations=cls.n_perms + 10, random_state=37) # Different n_perms
        cls.ds3 = Dataset(data=cls.data3, design=cls.design3, contrast=cls.contrast3,
                          stat_function=t, n_permutations=cls.n_perms, random_state=11)

    # --- Test Cases ---

    def test_case_1_two_datasets_no_refs(self):
        """Case 1: 2 Datasets, 0 Reference Maps."""
        results = spatial_similarity_permutation_analysis(
            datasets=[self.ds1, self.ds2], reference_maps=None, two_tailed=True)
        self.assertIsInstance(results, Bunch)
        expected_perms = min(self.ds1.n_permutations, self.ds2.n_permutations)

        # Check DS-DS results
        self.assertIsNotNone(results['corr_matrix_ds_ds'])
        self.assertEqual(results['corr_matrix_ds_ds'].shape, (2, 2))
        self.assertIsNotNone(results['p_matrix_ds_ds'])
        self.assertEqual(results['p_matrix_ds_ds'].shape, (2, 2))
        self.assertTrue(np.isnan(results['p_matrix_ds_ds'][0, 0])) # Diagonal NaN
        self.assertTrue(np.isnan(results['p_matrix_ds_ds'][1, 1]))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ds'])
        self.assertEqual(results['corr_matrix_perm_ds_ds'].shape, (expected_perms, 2, 2))

        # Check DS-Ref results (should be None)
        self.assertIsNone(results['corr_matrix_ds_ref'])
        self.assertIsNone(results['p_matrix_ds_ref'])
        self.assertIsNone(results['corr_matrix_perm_ds_ref'])

    def test_case_2_three_datasets_no_refs(self):
        """Case 2: 3 Datasets, 0 Reference Maps."""
        results = spatial_similarity_permutation_analysis(
            datasets=[self.ds1, self.ds2, self.ds3], reference_maps=None, two_tailed=True)
        self.assertIsInstance(results, Bunch)
        expected_perms = min(self.ds1.n_permutations, self.ds2.n_permutations, self.ds3.n_permutations)

        # Check DS-DS results
        self.assertIsNotNone(results['corr_matrix_ds_ds'])
        self.assertEqual(results['corr_matrix_ds_ds'].shape, (3, 3))
        self.assertIsNotNone(results['p_matrix_ds_ds'])
        self.assertEqual(results['p_matrix_ds_ds'].shape, (3, 3))
        self.assertTrue(all(np.isnan(np.diag(results['p_matrix_ds_ds'])))) # Check diagonal NaNs
        self.assertIsNotNone(results['corr_matrix_perm_ds_ds'])
        self.assertEqual(results['corr_matrix_perm_ds_ds'].shape, (expected_perms, 3, 3))

        # Check DS-Ref results (should be None)
        self.assertIsNone(results['corr_matrix_ds_ref'])
        self.assertIsNone(results['p_matrix_ds_ref'])
        self.assertIsNone(results['corr_matrix_perm_ds_ref'])

    def test_case_3_one_dataset_one_ref(self):
        """Case 3: 1 Dataset, 1 Reference Map."""
        # Note: Pass dataset as a list for consistency, even if single
        results = spatial_similarity_permutation_analysis(
            datasets=[self.ds1], reference_maps=self.ref1, two_tailed=True)
        self.assertIsInstance(results, Bunch)
        expected_perms = self.ds1.n_permutations

        # Check DS-DS results (should be [[1.0]] for corr, None for p/perm)
        self.assertIsNotNone(results['corr_matrix_ds_ds'])
        npt.assert_array_equal(results['corr_matrix_ds_ds'], np.array([[1.0]])) # Updated assertion
        self.assertIsNone(results['p_matrix_ds_ds']) # No permutations run for DS-DS
        self.assertIsNone(results['corr_matrix_perm_ds_ds'])

        # Check DS-Ref results
        self.assertIsNotNone(results['corr_matrix_ds_ref'])
        self.assertEqual(results['corr_matrix_ds_ref'].shape, (1, 1))
        self.assertIsNotNone(results['p_matrix_ds_ref'])
        self.assertEqual(results['p_matrix_ds_ref'].shape, (1, 1))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ref'])
        self.assertEqual(results['corr_matrix_perm_ds_ref'].shape, (expected_perms, 1, 1))

    def test_case_4_one_dataset_two_refs(self):
        """Case 4: 1 Dataset, 2 Reference Maps."""
        results = spatial_similarity_permutation_analysis(
            datasets=[self.ds1], reference_maps=[self.ref1, self.ref2], two_tailed=True)
        self.assertIsInstance(results, Bunch)
        expected_perms = self.ds1.n_permutations

        # Check DS-DS results (should be [[1.0]] for corr, None for p/perm)
        self.assertIsNotNone(results['corr_matrix_ds_ds'])
        npt.assert_array_equal(results['corr_matrix_ds_ds'], np.array([[1.0]])) # Updated assertion
        self.assertIsNone(results['p_matrix_ds_ds'])
        self.assertIsNone(results['corr_matrix_perm_ds_ds'])

        # Check DS-Ref results
        self.assertIsNotNone(results['corr_matrix_ds_ref'])
        self.assertEqual(results['corr_matrix_ds_ref'].shape, (1, 2))
        self.assertIsNotNone(results['p_matrix_ds_ref'])
        self.assertEqual(results['p_matrix_ds_ref'].shape, (1, 2))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ref'])
        self.assertEqual(results['corr_matrix_perm_ds_ref'].shape, (expected_perms, 1, 2))

    def test_case_5_two_datasets_two_refs(self):
        """Case 5: 2 Datasets, 2 Reference Maps."""
        results = spatial_similarity_permutation_analysis(
            datasets=[self.ds1, self.ds2], reference_maps=[self.ref1, self.ref2], two_tailed=True)
        self.assertIsInstance(results, Bunch)
        expected_perms = min(self.ds1.n_permutations, self.ds2.n_permutations)

        # Check DS-DS results
        self.assertIsNotNone(results['corr_matrix_ds_ds'])
        self.assertEqual(results['corr_matrix_ds_ds'].shape, (2, 2))
        self.assertIsNotNone(results['p_matrix_ds_ds'])
        self.assertEqual(results['p_matrix_ds_ds'].shape, (2, 2))
        self.assertTrue(all(np.isnan(np.diag(results['p_matrix_ds_ds']))))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ds'])
        self.assertEqual(results['corr_matrix_perm_ds_ds'].shape, (expected_perms, 2, 2))

        # Check DS-Ref results
        self.assertIsNotNone(results['corr_matrix_ds_ref'])
        self.assertEqual(results['corr_matrix_ds_ref'].shape, (2, 2))
        self.assertIsNotNone(results['p_matrix_ds_ref'])
        self.assertEqual(results['p_matrix_ds_ref'].shape, (2, 2))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ref'])
        self.assertEqual(results['corr_matrix_perm_ds_ref'].shape, (expected_perms, 2, 2))

    def test_case_6_one_dataset_no_refs_should_warn_and_return_defaults(self):
        """Case 6: 1 Dataset, 0 Refs (should warn and return Bunch of Nones)."""
        with self.assertWarnsRegex(UserWarning, "Insufficient inputs"):
             results = spatial_similarity_permutation_analysis(
                 datasets=[self.ds1], reference_maps=None, two_tailed=True) # Use list

        # Expect the default Bunchionary with None values because setup failed
        self.assertIsInstance(results, Bunch)
        self.assertTrue(all(v is None for v in results.values())) # Updated assertion

    def test_case_7_zero_datasets_one_ref_should_raise_error(self):
        """Case 7: 0 Datasets, 1 Reference Map (should raise ValueError)."""
        # Expect ValueError from _setup_and_validate, re-raised by run_analysis/wrapper
        with self.assertRaisesRegex(ValueError, "Dataset list cannot be empty"):
            spatial_similarity_permutation_analysis(
                datasets=[], reference_maps=[self.ref1], two_tailed=True)

    def test_case_8_custom_compare_func(self):
        """Case 8: Test with a custom comparison function (dot product)."""
        def simple_dot_product(vec1, vec2):
            return np.dot(vec1, vec2)

        results = spatial_similarity_permutation_analysis(
            datasets=[self.ds1, self.ds2],
            reference_maps=[self.ref1],
            two_tailed=True, # P-value interpretation depends on function range
            compare_func=simple_dot_product
        )
        self.assertIsInstance(results, Bunch)
        expected_perms = min(self.ds1.n_permutations, self.ds2.n_permutations)

        # Check shapes - values will depend on the dummy stat function and dot product
        self.assertIsNotNone(results['corr_matrix_ds_ds'])
        self.assertEqual(results['corr_matrix_ds_ds'].shape, (2, 2))
        self.assertIsNotNone(results['p_matrix_ds_ds'])
        self.assertEqual(results['p_matrix_ds_ds'].shape, (2, 2))
        self.assertTrue(all(np.isnan(np.diag(results['p_matrix_ds_ds']))))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ds'])
        self.assertEqual(results['corr_matrix_perm_ds_ds'].shape, (expected_perms, 2, 2))

        self.assertIsNotNone(results['corr_matrix_ds_ref'])
        self.assertEqual(results['corr_matrix_ds_ref'].shape, (2, 1))
        self.assertIsNotNone(results['p_matrix_ds_ref'])
        self.assertEqual(results['p_matrix_ds_ref'].shape, (2, 1))
        self.assertIsNotNone(results['corr_matrix_perm_ds_ref'])
        self.assertEqual(results['corr_matrix_perm_ds_ref'].shape, (expected_perms, 2, 1))


if __name__ == '__main__':
    unittest.main()