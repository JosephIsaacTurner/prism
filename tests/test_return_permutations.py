import unittest
import numpy as np
import os
import shutil
from prism.datasets import Dataset

class TestReturnPermutations(unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.n_features = 5
        np.random.seed(42)
        self.data = np.random.randn(self.n_samples, self.n_features)
        self.design = np.random.randn(self.n_samples, 2)
        self.contrast = np.array([1, 0])

    def test_return_permutations_voxelwise(self):
        # Test without return_permutations (default)
        ds_off = Dataset(
            data=self.data,
            design=self.design,
            contrast=self.contrast,
            n_permutations=5,
            random_state=42,
            return_permutations=False
        )
        results_off = ds_off.permutation_analysis()
        self.assertNotIn('permuted_stats', results_off)

        # Test with return_permutations
        ds_on = Dataset(
            data=self.data,
            design=self.design,
            contrast=self.contrast,
            n_permutations=5,
            random_state=42,
            return_permutations=True
        )
        results_on = ds_on.permutation_analysis()
        self.assertIn('permuted_stats', results_on)
        self.assertEqual(results_on.permuted_stats.shape, (5, self.n_features))
        
        # Verify they are actual statistics (not all zeros)
        self.assertFalse(np.all(results_on.permuted_stats == 0))

    def test_return_permutations_multiple_contrasts(self):
        contrast_2d = np.array([[1, 0], [0, 1]])
        ds = Dataset(
            data=self.data,
            design=self.design,
            contrast=contrast_2d,
            n_permutations=3,
            random_state=42,
            return_permutations=True
        )
        results = ds.permutation_analysis()
        self.assertIn('permuted_stats_c1', results)
        self.assertIn('permuted_stats_c2', results)
        self.assertEqual(results.permuted_stats_c1.shape, (3, self.n_features))
        self.assertEqual(results.permuted_stats_c2.shape, (3, self.n_features))

if __name__ == '__main__':
    unittest.main()
