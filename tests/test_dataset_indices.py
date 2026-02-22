import unittest
import numpy as np
import os
import shutil
from prism.datasets.dataset import Dataset

class TestDatasetIndices(unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.n_features = 3
        # Ensure repeatable results for the test itself
        np.random.seed(42)
        self.data = np.random.randn(self.n_samples, 5)
        self.design = np.random.randn(self.n_samples, self.n_features)
        self.contrast = np.array([1, 0, 0])
        self.output_prefix = "test_output/indices_test"
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
        os.makedirs("test_output", exist_ok=True)

    def test_generate_indices_single_contrast(self):
        ds = Dataset(
            data=self.data,
            design=self.design,
            contrast=self.contrast,
            n_permutations=5,
            random_state=42
        )
        indices = ds.generate_permutation_indices()
        
        self.assertIn('c1', indices)
        self.assertEqual(indices['c1'].shape, (5, self.n_samples))
        
        # Verify it matches permutation_analysis results
        results = ds.permutation_analysis()
        # For n_t_contrasts == 1, 'c1' is dropped in results
        np.testing.assert_array_equal(indices['c1'], results.permuted_indices)

    def test_generate_indices_multiple_contrasts(self):
        contrast_2d = np.array([[1, 0, 0], [0, 1, 0]])
        ds = Dataset(
            data=self.data,
            design=self.design,
            contrast=contrast_2d,
            n_permutations=5,
            random_state=42,
            f_contrast_indices=[1, 1]
        )
        indices = ds.generate_permutation_indices()
        
        self.assertIn('c1', indices)
        self.assertIn('c2', indices)
        self.assertIn('f', indices)
        
        results = ds.permutation_analysis()
        # For n_t_contrasts > 1, 'c1' and 'c2' suffixes are kept
        np.testing.assert_array_equal(indices['c1'], results.permuted_indices_c1)
        np.testing.assert_array_equal(indices['c2'], results.permuted_indices_c2)
        np.testing.assert_array_equal(indices['f'], results.permuted_indices_f)
        
        # Check that they are different (due to seed shifting)
        self.assertFalse(np.array_equal(indices['c1'], indices['c2']))
        self.assertFalse(np.array_equal(indices['c1'], indices['f']))

    def test_save_indices(self):
        ds = Dataset(
            data=self.data,
            design=self.design,
            contrast=self.contrast,
            n_permutations=5,
            random_state=42,
            output_prefix=self.output_prefix
        )
        ds.generate_permutation_indices()
        
        expected_file = f"{self.output_prefix}_permuted_indices_c1.csv"
        self.assertTrue(os.path.exists(expected_file), f"File {expected_file} not found")
        saved_indices = np.loadtxt(expected_file, delimiter=",", dtype=int)
        
        # Re-generate to compare
        indices = ds.generate_permutation_indices()
        np.testing.assert_array_equal(indices['c1'], saved_indices)

    def tearDown(self):
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")

if __name__ == '__main__':
    unittest.main()
