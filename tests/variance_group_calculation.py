import unittest
import numpy as np
from prism.permutation_logic import get_vg_vector


class TestGetVgVector(unittest.TestCase):

    def test_freely_exchangeable_implicit(self):
        """Test case for implicit free exchangeability (single block)."""
        eb = np.ones(6, dtype=int)
        expected_vg = np.ones(6, dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_freely_exchangeable_flags_neither(self):
        """Test 1D case with within=False, whole=False (should be free exch)."""
        eb = np.array([1, 1, 2, 2, 3, 3])
        expected_vg = np.ones(6, dtype=int)
        result_vg = get_vg_vector(eb, within=False, whole=False)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_freely_exchangeable_flags_simultaneous(self):
        """Test 1D case with within=True, whole=True (simultaneous -> free exch)."""
        eb = np.array([1, 1, 1, 2, 2, 2])
        expected_vg = np.ones(6, dtype=int)
        result_vg = get_vg_vector(eb, within=True, whole=True)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_freely_exchangeable_ex1_structure(self):
        """Test the specific 2D structure from Example 1 (free exch)."""
        eb = np.array([[1, i+1] for i in range(6)])
        expected_vg = np.ones(6, dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)
        
    def test_freely_exchangeable_single_block_multi_col_neg(self):
        """Test 2D structure implying within-block but only one block."""
        eb = np.array([[-1, 5], [-1, 5], [-1, 5]])
        expected_vg = np.ones(3, dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_within_block_1d_default(self):
        """Test 1D within-block using default flags."""
        eb = np.array([10, 10, 10, 10, 20, 20, 20]) # Use non-sequential IDs
        expected_vg = np.array([1, 1, 1, 1, 2, 2, 2])
        result_vg = get_vg_vector(eb) # within=True is default
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_within_block_1d_explicit(self):
        """Test 1D within-block using explicit within=True flag."""
        eb = np.array([3, 1, 3, 1, 3, 1]) # Interleaved blocks
        expected_vg = np.array([2, 1, 2, 1, 2, 1]) # Expected: 1 maps to 1, 3 maps to 2
        result_vg = get_vg_vector(eb, within=True, whole=False)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_within_block_ex2_structure(self):
        """Test the specific 2D structure from Example 2 (within-block)."""
        eb = np.array([[-1, 1], [-1, 1], [-1, 2], [-1, 2], [-1, 3], [-1, 3]])
        expected_vg = np.array([1, 1, 2, 2, 3, 3])
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)
        
    def test_within_block_ex2_structure_non_sequential_ids(self):
        """Test Example 2 structure with non-sequential sub-block IDs."""
        eb = np.array([[-2, 10], [-2, 10], [-2, 5], [-2, 5], [-2, 8], [-2, 8]])
        expected_vg = np.array([3, 3, 1, 1, 2, 2]) # 5->1, 8->2, 10->3
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_whole_block_1d(self):
        """Test 1D whole-block using explicit whole=True flag."""
        eb = np.array([1, 1, 1, 2, 2, 2])
        expected_vg = np.array([1, 2, 3, 1, 2, 3])
        result_vg = get_vg_vector(eb, within=False, whole=True)
        np.testing.assert_array_equal(result_vg, expected_vg)
        
    def test_whole_block_1d_interleaved(self):
        """Test 1D whole-block with interleaved blocks."""
        eb = np.array([1, 2, 1, 2, 1, 2]) # Blocks 1 and 2, size 3 each
        expected_vg = np.array([1, 1, 2, 2, 3, 3]) # Pos 1 (obs 0,1), Pos 2 (obs 2,3), Pos 3 (obs 4,5)
        result_vg = get_vg_vector(eb, within=False, whole=True)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_whole_block_ex3_simple(self):
        """Test the simple 2D structure from Example 3 (whole-block)."""
        eb = np.array([[10, 1], [10, 1], [10, 2], [10, 2], [10, 3], [10, 3]])
        expected_vg = np.array([1, 2, 1, 2, 1, 2])
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_whole_block_ex3_complex(self):
        """Test the complex 2D structure from Example 3 (whole-block)."""
        eb = np.array([[1,-1,1],[1,-1,2],[1,-2,1],[1,-2,2],[1,-3,1],[1,-3,2]])
        expected_vg = np.array([1, 2, 1, 2, 1, 2])
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_empty_input(self):
        """Test with an empty numpy array."""
        eb = np.array([], dtype=int)
        expected_vg = np.array([], dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)
        
    def test_empty_input_2d(self):
        """Test with an empty 2D numpy array."""
        eb = np.empty((0, 2), dtype=int)
        expected_vg = np.array([], dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_single_observation_1d(self):
        """Test with a single observation (1D)."""
        eb = np.array([5])
        expected_vg = np.ones(1, dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_single_observation_2d(self):
        """Test with a single observation (2D)."""
        eb = np.array([[1, 5]])
        expected_vg = np.ones(1, dtype=int)
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    def test_single_block_returns_ones(self):
        """Test that if only one block ID exists, VG is always 1."""
        eb = np.array([2, 2, 2, 2])
        expected_vg = np.ones(4, dtype=int)
        # Test default (within=True), explicit within, explicit whole
        np.testing.assert_array_equal(get_vg_vector(eb.copy()), expected_vg)
        np.testing.assert_array_equal(get_vg_vector(eb.copy(), within=True, whole=False), expected_vg)
        np.testing.assert_array_equal(get_vg_vector(eb.copy(), within=False, whole=True), expected_vg)
        np.testing.assert_array_equal(get_vg_vector(eb.copy(), within=True, whole=True), expected_vg)

    def test_float_input_conversion(self):
        """Test that float inputs representing integers are handled."""
        eb = np.array([1.0, 1.0, 2.0, 2.0])
        expected_vg = np.array([1, 1, 2, 2]) # Default within=True
        result_vg = get_vg_vector(eb)
        np.testing.assert_array_equal(result_vg, expected_vg)

    # --- Error Condition Tests ---

    def test_error_whole_block_non_uniform_1d(self):
        """Test ValueError for 1D whole-block with non-uniform sizes."""
        eb = np.array([1, 1, 2]) # Block sizes 2 and 1
        with self.assertRaisesRegex(ValueError, "requires all blocks to be the same size"):
            get_vg_vector(eb, within=False, whole=True)

    def test_error_whole_block_non_uniform_2d(self):
        """Test ValueError for 2D whole-block implied with non-uniform sub-blocks."""
        # Col 0 positive implies whole-block based on col 1 groups
        # Col 1 groups: 1 (size 2), 2 (size 1) -> non-uniform
        eb = np.array([[1, 1], [1, 1], [1, 2]])
        with self.assertRaisesRegex(ValueError, "requires effective sub-blocks.*to be the same size"):
            get_vg_vector(eb)

    def test_error_invalid_input_type(self):
        """Test TypeError for non-numpy array input."""
        eb = [1, 2, 3]
        with self.assertRaises(TypeError):
            get_vg_vector(eb)

    def test_error_non_integer_input(self):
        """Test ValueError for non-integer float values."""
        eb = np.array([1.0, 1.5, 2.0])
        with self.assertRaisesRegex(ValueError, "Non-integer values found"):
            get_vg_vector(eb)
            
    def test_error_non_integer_like_input(self):
        """Test ValueError for non-numeric values."""
        eb = np.array(['a', 'b', 'a'])
        with self.assertRaisesRegex(ValueError, "integer-like indices"):
             get_vg_vector(eb)

    def test_error_mixed_sign_2d_simple(self):
        """Test ValueError for ambiguous 2D structure (mixed signs in col 0)."""
        eb = np.array([[1, 1], [-1, 2], [1, 3]])
        # Update the regex to match the actual error message partially
        with self.assertRaisesRegex(ValueError, "mixed positive/negative indices in the first column across"):
            get_vg_vector(eb)
            
    def test_error_mixed_sign_2d_multi_block(self):
        """Test ValueError for ambiguous 2D structure (mixed signs across l0 blocks)."""
        eb = np.array([[1, 1], [1, 2], [-2, 3], [-2, 4]])
        with self.assertRaisesRegex(ValueError, "mixed positive/negative indices in the first column across"):
            get_vg_vector(eb)
            
    def test_error_0d_input(self):
        """Test ValueError for 0-dimensional input."""
        eb = np.array(5)
        with self.assertRaisesRegex(ValueError, "cannot be 0-dimensional"):
            get_vg_vector(eb)


if __name__ == '__main__':
    unittest.main()