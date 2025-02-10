import unittest
import numpy as np
from pstn.inference import permutation_analysis, yield_permuted_design
from pstn.stats import welchs_t_glm

class TestPermutationAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.n_samples = 36
        self.n_elements_per_sample = 2000
        self.n_features = 3
        self.n_permutations = 250
        self.n_exchangeability_groups = 2
        self.random_seed = 42
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Generate test data
        self.simulated_data = np.random.randn(self.n_samples, self.n_elements_per_sample)
        self.simulated_design = np.random.randn(self.n_samples, self.n_features)
        
        # Set up contrast vector
        self.simulated_contrast = np.zeros(self.n_features)
        self.simulated_contrast[0] = 1
        
        # Create exchangeability matrix
        base_groups = np.ones(np.floor(self.n_samples / self.n_exchangeability_groups).astype(int))
        self.simulated_exchangeability_matrix = np.hstack([
            base_groups * (i + 1) for i in range(self.n_exchangeability_groups)
        ])
        
        # Handle remainder if n_samples isn't evenly divisible
        remainder = self.n_samples % self.n_exchangeability_groups
        if remainder != 0:
            remainder_group = np.ones(remainder) * self.n_exchangeability_groups
            self.simulated_exchangeability_matrix = np.hstack([
                self.simulated_exchangeability_matrix, 
                remainder_group
            ])

    def test_permutation_analysis_significance_levels(self):
        """Test that the proportion of significant results is within expected bounds."""
        # Run permutation analysis
        unc_p, fdr_p, fwe_p = permutation_analysis(
            data=self.simulated_data,
            design=self.simulated_design,
            contrast=self.simulated_contrast,
            stat_function=welchs_t_glm,
            n_permutations=self.n_permutations,
            random_seed=self.random_seed,
            two_tailed=True,
            exchangeability_matrix=self.simulated_exchangeability_matrix,
            within=True,
            whole=False,
            accel_tail=True
        )
        
        # Test that the proportion of significant results is less than 6%
        # for all three p-value types
        significance_threshold = 0.05
        max_proportion = 0.06
        max_significant = int(max_proportion * self.n_elements_per_sample)
        
        for p_values, p_type in [(unc_p, "uncorrected"), 
                                (fdr_p, "FDR-corrected"), 
                                (fwe_p, "FWE-corrected")]:
            n_significant = np.sum(p_values < significance_threshold)
            self.assertLess(
                n_significant, 
                max_significant,
                f"Too many significant results for {p_type} p-values: "
                f"{n_significant} > {max_significant}"
            )


class TestYieldPermutedDesign(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # A simple design matrix with 6 rows and 2 features.
        # We make rows easily distinguishable.
        self.design = np.array([
            [ 0, 10],
            [ 1, 11],
            [ 2, 20],
            [ 3, 21],
            [ 4, 30],
            [ 5, 31],
        ])

    def test_no_exchangeability(self):
        """Case 1: No exchangeability blocks -> full permutation."""
        perms = list(yield_permuted_design(self.design, n_permutations=5, random_seed=42))
        self.assertEqual(len(perms), 5)
        for p in perms:
            self.assertEqual(p.shape, self.design.shape)
            # Each permutation should contain exactly the same rows (as tuples).
            self.assertCountEqual([tuple(row) for row in p],
                                [tuple(row) for row in self.design])

    def test_single_vector_within(self):
        """
        Case 2: Single vector, within=True.
        Blocks: [1,1,2,2,3,3]
        Permutation is done within each block so that the blocks appear in original order.
        """
        ex_vec = np.array([1, 1, 2, 2, 3, 3])
        perms = list(yield_permuted_design(self.design, n_permutations=5, random_seed=42,
                                         exchangeability_matrix=ex_vec, within=True))
        # The original block order is preserved: first two rows, then next two, etc.
        orig_blocks = {
            1: self.design[0:2],
            2: self.design[2:4],
            3: self.design[4:6]
        }
        for p in perms:
            # Split the permutation into 3 blocks of 2 rows.
            block1, block2, block3 = p[0:2], p[2:4], p[4:6]
            # Within each block, the rows should be a permutation of the original block.
            self.assertCountEqual([tuple(r) for r in block1],
                                [tuple(r) for r in orig_blocks[1]])
            self.assertCountEqual([tuple(r) for r in block2],
                                [tuple(r) for r in orig_blocks[2]])
            self.assertCountEqual([tuple(r) for r in block3],
                                [tuple(r) for r in orig_blocks[3]])

    def test_single_vector_whole(self):
        """
        Case 3: Single vector, whole=True.
        Blocks: [1,1,2,2,3,3]
        Permutation is at the block level, so within each block the order is preserved.
        """
        ex_vec = np.array([1, 1, 2, 2, 3, 3])
        perms = list(yield_permuted_design(self.design, n_permutations=5, random_seed=42,
                                         exchangeability_matrix=ex_vec, whole=True))
        # Original blocks (order preserved within block)
        orig_blocks = {
            1: self.design[0:2],
            2: self.design[2:4],
            3: self.design[4:6]
        }
        for p in perms:
            # Because blocks are permuted as a whole, the 6 rows should be
            # arranged as a concatenation of the 3 original blocks in some order.
            self.assertEqual(p.shape, self.design.shape)
            # Get the three contiguous blocks of length 2 from the permuted design.
            perm_blocks = [p[i:i+2] for i in range(0, 6, 2)]
            # Each block must match one of the original blocks exactly (order preserved).
            matched = []
            for pb in perm_blocks:
                found = False
                for key, ob in orig_blocks.items():
                    if np.array_equal(pb, ob) and key not in matched:
                        found = True
                        matched.append(key)
                        break
                self.assertTrue(found, "Block {} does not match any original block.".format(pb))
            self.assertCountEqual(matched, [1, 2, 3])

    def test_2d_simple(self):
        """
        Case 4: 2D exchangeability matrix, simple nested structure.
        Let the exchangeability matrix be 2D with 2 columns:
          - First column: block labels (e.g., [1,1,2,2,3,3])
          - Second column: all -1 (implying within-block shuffling at the deepest level).
        In this case, flags are ignored and the recursive procedure is used.
        """
        ex_mat = np.column_stack((np.array([1, 1, 2, 2, 3, 3]), -np.ones(6)))
        perms = list(yield_permuted_design(self.design, n_permutations=5, random_seed=42,
                                         exchangeability_matrix=ex_mat))
        for p in perms:
            self.assertEqual(p.shape, self.design.shape)
            # Overall, p must be a permutation of the design.
            self.assertCountEqual([tuple(row) for row in p],
                                [tuple(row) for row in self.design])
            # For each block defined by the first column, the order at the deepest level is permuted,
            # meaning that within each block the order might change.
            # We check that each block's rows (as a set) equal the corresponding original rows.
            orig_blocks = {
                1: self.design[0:2],
                2: self.design[2:4],
                3: self.design[4:6]
            }
            # Extract indices by matching rows.
            # (Since our rows are distinct, this is reliable.)
            for block_label, orig_block in orig_blocks.items():
                perm_block = [row for row in p if row[0] in orig_block[:,0]]
                self.assertCountEqual([tuple(r) for r in perm_block],
                                    [tuple(r) for r in orig_block])

    def test_2d_complex(self):
        """
        Case 5: Complex 2D exchangeability matrix with three columns.
        We construct a design matrix with 8 samples and an exchangeability matrix as follows:
        - Level 1: all ones (so all samples are in one super-block)
        - Level 2: Two blocks: first 4 samples get -1, next 4 get 1 
            (so that the two blocks are distinct and within-block permutation is applied)
        - Level 3: Within each level2 block, labels 1,2,3,4 (which in base case are used to shuffle if negative)
        Note: This test checks overall invariants.
        """
        n_samples = 8
        design = np.arange(n_samples).reshape(n_samples, 1)
        level1 = np.ones(n_samples)                    # All in one super-block.
        level2 = np.array([-1] * 4 + [1] * 4)         # Two distinct blocks.
        level3 = np.tile(np.arange(1, 5), 2)          # [1,2,3,4,1,2,3,4]
        ex_mat = np.column_stack((level1, level2, level3))
        perms = list(yield_permuted_design(design, n_permutations=5, random_seed=42,
                                         exchangeability_matrix=ex_mat))
        for p in perms:
            self.assertEqual(p.shape, design.shape)
            # Check that p is a permutation of design.
            self.assertCountEqual([tuple(row) for row in p],
                                [tuple(row) for row in design])
            # Because level2 now distinguishes two blocks, we expect that the first 4 rows of p
            # are a permutation of the original first block (design[0:4]), and the last 4 rows a permutation
            # of the second block (design[4:8]).
            group1 = p[0:4]
            group2 = p[4:8]
            self.assertCountEqual([tuple(r) for r in group1],
                                [tuple(r) for r in design[0:4]])
            self.assertCountEqual([tuple(r) for r in group2],
                                [tuple(r) for r in design[4:8]])


if __name__ == '__main__':
    unittest.main()