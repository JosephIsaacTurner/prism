import unittest
import numpy as np
from prism.inference import yield_permuted_indices


class TestYieldPermutedDesignRevised(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for the class."""
        # Use a fixed seed for reproducibility during test adjustment
        # Can be changed back to default_rng(None) or similar later
        cls.base_seed = 12345
        cls.rng = np.random.default_rng(cls.base_seed)

        # --- Design 6 (3 blocks, size 2) ---
        cls.design_6 = np.array([
            [ 0, 10], [ 1, 11], # Block 1 (idx 0, 1)
            [ 2, 20], [ 3, 21], # Block 2 (idx 2, 3)
            [ 4, 30], [ 5, 31], # Block 3 (idx 4, 5)
        ])
        cls.n_samples_6 = cls.design_6.shape[0]
        cls.eb_6_within = np.array([1, 1, 2, 2, 3, 3]) # Single level within
        cls.eb_6_multi_within = np.array([[-1, 1], [-1, 1], [-1, 2], [-1, 2], [-1, 3], [-1, 3]]) # PALM Ex2: Multi-level Within
        cls.eb_6_multi_whole = np.array([[1, 1], [1, 1], [1, 2], [1, 2], [1, 3], [1, 3]]) # PALM Ex3: Multi-level Whole (+Internal)
        cls.eb_6_multi_free = np.column_stack((np.ones(cls.n_samples_6), np.arange(1, cls.n_samples_6 + 1))) # PALM Ex1: Free

        # New EB Matrix for combined test (based on PALM Ex3 description)
        # L0=1 (shuffle L1 blocks), L1=-ve (shuffle within L1 blocks)
        cls.eb_6_true_combined = np.array([
            [1, -1, 1], [1, -1, 2], # L1 block -1 (orig block 1)
            [1, -2, 1], [1, -2, 2], # L1 block -2 (orig block 2)
            [1, -3, 1], [1, -3, 2], # L1 block -3 (orig block 3)
        ])


        cls.blocks_6_map = { # Map block ID to original indices
             1: np.array([0, 1]),
             2: np.array([2, 3]),
             3: np.array([4, 5]),
        }
        cls.block_data_6 = { # Map block ID to original data
            bid: cls.design_6[indices] for bid, indices in cls.blocks_6_map.items()
        }

        # --- Design 5 (2 unequal blocks) ---
        cls.design_5_unequal = np.array([
            [100, 0], [101, 1], [102, 2], # Block 1 (idx 0, 1, 2)
            [200, 3], [201, 4],          # Block 2 (idx 3, 4)
        ])
        cls.n_samples_5 = cls.design_5_unequal.shape[0]
        cls.eb_5_unequal = np.array([1, 1, 1, 2, 2])
        cls.blocks_5_map = {
             1: np.array([0, 1, 2]),
             2: np.array([3, 4]),
        }
        cls.block_data_5 = {
            bid: cls.design_5_unequal[indices] for bid, indices in cls.blocks_5_map.items()
        }

        cls.n_samples_16 = 16
        # Create pairs of values: [[100, 101], [110, 111], ..., [250, 251]]
        base_vals = np.arange(cls.n_samples_16 // 2) * 10 + 100 # [100, 110, ..., 170]
        cls.design_16 = np.column_stack((base_vals, base_vals + 1)).reshape(cls.n_samples_16, 1)
        # Example: design_16 = [[100],[101],[110],[111],...,[170],[171],[200],[201],...,[270],[271]]

        # L0=1 -> shuffle L1 groups {A: 0-7, B: 8-15}
        # L1: A=-1 (fix L2 sub-blocks within A), B=1 (shuffle L2 sub-blocks within B)
        # L2: Negative indices -> shuffle within L2 sub-blocks (pairs defined by L3)
        # L3: Positive indices -> allows base permutation of pairs
        cls.eb_16_complex = np.array([
            # Group A (Indices 0-7), L1 = -1 (Fixed L2 Order)
            [1, -1, -1, 1], [1, -1, -1, 2], # L2 block -1
            [1, -1, -2, 1], [1, -1, -2, 2], # L2 block -2
            [1, -1, -3, 1], [1, -1, -3, 2], # L2 block -3
            [1, -1, -4, 1], [1, -1, -4, 2], # L2 block -4
            # Group B (Indices 8-15), L1 = 1 (Shuffled L2 Order)
            [1,  1, -5, 1], [1,  1, -5, 2], # L2 block -5
            [1,  1, -6, 1], [1,  1, -6, 2], # L2 block -6
            [1,  1, -7, 1], [1,  1, -7, 2], # L2 block -7
            [1,  1, -8, 1], [1,  1, -8, 2], # L2 block -8
        ])
        cls.group_A_indices_16 = np.arange(0, 8)
        cls.group_B_indices_16 = np.arange(8, 16)
        cls.group_A_data_16 = cls.design_16[cls.group_A_indices_16]
        cls.group_B_data_16 = cls.design_16[cls.group_B_indices_16]
        cls.group_A_content_16 = set(tuple(r) for r in cls.group_A_data_16)
        cls.group_B_content_16 = set(tuple(r) for r in cls.group_B_data_16)

        cls.n_perms_test = 5 # Default number of permutations for most tests
        # Increased number for potentially flaky tests
        cls.n_perms_robust = 50

    # Reset RNG before each test to ensure isolation
    def setUp(self):
        self.rng = np.random.default_rng(self.base_seed)

    def _run_and_collect(self, design, n_perms, random_state, exchangeability_matrix=None, within=None, whole=None):
            """Helper to run the generator and return a list."""
            current_rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
            perm_idx_generator = yield_permuted_indices(
                design=design,
                n_permutations=n_perms,
                random_state=current_rng, # Use provided or fresh RNG
                exchangeability_matrix=exchangeability_matrix,
                within=within,
                whole=whole
            )
            perm_indices = list(perm_idx_generator)
            designs = [design[indices] for indices in perm_indices]
            return designs

    # --- Basic Cases (Free Exchange) ---
    def test_free_exchange_no_eb(self):
        """Free Exchange: No exchangeability matrix provided."""
        perms = self._run_and_collect(self.design_6, self.n_perms_test, self.rng)
        self.assertEqual(len(perms), self.n_perms_test)
        orig_content = set(tuple(r) for r in self.design_6)
        diff_from_orig_seen = False
        unique_perms = set()
        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape)
            p_content = set(tuple(r) for r in p)
            self.assertSetEqual(p_content, orig_content, "Content mismatch")
            unique_perms.add(p.tobytes()) # Store unique permutations seen
            if not np.array_equal(p, self.design_6): diff_from_orig_seen = True
        self.assertTrue(diff_from_orig_seen, "Free exchange should produce permutations different from original (n>1).")
        self.assertTrue(len(unique_perms) > 1 or self.n_perms_test <= 1, "Free exchange should produce multiple unique permutations (n>1).")

    def test_free_exchange_single_level_both_flags(self):
        """Free Exchange: Single level EB with within=True, whole=True."""
        perms = self._run_and_collect(self.design_6, self.n_perms_test, self.rng, exchangeability_matrix=self.eb_6_within, within=True, whole=True)
        self.assertEqual(len(perms), self.n_perms_test)
        orig_content = set(tuple(r) for r in self.design_6)
        diff_from_orig_seen = False; unique_perms = set()
        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape); self.assertSetEqual(set(tuple(r) for r in p), orig_content)
            unique_perms.add(p.tobytes());
            if not np.array_equal(p, self.design_6): diff_from_orig_seen = True
        self.assertTrue(diff_from_orig_seen or self.n_perms_test <=1)
        self.assertTrue(len(unique_perms) > 1 or self.n_perms_test <= 1)

    def test_free_exchange_single_level_no_flags_false(self):
        """Free Exchange: Single level EB with within=False, whole=False (should default to free)."""
        perms = self._run_and_collect(self.design_6, self.n_perms_test, self.rng, exchangeability_matrix=self.eb_6_within, within=False, whole=False)
        self.assertEqual(len(perms), self.n_perms_test)
        orig_content = set(tuple(r) for r in self.design_6)
        diff_from_orig_seen = False; unique_perms = set()
        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape); self.assertSetEqual(set(tuple(r) for r in p), orig_content)
            unique_perms.add(p.tobytes());
            if not np.array_equal(p, self.design_6): diff_from_orig_seen = True
        self.assertTrue(diff_from_orig_seen or self.n_perms_test <=1)
        self.assertTrue(len(unique_perms) > 1 or self.n_perms_test <= 1)

    def test_free_exchange_multi_level_example1(self):
        """Free Exchange: Multi-level EB from PALM Example 1."""
        perms = self._run_and_collect(self.design_6, self.n_perms_test, self.rng, exchangeability_matrix=self.eb_6_multi_free)
        self.assertEqual(len(perms), self.n_perms_test)
        orig_content = set(tuple(r) for r in self.design_6)
        diff_from_orig_seen = False; unique_perms = set()
        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape); self.assertSetEqual(set(tuple(r) for r in p), orig_content)
            unique_perms.add(p.tobytes());
            if not np.array_equal(p, self.design_6): diff_from_orig_seen = True
        self.assertTrue(diff_from_orig_seen or self.n_perms_test <=1)
        self.assertTrue(len(unique_perms) > 1 or self.n_perms_test <= 1)

    # --- Within-Block Shuffling ---

    def test_within_block_single_level_default(self):
        """Within-Block: Single level EB, default flags (within=None, whole=None)."""
        # Use more permutations to avoid stochastic failure
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_6, n_perms, self.rng,
                                    exchangeability_matrix=self.eb_6_within)
        self.assertEqual(len(perms), n_perms)
        internal_shuffle_seen = {bid: False for bid in self.blocks_6_map}

        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape)
            # Check block positions are fixed AND content is preserved
            for block_id, orig_indices in self.blocks_6_map.items():
                p_block_data = p[orig_indices]
                orig_block_data = self.block_data_6[block_id]
                self.assertCountEqual([tuple(r) for r in p_block_data],
                                      [tuple(r) for r in orig_block_data],
                                      f"Block {block_id} content mismatch.")
                # Check if internal order changed for this block in this permutation
                if not np.array_equal(p_block_data, orig_block_data):
                    internal_shuffle_seen[block_id] = True

        # Check that internal shuffling occurred for each block across the permutations
        for block_id, shuffled in internal_shuffle_seen.items():
            self.assertTrue(shuffled, f"Within-block shuffle did not occur for block {block_id} over {n_perms} permutations.")

    def test_within_block_single_level_explicit(self):
        """Within-Block: Single level EB, explicit within=True."""
        # Use more permutations to avoid stochastic failure
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_6, n_perms, self.rng,
                                    exchangeability_matrix=self.eb_6_within, within=True) # whole defaults to False
        self.assertEqual(len(perms), n_perms)
        internal_shuffle_seen = {bid: False for bid in self.blocks_6_map}
        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape)
            for block_id, orig_indices in self.blocks_6_map.items():
                p_block_data = p[orig_indices]
                orig_block_data = self.block_data_6[block_id]
                self.assertCountEqual([tuple(r) for r in p_block_data], [tuple(r) for r in orig_block_data])
                if not np.array_equal(p_block_data, orig_block_data):
                    internal_shuffle_seen[block_id] = True
        for block_id, shuffled in internal_shuffle_seen.items():
             self.assertTrue(shuffled, f"Within-block shuffle did not occur for block {block_id} over {n_perms} permutations.")

    def test_within_block_multi_level_example2(self):
        """Within-Block: Multi-level EB from PALM Example 2."""
        # Use more permutations to avoid stochastic failure
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_6, n_perms, self.rng,
                                    exchangeability_matrix=self.eb_6_multi_within)
        self.assertEqual(len(perms), n_perms)
        internal_shuffle_seen = {bid: False for bid in self.blocks_6_map}
        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape)
            for block_id, orig_indices in self.blocks_6_map.items():
                p_block_data = p[orig_indices] # Check data at original block positions
                orig_block_data = self.block_data_6[block_id]
                # Check content is preserved within original block indices
                self.assertCountEqual([tuple(r) for r in p_block_data], [tuple(r) for r in orig_block_data],
                                       f"Block {block_id} content mismatch at original indices.")
                if not np.array_equal(p_block_data, orig_block_data):
                    internal_shuffle_seen[block_id] = True
        for block_id, shuffled in internal_shuffle_seen.items():
            self.assertTrue(shuffled, f"Within-block shuffle (multi-level Ex2) did not occur for block {block_id} over {n_perms} permutations.")

    def test_within_block_unequal_sizes(self):
        """Within-Block: Single level EB with unequal block sizes."""
        # Use more permutations to avoid stochastic failure
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_5_unequal, n_perms, self.rng,
                                    exchangeability_matrix=self.eb_5_unequal, within=True)
        self.assertEqual(len(perms), n_perms)
        internal_shuffle_seen = {bid: False for bid in self.blocks_5_map}
        for p in perms:
            self.assertEqual(p.shape, self.design_5_unequal.shape)
            for block_id, orig_indices in self.blocks_5_map.items():
                p_block_data = p[orig_indices]
                orig_block_data = self.block_data_5[block_id]
                self.assertCountEqual([tuple(r) for r in p_block_data], [tuple(r) for r in orig_block_data])
                if not np.array_equal(p_block_data, orig_block_data):
                    internal_shuffle_seen[block_id] = True
        for block_id, shuffled in internal_shuffle_seen.items():
             self.assertTrue(shuffled, f"Within-block shuffle did not occur for unequal block {block_id} over {n_perms} permutations.")

    # --- Whole-Block Shuffling ---

    def test_whole_block_single_level_explicit(self):
        """Whole-Block: Single level EB, explicit whole=True."""
        perms = self._run_and_collect(self.design_6, self.n_perms_test, self.rng,
                                    exchangeability_matrix=self.eb_6_within, whole=True) # within defaults to False
        self.assertEqual(len(perms), self.n_perms_test)
        block_orders_seen = set()
        internal_order_preserved = True

        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape)
            current_block_order = []
            # Identify blocks by checking content and order preservation
            current_pos = 0
            matched_block_ids_this_perm = set()
            while current_pos < p.shape[0]:
                found_match = False
                for block_id, orig_data in self.block_data_6.items():
                    block_size = orig_data.shape[0]
                    # Check if block fits and hasn't been matched yet
                    if current_pos + block_size <= p.shape[0] and block_id not in matched_block_ids_this_perm:
                        p_slice = p[current_pos : current_pos + block_size]
                        # Whole block means internal order IS preserved
                        if np.array_equal(p_slice, orig_data):
                            current_block_order.append(block_id)
                            matched_block_ids_this_perm.add(block_id) # Mark as used
                            current_pos += block_size
                            found_match = True
                            break # Found match for this position
                if not found_match:
                    # Added more informative fail message
                    self.fail(f"Could not match original block data (exact match) starting at index {current_pos} in permutation {p.tolist()}. Matched so far: {current_block_order}")

            self.assertCountEqual(current_block_order, list(self.blocks_6_map.keys()), "Not all blocks found in permutation")
            block_orders_seen.add(tuple(current_block_order))

        # Use more robust check for >1 order, accounting for n_perms=1
        self.assertTrue(len(block_orders_seen) > 1 or self.n_perms_test <= 1,
                        "Whole-block shuffling did not change block order across permutations (unless n_perms=1).")


    def test_whole_block_multi_level_example3(self):
        """Whole-Block: Multi-level EB [[1,1],[1,2],[1,3]] -> shuffle block order & internal."""
        # Use more permutations to avoid stochastic failure for internal check
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_6, n_perms, self.rng,
                                      exchangeability_matrix=self.eb_6_multi_whole)
        self.assertEqual(len(perms), n_perms, "Generator did not yield the expected number of permutations.")

        block_orders_seen = set()
        internal_shuffle_seen_overall = False # Flag to check if internal shuffle ever happened

        for i, p in enumerate(perms):
            self.assertEqual(p.shape, self.design_6.shape, f"Perm {i}: Shape mismatch")

            current_block_order = []
            current_pos = 0
            matched_block_ids_this_perm = set()

            while current_pos < p.shape[0]:
                found_match = False
                slice_start = current_pos
                block_size = 2 # Hardcoded based on design_6

                if slice_start + block_size > p.shape[0]:
                     self.fail(f"Perm {i}: Ran out of data at position {slice_start} before matching all blocks.")

                p_slice = p[slice_start : slice_start + block_size]
                p_slice_content = set(tuple(row) for row in p_slice)

                # Find which original block matches the content of this slice
                for block_id, orig_data in self.block_data_6.items():
                    if block_id not in matched_block_ids_this_perm:
                        orig_content = set(tuple(row) for row in orig_data)
                        if p_slice_content == orig_content:
                            current_block_order.append(block_id)
                            matched_block_ids_this_perm.add(block_id)
                            # Check if internal order differs from original for this specific match
                            if not np.array_equal(p_slice, orig_data):
                                internal_shuffle_seen_overall = True
                            current_pos += block_size
                            found_match = True
                            break

                if not found_match:
                    self.fail(f"Perm {i}: Could not find content match for slice {p_slice.tolist()} starting at position {slice_start}.")

            self.assertCountEqual(current_block_order, list(self.blocks_6_map.keys()),
                                  f"Perm {i}: Did not find all expected blocks. Found: {current_block_order}")
            block_orders_seen.add(tuple(current_block_order))

        self.assertTrue(len(block_orders_seen) > 1 or n_perms <= 1,
                        "Multi-level whole-block shuffling (Ex3) should change block order across permutations (unless n_perms=1).")
        self.assertTrue(internal_shuffle_seen_overall or n_perms <= 1,
                        "Multi-level whole-block shuffling (Ex3) should cause internal shuffling for at least one block in at least one permutation (unless n_perms=1).")

    # --- Combined/Complex Shuffling ---

    def test_combined_multi_level(self):
        """Combined: Multi-level (PALM Ex3 Style [[1,-1,1],...]) -> shuffle block order ONLY."""
        # Use the new eb_6_true_combined matrix
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_6, n_perms, self.rng,
                                    exchangeability_matrix=self.eb_6_true_combined)
        self.assertEqual(len(perms), n_perms)
        block_orders_seen = set()
        internal_shuffle_seen = {bid: False for bid in self.blocks_6_map} # Still track if shuffle happens unexpectedly

        for p in perms:
            self.assertEqual(p.shape, self.design_6.shape)
            current_block_order = []
            matched_keys_this_perm = set()
            current_pos = 0
            while current_pos < p.shape[0]:
                 # ... (block identification logic remains the same) ...
                 found_match = False
                 block_size = 2
                 if current_pos + block_size > p.shape[0]: self.fail(f"Ran out of data matching blocks at pos {current_pos}")
                 p_slice_data = p[current_pos : current_pos + block_size]
                 p_slice_content = frozenset(tuple(r) for r in p_slice_data)
                 found_key = None
                 for key, orig_data in self.block_data_6.items():
                     orig_content = frozenset(tuple(r) for r in orig_data)
                     if p_slice_content == orig_content and key not in matched_keys_this_perm:
                         found_key = key
                         matched_keys_this_perm.add(key)
                         break
                 if found_key is None: self.fail(f"Could not match permuted slice content {p_slice_content} starting at pos {current_pos}")
                 current_block_order.append(found_key)

                 # Check if internal order IS THE SAME as original
                 if not np.array_equal(p_slice_data, self.block_data_6[found_key]):
                      internal_shuffle_seen[found_key] = True # Track unexpected shuffles

                 current_pos += block_size

            self.assertCountEqual(current_block_order, list(self.blocks_6_map.keys()),
                                  "Permutation did not contain exactly one instance of each original block's content.")
            block_orders_seen.add(tuple(current_block_order))

        # Check if block order actually changed across permutations
        self.assertTrue(len(block_orders_seen) > 1 or n_perms <= 1,
                        "Combined shuffling (Ex3 Style) should change block order (unless n_perms=1).")

        # Check that internal shuffling DID NOT happen for any block
        for block_id, shuffled in internal_shuffle_seen.items():
             self.assertFalse(shuffled, f"Combined shuffling (Ex3 Style [[1,-1,1],...]) unexpectedly changed internal order for block {block_id}.")

    def test_complex_multi_level_logic(self):
        """Complex: Multi-level check group swap and internal shuffle based on signs (n=16)."""
        # Using n=16 design and EB matrix
        # Use more permutations to avoid stochastic failure
        n_perms = self.n_perms_robust
        perms = self._run_and_collect(self.design_16, n_perms, self.rng,
                                    exchangeability_matrix=self.eb_16_complex)
        self.assertEqual(len(perms), n_perms)

        group_swap_seen = False
        group_A_internal_shuffle_seen = False
        group_B_internal_shuffle_seen = False

        for p in perms:
            self.assertEqual(p.shape, self.design_16.shape)
            p_first_half_data = p[0:8]   # Indices 0-7
            p_second_half_data = p[8:16] # Indices 8-15
            p_first_half_content = set(tuple(r) for r in p_first_half_data)
            p_second_half_content = set(tuple(r) for r in p_second_half_data)

            # Check if the content matches Group A and Group B in either order
            is_order_AB = (p_first_half_content == self.group_A_content_16 and
                           p_second_half_content == self.group_B_content_16)
            is_order_BA = (p_first_half_content == self.group_B_content_16 and
                           p_second_half_content == self.group_A_content_16)

            self.assertTrue(is_order_AB or is_order_BA,
                            "Permutation doesn't contain original group A and group B content correctly partitioned.")

            if is_order_AB:
                # Order A, B
                if not np.array_equal(p_first_half_data, self.group_A_data_16):
                    group_A_internal_shuffle_seen = True
                if not np.array_equal(p_second_half_data, self.group_B_data_16):
                    group_B_internal_shuffle_seen = True
            elif is_order_BA:
                # Order B, A
                group_swap_seen = True
                # Check internal shuffling against the correct original data based on content
                if not np.array_equal(p_first_half_data, self.group_B_data_16):
                     group_B_internal_shuffle_seen = True
                if not np.array_equal(p_second_half_data, self.group_A_data_16):
                     group_A_internal_shuffle_seen = True

        # Assertions after checking all permutations
        self.assertTrue(group_swap_seen or n_perms <= 1,
                        "Complex: Group order (A, B) vs (B, A) should change across permutations (L0=1) (unless n_perms=1).")
        # Now that blocks are size 2, internal shuffling should be detectable
        self.assertTrue(group_A_internal_shuffle_seen or n_perms <= 1,
                        "Complex: Group A internal order should shuffle (L1=-1 -> L2=-ve -> L3=+ve) (unless n_perms=1).")
        self.assertTrue(group_B_internal_shuffle_seen or n_perms <= 1,
                        "Complex: Group B internal order should shuffle (L1=1 -> shuffle L2 -> L2=-ve -> L3=+ve) (unless n_perms=1).")


    # --- Error and Edge Cases ---
    def test_error_whole_block_unequal(self):
        """Error: ValueError for whole-block shuffling with unequal block sizes."""
        with self.assertRaisesRegex(ValueError, "requires all blocks to be the same size"):
            list(self._run_and_collect(self.design_5_unequal, 1, self.rng, self.eb_5_unequal, whole=True)) # Need n=1 to trigger

    def test_error_multi_level_whole_unequal(self):
        """Error: ValueError for multi-level positive index requiring uniform sub-blocks."""
        eb_mat_bad = np.column_stack((np.ones(self.n_samples_5, dtype=int), self.eb_5_unequal)) # [[1,1],[1,1],[1,1], [1,2],[1,2]]
        with self.assertRaisesRegex(ValueError, "requires sub-blocks .* to be uniform size"):
             list(self._run_and_collect(self.design_5_unequal, 1, self.rng, exchangeability_matrix=eb_mat_bad))

    def test_error_zero_index(self):
        """Error: ValueError if exchangeability matrix contains zero."""
        eb_zero = np.array([1, 1, 0, 0, 2, 2])
        with self.assertRaisesRegex(ValueError, "contains index 0"):
             list(self._run_and_collect(self.design_6, 1, self.rng, exchangeability_matrix=eb_zero))
        eb_zero_multi = np.array([[1, 0], [1, 1], [-1, 2], [-1, 2], [3, 1], [3, 1]])
        with self.assertRaisesRegex(ValueError, "contains index 0"):
             list(self._run_and_collect(self.design_6, 1, self.rng, exchangeability_matrix=eb_zero_multi))

    def test_error_mixed_signs(self):
        """Error: ValueError for multi-level mixed positive/negative signs at same level."""
        eb_mixed = np.array([[1, 1], [1, 1], [-2, 2], [-2, 2], [3, 3], [3, 3]]) # L0 has 1, -2, 3
        with self.assertRaisesRegex(ValueError, "Mixed positive/negative block indices found"):
             list(self._run_and_collect(self.design_6, 1, self.rng, exchangeability_matrix=eb_mixed))

    def test_error_wrong_eb_rows(self):
        """Error: ValueError if eb_matrix rows don't match design rows."""
        eb_wrong_rows = np.array([1, 1, 2, 2]) # 4 rows for design_6
        with self.assertRaisesRegex(ValueError, "must match design matrix rows"):
            list(self._run_and_collect(self.design_6, 1, self.rng, exchangeability_matrix=eb_wrong_rows))

    def test_error_eb_type(self):
        """Error: TypeError for non-numeric eb_matrix."""
        eb_str = np.array(['1', '1', '2', '2', '3', '3'])
        with self.assertRaisesRegex(TypeError, "numeric indices"):
             list(self._run_and_collect(self.design_6, 1, self.rng, exchangeability_matrix=eb_str))
        # Also test non-integer floats
        eb_float = np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0])
        with self.assertRaisesRegex(ValueError, "Non-integer values found"):
             list(self._run_and_collect(self.design_6, 1, self.rng, exchangeability_matrix=eb_float))

    def test_error_design_type(self):
        """Error: TypeError for non-numpy design matrix."""
        design_list = [[1, 2], [3, 4]]
        with self.assertRaisesRegex(TypeError, "design must be a numpy array"):
             list(self._run_and_collect(design_list, 1, self.rng))

    def test_edge_single_row(self):
        """Edge Case: Design matrix with a single row."""
        design_1 = np.array([[100, 200]])
        eb_1 = np.array([1])
        eb_1_multi = np.array([[1, -1]])
        for eb in [None, eb_1, eb_1_multi]:
            for within_flag in [None, True, False]:
                for whole_flag in [None, True, False]:
                    # Simple skip for known error case, might need refinement
                    if whole_flag and isinstance(eb, np.ndarray) and eb.shape[0] == 1: continue
                    try:
                        perms = self._run_and_collect(design_1, 3, self.rng, eb, within_flag, whole_flag)
                        self.assertEqual(len(perms), 3)
                        for p in perms:
                            np.testing.assert_array_equal(p, design_1, f"Failed for eb={eb}, w={within_flag}, wh={whole_flag}")
                    except ValueError as e:
                        # Fail only if error is unexpected
                        if "requires all blocks to be the same size" in str(e) and whole_flag:
                             pass # Expected error for whole-block on single element potentially
                        else:
                             self.fail(f"Unexpected ValueError for eb={eb}, w={within_flag}, wh={whole_flag}: {e}")

    def test_edge_zero_permutations(self):
        """Edge Case: Requesting zero permutations."""
        perms = self._run_and_collect(self.design_6, 0, self.rng)
        self.assertEqual(len(perms), 0)

    def test_edge_empty_design(self):
        """Edge Case: Empty design matrix."""
        design_empty = np.empty((0, 2))
        eb_empty = np.empty((0, 1))
        perms = self._run_and_collect(design_empty, 5, self.rng, exchangeability_matrix=eb_empty)
        self.assertEqual(len(perms), 0)
        perms_no_eb = self._run_and_collect(design_empty, 5, self.rng)
        self.assertEqual(len(perms_no_eb), 0)

if __name__ == "__main__":
    unittest.main()