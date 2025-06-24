import numpy as np


def permute_indices_recursive(
    current_original_indices, level, eb_matrix, rng, parent_instructed_fix_order=False
):
    """
    Recursively permute indices based on the exchangeability matrix.
    Args:
        current_original_indices (np.ndarray): Current indices to permute.
        level (int): Current level in the exchangeability matrix.
        eb_matrix (np.ndarray): Exchangeability matrix.
        rng (np.random.Generator): Random number generator.
        parent_instructed_fix_order (bool): Whether the parent instructed to fix order.
    Returns:
        np.ndarray: Permuted indices.
    """
    if len(current_original_indices) == 0:
        return np.array([], dtype=int)
    if level >= eb_matrix.shape[1]:
        return rng.permutation(current_original_indices)

    is_last_defined_level = level == eb_matrix.shape[1] - 1
    current_eb_level_values = eb_matrix[current_original_indices, level]
    unique_blocks, block_inverse_indices = np.unique(
        current_eb_level_values, return_inverse=True
    )
    n_unique_blocks_at_level = len(unique_blocks)

    # --- Single Block Logic ---
    if n_unique_blocks_at_level == 1:
        block_val = unique_blocks[0]
        if block_val == 0:
            raise ValueError(
                f"Block index 0 found at level {level} for indices subset {current_original_indices[:5]}..., which is not supported."
            )

        if is_last_defined_level:  # Explicit termination for last level
            if block_val > 0:
                return rng.permutation(current_original_indices)
            else:
                return np.copy(current_original_indices)  # Neg@Last = Identity
        else:  # Not last defined level - recurse
            instruct_fix_next = block_val < 0  # Instruction for NEXT level
            if block_val > 0:
                # Positive: Shuffle order of sub-blocks found at next level.
                next_eb_level_values = eb_matrix[current_original_indices, level + 1]
                unique_sub_blocks, sub_block_inverse, sub_block_counts = np.unique(
                    next_eb_level_values, return_inverse=True, return_counts=True
                )
                n_sub_blocks = len(unique_sub_blocks)
                if n_sub_blocks <= 1:
                    # Pass parent_fix based on current block sign (False here)
                    return permute_indices_recursive(
                        current_original_indices,
                        level + 1,
                        eb_matrix,
                        rng,
                        parent_instructed_fix_order=instruct_fix_next,
                    )
                if len(np.unique(sub_block_counts)) > 1:
                    # Corrected Msg
                    raise ValueError(
                        f"Level {level} (positive index {block_val}) requires sub-blocks "
                        f"defined by level {level + 1} to be uniform size for whole-block shuffling. "
                        f"Indices subset starting with: {current_original_indices[:5]}.... "
                        f"Sub-block IDs: {unique_sub_blocks}. "
                        f"Sub-block sizes: {sub_block_counts}."
                    )
                sub_block_indices_list = [
                    current_original_indices[sub_block_inverse == i]
                    for i in range(n_sub_blocks)
                ]
                shuffled_sub_block_order = rng.permutation(n_sub_blocks)
                # Pass down fix instruction based on the SUB-BLOCK'S sign
                permuted_sub_blocks = [
                    permute_indices_recursive(
                        sub_block_indices_list[i],
                        level + 1,
                        eb_matrix,
                        rng,
                        parent_instructed_fix_order=(unique_sub_blocks[i] < 0),
                    )
                    for i in range(n_sub_blocks)
                ]
                return np.concatenate(
                    [permuted_sub_blocks[idx] for idx in shuffled_sub_block_order]
                )
            else:  # block_val < 0
                # Negative: Recurse, instructing next level to fix order.
                return permute_indices_recursive(
                    current_original_indices,
                    level + 1,
                    eb_matrix,
                    rng,
                    parent_instructed_fix_order=True,
                )

    # --- Multi Block Logic ---
    else:  # n > 1
        signs = np.sign(unique_blocks)
        if np.any(unique_blocks == 0):
            raise ValueError(
                f"Block index 0 found at level {level} among {unique_blocks} for indices {current_original_indices[:5]}..., which is not supported."
            )
        if len(np.unique(signs)) > 1:
            raise ValueError(
                f"Level {level}: Mixed positive/negative block indices found "
                f"({unique_blocks}) within the same parent block structure "
                f"for indices starting with {current_original_indices[:5]}..., which is ambiguous and not supported by PALM."
            )

        # *** Prioritize Parent Instruction & Last Level Check ***
        if parent_instructed_fix_order:
            # Parent said fix order -> MUST concatenate this level in order. Recurse within.
            permuted_indices_list = []
            for i, block_val_i in enumerate(unique_blocks):
                mask = block_inverse_indices == i
                indices_in_this_block_i = current_original_indices[mask]
                # Recurse, passing instruction based on this block's sign
                instruct_fix_i = block_val_i < 0
                permuted_subset = permute_indices_recursive(
                    indices_in_this_block_i,
                    level + 1,
                    eb_matrix,
                    rng,
                    parent_instructed_fix_order=instruct_fix_i,
                )
                permuted_indices_list.append(permuted_subset)
            return np.concatenate(permuted_indices_list)  # Concat in order

        elif is_last_defined_level:
            # Parent allowed shuffle AND this is the last level. Terminate based on signs.
            if signs[0] > 0:
                # Freely permute all involved indices together.
                return rng.permutation(current_original_indices)
            else:  # signs[0] < 0
                # Identity for each block, concatenate in order.
                permuted_indices_list = [
                    np.copy(current_original_indices[block_inverse_indices == i])
                    for i in range(n_unique_blocks_at_level)
                ]
                return np.concatenate(permuted_indices_list)
        else:
            # Intermediate level AND parent allowed shuffle.
            # Recurse within each block. Concatenate results based on *this* level's signs (Original V1 logic).
            permuted_indices_list = []
            for i, block_val_i in enumerate(unique_blocks):
                mask = block_inverse_indices == i
                indices_in_this_block_i = current_original_indices[mask]
                instruct_fix_i = block_val_i < 0
                permuted_subset = permute_indices_recursive(
                    indices_in_this_block_i,
                    level + 1,
                    eb_matrix,
                    rng,
                    parent_instructed_fix_order=instruct_fix_i,
                )
                permuted_indices_list.append(permuted_subset)

            # Use original concatenation logic because parent_fix is False
            if signs[0] > 0:  # Shuffle order
                shuffled_block_order = rng.permutation(n_unique_blocks_at_level)
                return np.concatenate(
                    [permuted_indices_list[idx] for idx in shuffled_block_order]
                )
            else:  # Preserve order
                return np.concatenate(permuted_indices_list)


def yield_permuted_indices(
    design,
    n_permutations,
    contrast=None,
    exchangeability_matrix=None,
    within=None,
    whole=None,
    random_state=None,
):
    """Generator for permuting the design matrix per PALM documentation.

    Handles free exchange, within-block, whole-block, combined within/whole,
    and multi-level exchangeability via positive/negative indices.
    Docs: https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)ExchangeabilityBlocks.html

    Args:
        design (np.ndarray): Design matrix. Shape (n_samples, n_features).
        n_permutations (int): Number of permutations to generate.
        random_state (int or np.random.Generator or None): Seed or Generator
            for the random number generator.
        exchangeability_matrix (np.ndarray or None): Matrix or vector defining
            exchangeability blocks. Shape (n_samples,) or (n_samples, n_levels).
            If None, free exchange is assumed. Defaults to None.
        within (bool | None): For single-column blocks, allow shuffling within blocks.
                       If None (default): Behavior depends on 'whole'. If 'whole' is also None or False,
                       defaults to True. If 'whole' is True, defaults to False.
                       Ignored if exchangeability_matrix has >1 column or if None.
        whole (bool | None): For single-column blocks, shuffle blocks as wholes.
                      If None (default): Defaults to False.
                      Ignored if exchangeability_matrix has >1 column or if None.

    Yields:
        np.ndarray: A permuted version of the design matrix.

    Raises:
        ValueError: If inputs are inconsistent (e.g., non-uniform block sizes
                    required for whole-block shuffling, ambiguous multi-col structure,
                    zero indices in eb_matrix).
        TypeError: If design or exchangeability_matrix is not a numpy array or
                   if eb_matrix contains non-numeric data.
    """
    # --- Input Validation ---
    if not isinstance(design, np.ndarray):
        raise TypeError("design must be a numpy array.")
    if design.ndim != 2:
        raise ValueError(
            f"design must be 2D (samples x features), got shape {design.shape}"
        )
    
    design_ = design.copy()  # Work on a copy to avoid modifying the original

    n_samples = design_.shape[0]
    if n_samples == 0:
        # Handle empty design matrix - yield nothing or raise error?
        # Let's yield nothing as n_permutations would be irrelevant.
        return

    # Initialize RNG
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    original_indices = np.arange(n_samples)

    # --- Preprocess exchangeability_matrix (eb_matrix) ---
    is_eb_provided = exchangeability_matrix is not None
    eb_matrix = None
    n_levels = 0
    use_flags = False  # Default

    if is_eb_provided:
        if not isinstance(exchangeability_matrix, np.ndarray):
            raise TypeError("exchangeability_matrix must be a numpy array.")
        if exchangeability_matrix.size == 0 and n_samples > 0:
            raise ValueError(
                "exchangeability_matrix is empty but design matrix is not."
            )
        if exchangeability_matrix.size > 0:
            if not np.issubdtype(exchangeability_matrix.dtype, np.number):
                raise TypeError("exchangeability_matrix must contain numeric indices.")

            # Check for non-integer values that aren't trivially convertible (e.g., 1.5 vs 1.0)
            if not np.all(np.mod(exchangeability_matrix, 1) == 0):
                # Check more robustly if conversion is possible without loss
                try:
                    int_eb = exchangeability_matrix.astype(int)
                    if not np.all(np.isclose(exchangeability_matrix, int_eb)):
                        raise ValueError(
                            "Non-integer values found in exchangeability_matrix."
                        )
                    eb_matrix = int_eb
                except (ValueError, TypeError):
                    raise ValueError(
                        "Non-integer values found in exchangeability_matrix."
                    )
            else:
                eb_matrix = exchangeability_matrix.astype(int)

            if eb_matrix.shape[0] != n_samples:
                raise ValueError(
                    f"exchangeability_matrix rows ({eb_matrix.shape[0]}) "
                    f"must match design matrix rows ({n_samples})."
                )
            if eb_matrix.ndim == 1:
                eb_matrix = eb_matrix.reshape(-1, 1)
            elif eb_matrix.ndim > 2:
                raise ValueError(
                    "exchangeability_matrix cannot have more than 2 dimensions."
                )
            elif (
                eb_matrix.ndim == 0
            ):  # Should be caught by shape[0] check if n_samples > 0
                raise ValueError("exchangeability_matrix cannot be 0-dimensional.")

            n_levels = eb_matrix.shape[1]
            use_flags = n_levels == 1  # Flags only relevant if effectively single-level

            # Final check for 0 indices which are unsupported by PALM logic
            if np.any(eb_matrix == 0):
                raise ValueError(
                    "Exchangeability matrix contains index 0, which is not supported (use positive/negative integers)."
                )

    # --- Determine Effective within/whole for single-level ---
    eff_within = within
    eff_whole = whole
    if use_flags:
        # Apply default logic only if flags are relevant (single level)
        if eff_whole is None:
            eff_whole = False
        if eff_within is None:
            eff_within = not eff_whole  # Default within=True unless whole=True

    # --- Define the permutation function for one iteration ---
    def get_permuted_indices():
        if not is_eb_provided or eb_matrix is None:
            # Case 0: Free exchange (no eb_matrix provided or it was empty)
            return rng.permutation(original_indices)

        # --- Determine strategy based on levels and flags ---
        if use_flags:
            # --- Case 1: Single Level - Use Flags ---
            block_ids = eb_matrix[:, 0]
            unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
            n_unique_blocks = len(unique_blocks)

            # Trivial case: only one block behaves like free exchange
            if n_unique_blocks <= 1:
                return rng.permutation(original_indices)

            if eff_within and eff_whole:
                # Simultaneous within & whole -> Treat as free exchange
                # (Based on VG interpretation suggesting equivalence to simplest case)
                return rng.permutation(original_indices)
            elif eff_whole:
                # Whole-block shuffling
                unique_blocks, inverse, counts = np.unique(
                    block_ids, return_inverse=True, return_counts=True
                )
                if len(np.unique(counts)) > 1:
                    raise ValueError(
                        "Whole-block shuffling requires all blocks to be the same size. "
                        f"Found sizes: {counts} for blocks {unique_blocks}"
                    )
                n_blocks = len(unique_blocks)
                # Group original indices by block ID
                blocks_indices = [
                    original_indices[inverse == i] for i in range(n_blocks)
                ]
                # Shuffle the order of the blocks
                shuffled_block_order = rng.permutation(n_blocks)
                # Concatenate blocks in the new shuffled order
                return np.concatenate([blocks_indices[i] for i in shuffled_block_order])
            elif eff_within:
                # Within-block shuffling
                permuted_indices = np.copy(original_indices)  # Start with identity
                for i in range(n_unique_blocks):
                    mask = inverse == i
                    indices_this_block = original_indices[mask]
                    # Permute the indices *within* this block
                    shuffled_subset = rng.permutation(indices_this_block)
                    # Assign the permuted indices back to the original positions of the block
                    permuted_indices[mask] = shuffled_subset
                return permuted_indices
            else:  # within=False, whole=False explicitly set
                # This state isn't clearly defined by PALM for permutations.
                # Defaulting to free exchange as the least restrictive assumption.
                return rng.permutation(original_indices)
        else:
            # --- Case 2: Multi-Level (Ignore flags) ---
            # Call the recursive helper starting at level 0
            return permute_indices_recursive(original_indices, 0, eb_matrix, rng)

    # --- Generator Loop ---
    for i in range(n_permutations):
        permuted_row_indices = get_permuted_indices()
        # Check if the permutation is valid before yielding
        if len(permuted_row_indices) != n_samples:
            raise RuntimeError(
                f"Permutation {i+1} generated incorrect number of indices: "
                f"{len(permuted_row_indices)}, expected {n_samples}"
            )
        if contrast is not None:
            # If a contrast is provided, permute only the subset of columns in the design matrix that are being tested.
            # Note: This method is the Draper-Stoneman method, which is not what Anderson Winkler recommends.
            # To me, it makes more sense and is easier to implement than the Freedman-Lane method recommended by Anderson Winkler.
            contrast = np.atleast_2d(contrast)
            contrast_indices = np.ravel(contrast[0, :]).astype(bool)
            design_subset = design_[:, contrast_indices]
            design_subset = design_subset[permuted_row_indices, :]
            design_[:, contrast_indices] = design_subset
        else:
            design_ = design_[permuted_row_indices, :]
        yield permuted_row_indices


def get_vg_vector(exchangeability_matrix, within=True, whole=False):
    """
    Calculates the variance group (VG) vector based on exchangeability rules.

    Args:
        exchangeability_matrix (np.ndarray):
            A 1D or 2D numpy array defining exchangeability blocks.
            - For 1D: Integer indices defining blocks. 'within' and 'whole' flags matter.
            - For 2D: Defines nested exchangeability. Flags are ignored.
              - Positive index in col k: Sub-indices in col k+1 shuffle as a whole.
              - Negative index in col k: Sub-indices in col k+1 shuffle within block.
        within (bool, optional):
            If True and exchangeability_matrix is 1D and whole=False,
            indicates within-block exchangeability. Defaults to True.
        whole (bool, optional):
            If True and exchangeability_matrix is 1D, indicates whole-block
            exchangeability. Overrides 'within' if both are True for VG calc.
            Defaults to False.

    Returns:
        np.ndarray: A 1D numpy array of unique integer identifiers (starting from 1)
                    defining the variance groups (vg_vector) for each observation.

    Raises:
        ValueError: If inputs are inconsistent (e.g., non-uniform block sizes
                    required for whole-block shuffling, ambiguous multi-col structure).
        TypeError: If exchangeability_matrix is not a numpy array.
    """

    if not isinstance(exchangeability_matrix, np.ndarray):
        raise TypeError("exchangeability_matrix must be a numpy array.")

    # Check if dtype is already integer
    if not np.issubdtype(exchangeability_matrix.dtype, np.integer):
        # If not integer, check if it contains only integer-like values (e.g., floats like 1.0)
        try:
            # Use np.mod and check closeness to 0 for float precision issues
            is_integer_like = np.all(np.isclose(np.mod(exchangeability_matrix, 1), 0))
        except TypeError:
            # This catches errors if np.mod fails (e.g., non-numeric types)
            raise ValueError(
                "exchangeability_matrix must contain numeric integer-like indices."
            )

        if is_integer_like:
            # If all are integer-like, convert safely
            exchangeability_matrix = exchangeability_matrix.astype(int)
        else:
            # If any are truly non-integer floats (like 1.5), raise specific error
            raise ValueError("Non-integer values found in exchangeability_matrix.")

    # Store original dimension and force to 2D for consistent processing
    original_ndim = exchangeability_matrix.ndim
    if original_ndim == 0:
        raise ValueError("exchangeability_matrix cannot be 0-dimensional.")
    elif original_ndim == 1:
        # Reshape 1D array to a 2D array with one column
        eb_matrix = exchangeability_matrix.reshape(-1, 1)
    else:
        eb_matrix = exchangeability_matrix

    n_observations = eb_matrix.shape[0]
    n_levels = eb_matrix.shape[1]

    if n_observations == 0:
        return np.array([], dtype=int)
    if n_observations == 1:
        return np.ones(1, dtype=int)

    # --- Determine the effective VG rule ---

    use_flags = original_ndim == 1
    # According to the description, multi-column structure overrides flags
    if n_levels > 1:
        use_flags = False

    vg_vector = np.ones(n_observations, dtype=int)  # Default to single group

    # --- Case 1: Use Flags (Original matrix was 1D) ---
    if use_flags:
        block_ids = eb_matrix[:, 0]
        # Handle potentially non-contiguous block IDs by mapping them
        unique_blocks, block_indices = np.unique(block_ids, return_inverse=True)

        # Calculate counts based on the mapped indices
        block_counts = np.bincount(block_indices)

        # If only one effective block, it's always a single VG
        if len(unique_blocks) <= 1:
            return np.ones(
                n_observations, dtype=int
            )  # Correctly handles single block case

        if whole and within:
            # Simultaneous whole- and within-block => freely exchangeable => single VG
            return np.ones(n_observations, dtype=int)
        elif whole:
            # Whole-block shuffling (-whole flag)
            # Check for uniform block sizes using the calculated counts
            if len(np.unique(block_counts)) > 1:
                raise ValueError(
                    "Whole-block shuffling requires all blocks to be the same size. "
                    f"Found sizes: {block_counts}"  # Show counts for unique blocks
                )
            block_size = block_counts[0]
            # VG = position within block (1 to block_size)
            # Generate VG based on position within original blocks
            temp_vg = np.zeros(n_observations, dtype=int)
            current_pos_in_block = {}  # Key: block_id, Value: next position
            for i in range(n_observations):
                block_val = block_ids[i]
                pos = current_pos_in_block.get(block_val, 0)
                temp_vg[i] = pos + 1
                current_pos_in_block[block_val] = pos + 1
            vg_vector = temp_vg

        elif within:
            # Within-block shuffling (-within flag, default)
            # VG = block index (1-based) based on unique values encountered
            vg_vector = block_indices + 1
        else:
            # Neither within nor whole specified -> freely exchangeable -> single VG
            return np.ones(n_observations, dtype=int)

    # --- Case 2: Multi-Column Matrix (Flags ignored) ---
    elif n_levels > 1:
        col_0 = eb_matrix[:, 0]
        col_1 = eb_matrix[:, 1]

        # Determine unique groups based on first column
        unique_l0, indices_l0 = np.unique(col_0, return_inverse=True)

        # Check if the first level implies whole or within block shuffling
        # Assuming uniformity *within each block* defined by unique_l0

        # Check for mixed signs *across* blocks if multiple l0 blocks exist
        if len(unique_l0) > 1 and (np.any(unique_l0 > 0) and np.any(unique_l0 < 0)):
            raise ValueError(
                "Multi-column exchangeability matrix contains mixed positive/negative "
                "indices in the first column across different top-level blocks. "
                "Automatic VG determination for this specific structure is not supported."
            )

        # Determine effective rule (all positive or all negative in first relevant column)
        first_sign = col_0[0]  # Check based on the first entry's sign
        all_positive = np.all(col_0 > 0)
        all_negative = np.all(col_0 < 0)

        if not (all_positive or all_negative):
            # If not uniformly positive or negative, check if it's just one block type
            if len(unique_l0) == 1:
                all_positive = unique_l0[0] > 0
                all_negative = unique_l0[0] < 0
            else:  # Mixed signs within a block or across blocks was checked earlier
                raise ValueError(
                    "Ambiguous multi-column structure: first column indices are not "
                    "consistently positive or negative."
                )

        if all_positive:
            # Positive indices in col 0 -> Whole-block shuffling implied for col 1 groups
            # VG = position within the blocks defined by col 1

            # Need to determine block sizes based on col_1 *within* each col_0 group
            block_sizes = []
            temp_vg = np.zeros(n_observations, dtype=int)

            # Map positions within each unique block defined by col_1
            current_pos_in_sub_block = {}  # key=col_1 value, val=next_pos
            for i in range(n_observations):
                sub_block_val = col_1[i]
                pos = current_pos_in_sub_block.get(sub_block_val, 0)
                temp_vg[i] = pos + 1
                current_pos_in_sub_block[sub_block_val] = pos + 1

            # Now check uniformity of sizes for blocks defined by col_1
            unique_sub_blocks, sub_block_indices, sub_block_counts = np.unique(
                col_1, return_inverse=True, return_counts=True
            )

            # Special case: If overall structure results in only one group -> VG=1
            # e.g. [[1,1],[1,2],[1,3]] -> sub_block_counts = [1,1,1] -> block_size=1
            if len(np.unique(sub_block_counts)) > 1:
                # Check if it's just trivial blocks of size 1
                if not np.all(sub_block_counts == 1):
                    raise ValueError(
                        "Whole-block shuffling implied by positive indices requires "
                        "effective sub-blocks (from the second level) to be the same size. "
                        f"Found sizes based on column 1: {sub_block_counts}"
                    )

            block_size = sub_block_counts[0]
            # Special case check: If block size is 1, it's like free exchangeability
            if block_size == 1:
                return np.ones(n_observations, dtype=int)

            vg_vector = temp_vg  # Use the calculated positions

        elif all_negative:
            # Negative indices in col 0 -> Within-block shuffling implied for col 1 groups
            # VG = index of the block defined by col 1 (make unique IDs 1-based)
            unique_sub_blocks, sub_block_indices = np.unique(col_1, return_inverse=True)

            # If only one effective sub-block, implies single VG
            if len(unique_sub_blocks) <= 1:
                return np.ones(n_observations, dtype=int)

            vg_vector = sub_block_indices + 1

        # The case where neither all_positive nor all_negative should be caught by prior checks

    # --- Fallback for single level (if somehow missed) ---
    elif n_levels == 1:
        # Treat as 1D case with default flags (within=True, whole=False)
        # This case should theoretically be handled by use_flags=True path
        block_ids = eb_matrix[:, 0]
        unique_blocks, block_indices = np.unique(block_ids, return_inverse=True)
        if len(unique_blocks) <= 1:
            return np.ones(n_observations, dtype=int)
        else:
            vg_vector = block_indices + 1  # Default 'within' logic

    return vg_vector.astype(int)