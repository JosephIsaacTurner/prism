from math import e
import subprocess
from getpass import getpass
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from IPython.display import display
from sklearn.linear_model import LinearRegression
from nilearn.maskers import NiftiMasker


def pretty_print_df_info(df):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    try:
        print(f"{GREEN}✔ Found CSV with {df.shape[0]:,} rows and {df.shape[1]} columns.{RESET}")

        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0]

        if not nan_columns.empty:
            print(f"{YELLOW}⚠ Columns with missing (NaN) values:{RESET}")
            for col, count in nan_columns.items():
                print(f"  {RED}{col}{RESET}: {count:,} NaN values")
        else:
            print(f"{GREEN}✅ No missing (NaN) values found in any columns.{RESET}")
        print(f"\n{YELLOW}📊 Preview of first few rows:{RESET}")
        display(df.head(5))

    except Exception as e:
        print(f"{RED}❌ Error processing DataFrame: {e}{RESET}")
        return


def pretty_print_input_images_info(input_images):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"

    try:
        total = len(input_images)
        n_missing = sum(pd.isna(input_images))
        n_bad_ext = 0
        n_missing_file = 0
        n_unreadable = 0

        print(f"{GREEN}✔ Checking {total} input image paths...{RESET}")

        for i, path in enumerate(input_images):
            if pd.isna(path):
                continue

            if not (path.endswith(".nii") or path.endswith(".nii.gz")):
                n_bad_ext += 1

            if not os.path.exists(path):
                n_missing_file += 1
            elif not os.access(path, os.R_OK):
                n_unreadable += 1

        if n_missing > 0:
            print(f"{RED}⚠ {n_missing} paths are missing (NaN).{RESET}")
        if n_bad_ext > 0:
            print(f"{YELLOW}⚠ {n_bad_ext} files do not end in '.nii' or '.nii.gz'.{RESET}")
        if n_missing_file > 0:
            print(f"{RED}⚠ {n_missing_file} files do not exist on disk.{RESET}")
        if n_unreadable > 0:
            print(f"{RED}⚠ {n_unreadable} files exist but are not readable (check permissions).{RESET}")

        if all([
            n_missing == 0,
            n_bad_ext == 0,
            n_missing_file == 0,
            n_unreadable == 0
        ]):
            print(f"{GREEN}✅ All image paths exist and you have read permissions.{RESET}")
    except Exception as e:
        print(f"{RED}❌ Error checking input images: {e}{RESET}")


def pretty_print_design_info(variables_of_interest, nuisance_variables, design):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    try:
        print(f"{GREEN}✔ Design matrix created with {design.shape[0]:,} rows and {design.shape[1]} columns.{RESET}")
        print(f"{YELLOW}Variables of interest:{RESET} {', '.join(variables_of_interest) or '(none)'}")
        print(f"{YELLOW}Nuisance variables:{RESET} {', '.join(nuisance_variables) or '(none)'}\n")

        print(f"{CYAN}📊 Column-by-column summary:{RESET}")
        continuous_cols = []
        for col in design.columns:
            dtype = design[col].dtype
            nunique = design[col].nunique(dropna=True)
            n_missing = design[col].isna().sum()

            if pd.api.types.is_numeric_dtype(design[col]):
                var_type = "categorical" if nunique <= 10 else "continuous"
            else:
                var_type = "non-numeric"

            msg = f"- {col}: {var_type}, {nunique} unique"
            if n_missing > 0:
                msg += f", {RED}{n_missing} NaNs{RESET}"
            else:
                msg += f", {GREEN}no NaNs{RESET}"

            if var_type == "non-numeric":
                msg += f" {RED}[non-numeric!]{RESET}"

            if var_type == "continuous":
                continuous_cols.append(col)
                std = design[col].std()
                max_val = design[col].max()
                min_val = design[col].min()
                range_val = max_val - min_val

                # Flag wide ranges
                if range_val > 100 or std > 50:
                    msg += f" {YELLOW}[wide range: {min_val:.1f}–{max_val:.1f}, std={std:.1f}] consider scaling{RESET}"

                # Flag outliers using z-score
                z = np.abs(zscore(design[col].fillna(0)))
                n_outliers = (z > 3).sum()
                if n_outliers > 0:
                    msg += f" {RED}[{n_outliers} extreme outliers >3 SD]{RESET}"

            print(msg)

        # Multicollinearity check
        if len(continuous_cols) > 1:
            corr_matrix = design[continuous_cols].corr()
            print(f"\n{CYAN}🧠 Multicollinearity check (Pearson r > 0.8):{RESET}")
            warned = False
            for i in range(len(continuous_cols)):
                for j in range(i + 1, len(continuous_cols)):
                    c1, c2 = continuous_cols[i], continuous_cols[j]
                    r = corr_matrix.loc[c1, c2]
                    if abs(r) > 0.8:
                        print(f"- {RED}{c1} vs {c2}: r = {r:.2f} [⚠ strong collinearity]{RESET}")
                        warned = True
            if not warned:
                print(f"{GREEN}✓ No strong collinearity detected.{RESET}")
        
        print(f"\n{CYAN}📝 Preview of first few rows of design matrix:{RESET}")
        display(design.head(5))

    except Exception as e:
        print(f"{RED}❌ Error processing design matrix: {e}{RESET}")


def pretty_print_contrast_info(contrast, design, variables_of_interest, nuisance_variables):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    try:
        contrast_cols = set(contrast.columns)
        design_cols = set(design.columns)
        missing_from_contrast = design_cols - contrast_cols
        extra_in_contrast = contrast_cols - design_cols

        print(f"{GREEN}✔ Contrast vector created with {contrast.shape[1]} columns.{RESET}")
        print(f"{YELLOW}Variables of interest:{RESET} {', '.join(variables_of_interest) or '(none)'}")
        print(f"{YELLOW}Nuisance variables:{RESET} {', '.join(nuisance_variables) or '(none)'}\n")

        if missing_from_contrast:
            print(f"{RED}⚠ Warning: contrast missing values for these design columns:{RESET} {', '.join(missing_from_contrast)}")
        if extra_in_contrast:
            print(f"{RED}⚠ Warning: contrast includes columns not in design matrix:{RESET} {', '.join(extra_in_contrast)}")

        print(f"\n{CYAN}📊 Contrast column contributions:{RESET}")
        for col in design.columns:
            val = contrast[col].values[0] if col in contrast.columns else None
            if val is None:
                print(f"- {col}: {RED}MISSING from contrast{RESET}")
                continue

            if col in variables_of_interest:
                if val == 0:
                    print(f"- {col}: {RED}{val} (⚠ variable of interest is zeroed — should usually be 1){RESET}")
                else:
                    print(f"- {col}: {GREEN}{val} (✓ variable of interest — included in effect){RESET}")
            elif col in nuisance_variables:
                if val != 0:
                    print(f"- {col}: {RED}{val} (⚠ nuisance variable should usually be 0){RESET}")
                else:
                    print(f"- {col}: {GREEN}{val} (✓ nuisance regressor — properly controlled){RESET}")
            else:
                print(f"- {col}: {val} (not flagged as interest or nuisance)")
        print("\nPreview of the contrast matrix:")
        display(contrast)
    except Exception as e:
        print(f"{RED}❌ Error processing contrast matrix: {e}{RESET}")

def pretty_print_f_contrast_info(f_contrast_indices, contrast):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    try:
        print(f"{GREEN}✔ F-contrast created with {len(f_contrast_indices)} indices.{RESET}")
        print(f"{YELLOW}Indices for F-contrast:{RESET} {', '.join(map(str, f_contrast_indices))}\n")
        if len(f_contrast_indices) > contrast.shape[0]:
            print(f"{RED}⚠ Warning: F-contrast indices length ({len(f_contrast_indices)}) is greater than number of contrast rows ({contrast.shape[0]}).{RESET}")
        elif len(f_contrast_indices) < contrast.shape[0]:
            print(f"{RED}⚠ Warning: F-contrast indices length ({len(f_contrast_indices)}) is less than number of contrast rows ({contrast.shape[0]}).{RESET}")
        else:
            print(f"{GREEN}✓ F-contrast indices match the number of contrast rows.{RESET}")
        print(f"\n{CYAN}📊 F-contrast preview:{RESET}")
        display(contrast.iloc[f_contrast_indices.astype(bool)])
    except Exception as e:
        print(f"{RED}❌ Error processing F-contrast: {e}{RESET}")


def pretty_print_exchangeability_matrix(exchangeability_matrix, design, variables_of_interest):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    try:    
        # Ensure input is a pandas Series
        if isinstance(exchangeability_matrix, (np.ndarray, list)):
            exchangeability_matrix = pd.Series(np.ravel(exchangeability_matrix), name="exchangeability_block")

        print(f"{CYAN}🧩 Exchangeability Matrix Summary:{RESET}")
        print(f"- Total observations: {len(exchangeability_matrix)}")
        print(f"- Distinct blocks: {exchangeability_matrix.nunique()}")

        block_counts = dict(exchangeability_matrix.value_counts().sort_index())
        block_counts = {k: int(v) for k, v in block_counts.items()}
        print(f"- Subjects per block: {dict(block_counts)}")

        # Check 1: All integers
        if not np.issubdtype(exchangeability_matrix.dtype, np.integer):
            print(f"{RED}⚠ Warning: exchangeability block values are not integers!{RESET}")
        elif (exchangeability_matrix < 0).any():
            print(f"{RED}⚠ Warning: negative integers detected in exchangeability blocks. Is this expected?{RESET}")
        else:
            print(f"{GREEN}✔ Block values are valid integers.{RESET}")

        # Check 2: Match design matrix rows
        if len(exchangeability_matrix) != design.shape[0]:
            print(f"{RED}❌ Error: exchangeability matrix length ({len(exchangeability_matrix)}) does not match number of observations in design matrix ({design.shape[0]}).{RESET}")
        else:
            print(f"{GREEN}✔ Length matches design matrix rows.{RESET}")

        # Check 3: Colinearity with variables of interest
        colinear_vois = []
        for var in variables_of_interest:
            if var not in design.columns:
                continue
            model = LinearRegression().fit(pd.get_dummies(exchangeability_matrix, drop_first=True), design[var])
            r2 = model.score(pd.get_dummies(exchangeability_matrix, drop_first=True), design[var])
            if r2 > 0.99:
                colinear_vois.append((var, r2))

        if colinear_vois:
            print(f"{RED}⚠ Warning: exchangeability structure may be colinear with variable(s) of interest:{RESET}")
            for var, r2 in colinear_vois:
                print(f"  - {var}: R² = {r2:.3f} (too similar to block structure)")
            print(f"{YELLOW}This may invalidate permutations — VOIs should not be fully determined by the exchangeability blocks. If they are, it means you're not actually allowing permutation of that variable, which defeats the purpose of testing its effect.{RESET}")
        else:
            print(f"{GREEN}✔ No strong colinearity detected between exchangeability blocks and variables of interest.{RESET}")
    
    except Exception as e:
        print(f"{RED}❌ Error processing exchangeability matrix: {e}{RESET}")

def pretty_print_analysis_summary(analysis_type, n_permutations, exchangeability_blocks, f_contrast_indices, design, contrast, input_images, variables_of_interest, nuisance_variables):
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    try:
        print(f"{CYAN}🧠 Analysis Summary:{RESET}")
        print(f"- Analysis type: {YELLOW}{analysis_type.upper()}{RESET}")
        print(f"- Number of input images: {len(input_images)}")
        print(f"- Design matrix: {design.shape[0]} rows × {design.shape[1]} columns")
        print(f"- Contrast shape: {contrast.shape}")
        print(f"- Variables of interest: {', '.join(variables_of_interest) or '(none)'}")
        print(f"- Nuisance variables: {', '.join(nuisance_variables) or '(none)'}")
        print(f"- Permutations: {n_permutations} (Freedman–Lane scheme)")

        if exchangeability_blocks is not None:
            print(f"- Exchangeability blocks: {YELLOW}enabled{RESET} (non-independent samples)")

        if f_contrast_indices is not None:
            print(f"- F-contrast indices: {f_contrast_indices} (will test joint effect)\n")
        else:
            print()

        print(f"{CYAN}📐 GLM Model Structure:{RESET}")
        print("We model the voxelwise imaging data (Y) as a linear combination of:")
        print(f"- X: the variable(s) of interest ({', '.join(variables_of_interest) or 'none'})")
        print(f"- Z: nuisance regressors ({', '.join(nuisance_variables) or 'none'})")
        print("So the model looks like:")
        print(f"{GREEN}    Y = Xβ + Zγ + ε{RESET}")

        print(f"\n{CYAN}🧪 Hypothesis Testing:{RESET}")

        if contrast.shape[0] == 1:
            print("- A single contrast vector is provided. We'll test the effect of that specific linear combination of regressors on Y.")
        else:
            print(f"- Multiple contrast vectors ({contrast.shape[0]}) are provided. Each will be tested {YELLOW}separately{RESET}, generating one statistical map per row.")

        if f_contrast_indices is not None:
            print(f"- An {YELLOW}F-contrast{RESET} will also be computed to jointly test the null hypothesis that *all* effects at indices {f_contrast_indices} are simultaneously zero.")
            print("  This is a one-tailed test, assessing whether any linear combination of the selected contrasts explains significant variance in Y.")
            print("  Under the hood, this corresponds to an F-statistic (or G-statistic if using exchangeability blocks).")

        if analysis_type == "t":
            if f_contrast_indices is not None:
                stat_type = "F" if exchangeability_blocks is None else "G"
                print(f"- The joint test will use a {YELLOW}{stat_type}-statistic{RESET} for the F-contrast.")
            else:
                stat_type = "t" if exchangeability_blocks is None else "v"
                print(f"- We'll compute a {YELLOW}{stat_type}-statistic{RESET} to quantify how strong the effect of X is.")
        elif analysis_type == "pearson":
            print(f"- We'll compute {YELLOW}Pearson correlations{RESET} between X and Y, while controlling for Z. This is equivalent to a partial correlation.")
        else:
            print(f"{RED}⚠ Unknown analysis type: {analysis_type}{RESET}")

        print(f"\n{CYAN}🔁 Permutation Inference:{RESET}")
        print(f"To assess statistical significance, we'll run {n_permutations} permutations using the {YELLOW}Freedman–Lane{RESET} method.")
        print(f"For FDR correction, we'll use the {YELLOW}Benjamini–Hochberg{RESET} procedure on the uncorrected p-values.")
        print(f"For FWE significance, we'll use the {YELLOW}max{RESET} statistic across all permutations (i.e., the {YELLOW}Westfall-Young{RESET} method).")

        if exchangeability_blocks is not None:
            print(f"- Exchangeability blocks are being used, meaning the test statistics will change from:")
            print(f"  {GREEN}t → v{RESET} (Aspin-Welch v-statistic)")
            print(f"  {GREEN}f → g{RESET} (G-statistic)")

        print(f"\n{CYAN}🧾 Output Files:{RESET}")
        print(f"- The final outputs will be saved as NIfTI files (*.nii/.nii.gz) or NumPy arrays (*.npy), depending on context.")
        print(f"- For each contrast or F-test, the following maps will be generated:")
        print(f"  • Raw test statistic map (e.g., t, v, F, G, r)")
        print(f"  • Uncorrected p-value map, FDR-corrected p-value map, FWE-corrected p-value map")

        print(f"\n{GREEN}✔ Setup complete. Ready to run the analysis.{RESET}")
    except Exception as e:
        print(f"{RED}❌ Error generating analysis summary: {e}{RESET}")


def submit_slurm_job(
    hostname,
    username,
    email,
    conda_env,
    output_dir,
    output_basename,
    input_images,
    design,
    contrast,
    analysis_type="t",
    n_permutations=5000,
    f_contrast_indices=None,
    eb_matrix=None,
    two_tailed=True,
    mask_img=None,
    tfce=False,
    accel_tail=True,
    save_1minusp=True,
    save_neglog10p=False,
    save_permutations=False,
    demean=False,
    flip_signs=False,
    vg_auto=False,
    correct_across_contrasts=False,
    zstat=False,
    use_legacy_palm=False,
    n_cores=12,
    memory="16000",
    time="12:00:00",
    dry_run=False
):
    os.makedirs(output_dir, exist_ok=True)

    # Prepare input images
    if use_legacy_palm:
        print("Using legacy PALM; ensure FSL/MATLAB env is loaded.")
        print("Concatenating into a 4D NIfTI may use lots of space.")
        masker = NiftiMasker(mask_img=mask_img) if mask_img else NiftiMasker()
        local_cores = 1
        input_images_filepath = os.path.join(
            output_dir, f"{output_basename}_4d_input_images_concat.nii"
        )
        if os.path.exists(input_images_filepath):
            print(f"Skipping concatenation; {input_images_filepath} exists.")
        else:
            data = masker.fit_transform(input_images)
            img4d = masker.inverse_transform(data)
            img4d.to_filename(input_images_filepath)
            print(f"Saved concatenated images to {input_images_filepath}")
        env_setup = ". ~/.bashrc; module load MATLAB/2019b;"
    else:
        local_cores = n_cores
        input_images_filepath = os.path.join(
            output_dir, f"{output_basename}_input_images.csv"
        )
        pd.DataFrame(input_images, columns=['image']) \
          .to_csv(input_images_filepath, index=False, header=False)
        env_setup = ". ~/.bashrc;"

    # Save design & contrast
    design_fp = os.path.join(output_dir, f"{output_basename}_design.csv")
    contrast_fp = os.path.join(output_dir, f"{output_basename}_contrast.csv")
    design.to_csv(design_fp, index=False, header=False)
    contrast.to_csv(contrast_fp, index=False, header=False)

    # Build the command
    cmd = "palm" if use_legacy_palm else "prism"
    cmd += f" -i {input_images_filepath}"
    cmd += f" -d {design_fp}"
    cmd += f" -t {contrast_fp}"
    cmd += f" -n {n_permutations}"
    cmd += f" -o {os.path.join(output_dir, output_basename)}"
    cmd += " -fdr -ee"

    if mask_img:
        cmd += f" -m {mask_img}"
    if f_contrast_indices is not None:
        f_fp = os.path.join(output_dir, f"{output_basename}_f_contrast_indices.csv")
        pd.DataFrame(f_contrast_indices).to_csv(f_fp, index=False, header=False)
        cmd += f" -f {f_fp}"
    if eb_matrix is not None:
        eb_fp = os.path.join(output_dir, f"{output_basename}_eb_matrix.csv")
        pd.DataFrame(eb_matrix).to_csv(eb_fp, index=False, header=False)
        cmd += f" -eb {eb_fp}"

    # Additional flags
    if vg_auto:                cmd += " -vg auto"
    if two_tailed:             cmd += " -twotail"
    if analysis_type == "pearson": cmd += " -pearson"
    if zstat:                  cmd += " -zstat"
    if accel_tail:             cmd += " -accel tail"
    if demean:                 cmd += " -demean"
    if correct_across_contrasts: cmd += " -corrcon"
    if tfce:                   cmd += " -T"
    if flip_signs:             cmd += " -ise"
    if save_1minusp:           cmd += " -save1-p"
    elif save_neglog10p:       cmd += " -logp"
    if save_permutations:      cmd += " -saveperms"

    # Create SLURM script
    job_script = f"""#!/bin/bash
#SBATCH -p nimlab,normal,bigmem,long
#SBATCH -c {local_cores}
#SBATCH --mem {memory}
#SBATCH -o {output_dir}/slurm.%N.%j.out
#SBATCH -e {output_dir}/slurm.%N.%j.err
#SBATCH -t {time}
#SBATCH --mail-user={email}
#SBATCH --mail-type=END

{env_setup}
conda activate {conda_env}

{cmd}
"""
    script_fp = os.path.join(output_dir, f"{output_basename}_palm_job.sh")
    with open(script_fp, "w") as f:
        f.write(job_script)
    print(f"Job script saved to {script_fp}")
    print(f"Submit with: \nsbatch {script_fp}")

    if not dry_run:
        # Submit via SSH
        ssh_cmd = f"echo sbatch {script_fp} | sshpass -p {getpass('Enter your cluster password: ')} ssh {username}@{hostname}"
        subprocess.run(ssh_cmd, shell=True)
        print(f"Job submitted to {hostname} as {username}.")
