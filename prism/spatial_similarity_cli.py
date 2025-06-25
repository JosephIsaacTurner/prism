import argparse
import os
from prism.datasets import Dataset
from prism.spatial_similarity import spatial_similarity_permutation_analysis
from prism.notebook_utils import pretty_print_all_datasets, save_and_summarize_similarity


def parse_args():
    parser = argparse.ArgumentParser(description='Run spatial similarity analysis from configs.')
    parser.add_argument('--dataset-configs', nargs='+', required=True,
                        help='Paths to dataset config JSON files.')
    parser.add_argument('--reference-maps', nargs='*', default=None,
                        help='Paths to reference map image files.')
    parser.add_argument('--mask_img', default=None,
                        help='Path to mask image file (optional).')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save output results.')
    parser.add_argument('--n-permutations', type=int, default=1000,
                        help='Number of permutations for analysis.')
    parser.add_argument('--two-tailed', action='store_true',
                        help='Use two-tailed test.')
    parser.add_argument('--accel-tail', action='store_true',
                        help='Use accelerated tail approximation.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets from config files
    datasets = []
    for cfg_path in args.dataset_configs:
        ds = Dataset(config_path=cfg_path)
        ds.n_permutations = args.n_permutations
        ds.mask_img = args.mask_img if args.mask_img else None
        datasets.append(ds)

    pretty_print_all_datasets(datasets)

    # Run spatial similarity analysis
    results = spatial_similarity_permutation_analysis(
        datasets=datasets,
        reference_maps=args.reference_maps,
        two_tailed=args.two_tailed,
        accel_tail=args.accel_tail
    )

    # Prepare names for saving
    dataset_names = [os.path.basename(ds.output_prefix) for ds in datasets]
    reference_names = [os.path.basename(r) for r in args.reference_maps] if args.reference_maps else None

    # Save and summarize results
    save_and_summarize_similarity(
        results=results,
        dataset_names=dataset_names,
        reference_names=reference_names,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
