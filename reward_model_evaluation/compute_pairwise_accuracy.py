import argparse
import json

from modular_rm_eval.scoring import evaluate_directory


def build_parser():
    parser = argparse.ArgumentParser(description="Compute pairwise accuracy from inference result json files")
    parser.add_argument("--result-dir", required=True, type=str)
    parser.add_argument("--task-type", required=True, choices=["edit", "gen"])
    parser.add_argument("--mode", default="all", choices=["all", "text", "visual"])
    parser.add_argument(
        "--label-source",
        default="chosen",
        choices=["chosen", "ground_truth_fields"],
        help="How to derive ground-truth ranking",
    )
    parser.add_argument("--tolerance", default=0.0, type=float)
    parser.add_argument("--fallback-score", default=2.5, type=float)
    parser.add_argument("--show-failures", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = evaluate_directory(
        result_dir=args.result_dir,
        task_type=args.task_type,
        mode=args.mode,
        label_source=args.label_source,
        tolerance=args.tolerance,
        fallback=args.fallback_score,
    )

    print(f"Result dir: {result['result_dir']}")
    print(f"Task type: {result['task_type']}")
    print(f"Mode: {result['mode']}")
    print(f"Label source: {result['label_source']}")
    print(f"Total evaluated: {result['total_evaluated']}")
    print(f"Passed: {result['passed']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    if args.show_failures and result["failed_cases"]:
        print(json.dumps(result["failed_cases"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

