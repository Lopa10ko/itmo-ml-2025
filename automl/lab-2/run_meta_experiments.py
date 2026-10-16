import argparse

from lab2_experiment import (
    TARGET_DATASETS,
    export_completed_runs,
    run_experiments,
    run_invariance_check,
)


parser = argparse.ArgumentParser(description="Run or resume the AutoML Lab 2 experiment.")
parser.add_argument("--target", type=int, default=TARGET_DATASETS)
parser.add_argument("--max-new", type=int)
parser.add_argument("--refresh-catalog", action="store_true")
parser.add_argument("--export-only", action="store_true")
parser.add_argument("--skip-invariance", action="store_true")
args = parser.parse_args()

if args.export_only:
    frame = export_completed_runs()
    print(f"Exported {len(frame)} completed runs.")
else:
    if not args.skip_invariance:
        check = run_invariance_check()
        print(f"Invariance checks passed: {int(check['Invariant'].sum())}/{len(check)}")
    run_experiments(
        target=args.target,
        max_new=args.max_new,
        refresh_catalog=args.refresh_catalog,
    )
