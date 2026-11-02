"""Plot recorded training loss or path spring from an experiment directory."""

import argparse
import csv
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quantity", choices=("train-loss", "zeta"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16777216,
                        help="Validation tokens used for zeta (default: 16777216).")
    args = parser.parse_args()
    points = []
    if args.quantity == "train-loss":
        log = args.run_dir / "training" / "train.log"
        if not log.is_file():
            parser.error(f"training log missing: {log}")
        pattern = re.compile(r"^step:(\d+)/\d+\s+train_loss:([\d.eE+-]+)")
        for line in log.read_text().splitlines():
            if match := pattern.match(line):
                points.append((int(match[1]), float(match[2])))
        xlabel, ylabel = "Completed training updates", "Training loss per token"
        title = "Training loss"
    else:
        for path in sorted((args.run_dir / "spectra").glob("update_*/eigenvalues.csv")):
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if int(row["mode"]) == 1 and int(row["tokens"]) == args.tokens:
                        points.append((int(row["update"]), float(row["eigenvalue"])))
        xlabel, ylabel = "Training update $t$", r"Path spring $\zeta_t$"
        title = f"Path spring ({args.tokens:,} validation tokens)"
    if not points:
        parser.error(f"no {args.quantity} measurements found in {args.run_dir}; "
                     "older training logs lack train_loss; for zeta, check --tokens")
    points.sort()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.plot(*zip(*points), color="#0072B2", linewidth=1.3,
              marker="o" if args.quantity == "zeta" else None, markersize=3)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.grid(alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
