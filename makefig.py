"""Plot recorded training loss or path spring from an experiment directory."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_train_loss(run_dir: Path, output: Path) -> None:
    log = run_dir / "training" / "train.log"
    if not log.is_file():
        raise ValueError(f"training log missing: {log}")
    pattern = re.compile(r"^step:(\d+)/\d+\s+train_loss:([\d.eE+-]+)")
    points = []
    for line in log.read_text().splitlines():
        if match := pattern.match(line):
            points.append((int(match[1]), float(match[2])))
    if not points:
        raise ValueError(f"no train-loss measurements found in {run_dir}; "
                         "older training logs lack train_loss")
    points.sort()

    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.plot(*zip(*points), color="#0072B2", linewidth=1.3)
    axis.set(xlabel="Completed training updates", ylabel="Training loss per token",
             title="Training loss")
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


def plot_zeta(run_dir: Path, output: Path, tokens: int = 16777216) -> None:
    points = []
    for path in sorted((run_dir / "spectra").glob("update_*/eigenvalues.csv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["mode"]) == 1 and int(row["tokens"]) == tokens:
                    points.append((int(row["update"]), float(row["eigenvalue"])))
    if not points:
        raise ValueError(f"no zeta measurements found in {run_dir}; check --tokens")
    points.sort()

    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.plot(*zip(*points), color="#0072B2", linewidth=1.3,
              marker="o", markersize=3)
    axis.set(xlabel="Training update $t$", ylabel=r"Path spring $\zeta_t$",
             title=f"Path spring ({tokens:,} validation tokens)")
    axis.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quantity", choices=("train-loss", "zeta"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16777216,
                        help="Validation tokens used for zeta (default: 16777216).")
    args = parser.parse_args()
    try:
        if args.quantity == "train-loss":
            plot_train_loss(args.run_dir, args.output)
        else:
            plot_zeta(args.run_dir, args.output, args.tokens)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
