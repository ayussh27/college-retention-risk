"""Run the whole pipeline in order:  python -m src.run_all

clean -> data dictionary -> baselines. Needs the raw CSV in Dataset/ for the first step;
pass --skip-clean to reuse the committed data/processed/cohort.csv instead.
"""
import sys

from src import baselines, clean, data_dictionary


def main() -> None:
    steps = [("clean", clean.main), ("data dictionary", data_dictionary.main), ("baselines", baselines.main)]
    if "--skip-clean" in sys.argv:
        steps = steps[1:]
    for name, step in steps:
        print(f"\n=== {name} ===")
        step()


if __name__ == "__main__":
    main()
