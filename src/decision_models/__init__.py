import argparse
import importlib

from decision_models.samples import SAMPLES, module_name


def main(argv: list[str] | None = None) -> None:
    """Run one of the samples, e.g. `uv run decision-models triage`. Without a name, list the samples."""
    parser = argparse.ArgumentParser(prog='decision-models', description='Run one of the decision model samples.')
    parser.add_argument('sample', nargs='?', choices=SAMPLES, help='the sample to run')
    args = parser.parse_args(argv)

    if args.sample is None:
        print('Available samples (run one with `uv run decision-models <sample>`):')
        for name, description in SAMPLES.items():
            print(f'  {name:<18} {description}')
        return

    importlib.import_module(module_name(args.sample)).main()
