import sys

from amanda.cli.utils import import_from_name


def cli():
    func_name = sys.argv[1]
    func = import_from_name(func_name)
    func(list(sys.argv[2:]))


if __name__ == "__main__":
    cli()
