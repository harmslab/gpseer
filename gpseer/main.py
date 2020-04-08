import sys
import logging
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import maximum_likelihood
from . import goodness_of_fit


DESCRIPTION = """
GPSeer: software to infer missing data in sparsely sampled genotype-phenotype maps.
"""

ARGUMENTS = {
    # Positional Arguments
    "input_file": dict(
        type=str,
        help="""
        A CSV file containing the observed/measured genotype-phenotype map data.
        """,
    )
}

OPTIONAL_ARGUMENTS = {
    # Optional Arguments:
    # All optional arguments should be prefixed
    # with a `--` to make the optional.
    "--wildtype": dict(
        type=str,
        help="""
        The reference/wildtype genotype. If this is not specified, GPSeer
        will use the first sequence in the input_file.
        """,
        default=None
    ),
    "--threshold": dict(
        type=float,
        help="""
        The minimum quantitative phenotype value that is detectable or
        measurable. GPSeer will treat any phenotypes below this threshold as a
        separate class of data-points.
        """,
        default=None
    ),
    "--spline_order": dict(
        type=int,
        help="""
        The order of the spline used to estimate the nonlinearity in the
        genotype-phenotype map.""",
        default=None,
    ),
    "--spline_smoothness": dict(
        type=int,
        help="""
        The 'smoothness' parameter used to smooth the spline when
        estimating the nonlinearity in a genotype-phenotype map
        """,
        default=10,
    ),
    "--epistasis_order": dict(
        type=int,
        help="""
        The order of epistasis to include in the linear, high-order epistasis model
        """,
        default=1
    ),
}


def setup_logger(stream_out=sys.stdout):
    """Build a basic console logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(stream_out)
    logger.addHandler(handler)
    formatter = logging.Formatter(
        "[GPSeer] %(message)s"
    )
    handler.setFormatter(formatter)
    return logger


def build_command_line():
    """Construct the GPSeer entrypoint.

    GPSeer has the following subcommands:
    * estimate-ml
    * goodness-of-fit
    """
    # The main entrypoint is GPSeer, which
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    submodules = {maximum_likelihood, goodness_of_fit}

    # Load subparsers from each submodule listed above.
    subparsers = parser.add_subparsers()
    for mod in submodules:
        # Constructs
        subparser = subparsers.add_parser(
            mod.SUBCOMMAND,
            description=mod.DESCRIPTION,
            help=mod.HELP
        )
        # Construct the list of arguments from gpseer and subcommand.
        all_args = {}
        all_args.update(ARGUMENTS)
        all_args.update(mod.ARGUMENTS)
        all_args.update(OPTIONAL_ARGUMENTS)
        all_args.update(mod.OPTIONAL_ARGUMENTS)

        for key, val in all_args.items():
            subparser.add_argument(key, **val)

        # Add the main function for this submodule
        # as the entrpoint to its subcommand.
        subparser.set_defaults(main=mod.main)
    return parser


def run(parser):
    """Convert a gpseer
    """
    # Initialize a logger for the running program.
    logger = setup_logger()
    # Parse incomding command.
    args = parser.parse_args()
    kwargs = vars(args)
    # Call the subcommmand.
    main = kwargs.pop('main')
    main(logger, **kwargs)


def entrypoint():
    # Build main parser + subparsers
    parser = build_command_line()
    run(parser)


if __name__ == "__main__":
    entrypoint()