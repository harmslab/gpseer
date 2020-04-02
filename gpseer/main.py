import sys
import logging
import argparse

from .maximum_likelihood import run_estimate_ml
from .goodness_of_fit import run_goodness_of_fit


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
    ),
    "output_file": dict(
        type=str,
        help="""
        A CSV file GPSeer will create with final predictions.
        """
    ),
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
    "--nreplicates": dict(
        type=int,
        help="""
        The number of pseudo-replicates to generate when sampling uncertainty
        in the epistasis model.
        """,
        default=None
    ),
    "--genotype_file": dict(
        type=str,
        help="""
        A CSV file with a list of genotypes to predict given the input_file
        and epistasis model.
        """,
        default=None
    )
}


def setup_logger():
    """Build a basic console logger.
    """
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s | GPSeer] %(message)s"
    )
    handler.setFormatter(formatter)
    return logger


def build_command_line():
    """Build a generic argparse CLI with list of arguments.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers()

    # Build the ml_estimate subparser.
    estimate_ml = subparsers.add_parser(
        "estimate-ml",
        description="""
        estimate-ml: GPSeer's maximum likelihood calculator—
        predicts the maximum-likelihood estimates for missing
        phenotypes in a sparsely sampled genotype-phenotype map.
        """,
        help="""
        Predict the maximum-likelihood estimates for missing
        phenotypes in a sparsely sampled genotype-phenotype map.
        """
    )
    for key, val in ARGUMENTS.items():
        estimate_ml.add_argument(key, **val)

    estimate_ml.set_defaults(func=run_estimate_ml)

    # Build the sampling subparser.
    goodness_of_fit = subparsers.add_parser(
        "goodness-of-fit",
        description="""
        goodness-of-fit: GPSeer's goodness-of-fit calculator—
        estimates how well an epistasis model fits a given
        genotype-phenotype map.
        """,
        help="""
        Calculate the 'goodness-of-fit' from a training set of
        genotypes in a genotype-phenotype map.
        """
    )
    for key, val in ARGUMENTS.items():
        goodness_of_fit.add_argument(key, **val)

    goodness_of_fit.set_defaults(func=run_goodness_of_fit)
    return parser


def entrypoint():
    logger = setup_logger()
    parser = build_command_line()
    args = parser.parse_args()

    input_args = {}
    for a in ARGUMENTS:
        if a[0:2] == '--':
            a = a[2:]
        input_args[a] = getattr(args, a)
    args.func(logger, **input_args)


if __name__ == "__main__":
    entrypoint()