import argparse


arguments = {
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
        genotype-phenotype map."""
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


def run():
    parser = argparse.ArgumentParser(
        description="""
        GPSeer: software to infer missing data in sparsely sampled genotype-phenotype maps.
        """
    )
    for key, val in arguments.items():
        parser.add_argument(key, **val)
    parser.parse_args()


if __name__ == "__main__":
    run()