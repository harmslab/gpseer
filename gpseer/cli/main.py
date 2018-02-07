


import argparse

from . import cont
from . import predict

def main():

    # Handle command line argument.
    parser = argparse.ArgumentParser(
        description="A command line tool to infer missing data in a sparsely "
        "sampled genotype-phenotype map.")

    # SUBPARSERS
    subparsers = parser.add_subparsers()

    # ---------- predict subparser ------------
    predict_parser = subparsers.add_parser('predict')

    predict_parser.add_argument('-i', '--input',
                        help="File containing genotype-phenotype map data.")
    predict_parser.add_argument('-o', '--output', default='results.csv',
                        type=str, help="Output filename.")
    predict_parser.add_argument('-m', '--model', default='EpistasisLinearRegression',
                        type=str,
                        help="Name of epistasis model to use. See list of "
                        "models in epistasis package.")
    predict_parser.add_argument('-n', '--nsamples', default=100, type=int,
                        help="Number of steps taken by MCMC walkers. Total "
                        "number of steps will be (number of samples * number "
                        "of reference states)")
    predict_parser.add_argument('--range', nargs=2, type=float,
                        help="Bounds of the posterior distribution.")
    predict_parser.add_argument('--order', default=1, type=int,
                        help="Order of epistasis to include in model.")
    predict_parser.add_argument('--binsize', default=None, type=float,
                        help="Binsize for posterior distributions. Think of "
                        "this as the resolution of the predictions. Default "
                        "is (0.01 * posterior_window)")
    predict_parser.add_argument('--perspective', default='single', type=str,
                        help="Sample predictions from a single reference "
                        "state (wildtype genotype) or multiple references "
                        "states.")
    predict_parser.add_argument('--distributed', default=True, type=bool,
                        help="Distribute the computations in parallel using "
                        "Dask distributed.")
    predict_parser.add_argument('--format', default='json', type=str,
                        help="File format of the genotype-phenotype map data.")
    predict_parser.add_argument('-d', '--db_dir', default='database',
                        type=str, help="Database directory.")
    predict_parser.add_argument('--model_type', default='global', type=str,
                        help="Type of model matrix to use.")
    predict_parser.add_argument('--threshold', default=0, type=float,
                        help="threshold below which classifier will set to 0")
    predict_parser.add_argument('-c', '--preclassify', default=False, type=bool,
                        help="Classify the phenotypes before fitting?")
    predict_parser.add_argument('--parameters', nargs='+',
                        type=str, help="Model parameters")
    predict_parser.set_defaults(func=predict.main)

    # ---------- continue subparser ------------
    cont_parser = subparsers.add_parser('continue')

    cont_parser.add_argument('-d', '--db_dir',
                        help="Directory containing previously sampled GPSeer.")
    cont_parser.add_argument('-o', '--output', default='results.csv',
                        type=str, help="Output filename.")
    cont_parser.add_argument('-n', '--nsamples', default=100, type=int,
                        help="Number of steps taken by MCMC walkers. Total "
                        "number of steps will be (number of samples * number "
                        "of reference states)")
    cont_parser.add_argument('--distributed', default=True, type=bool,
                        help="Distribute the computations in parallel "
                        "using Dask distributed.")
    cont_parser.set_defaults(func=cont.main)

    # Parse arguments
    args = parser.parse_args()

    # Run programs.
    args.func(arg)
