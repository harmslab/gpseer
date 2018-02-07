# imports
import numpy as np

from gpmap import GenotypePhenotypeMap
import epistasis.models

from gpseer import load


def main(args):
    # Handle distribute
    if args.distributed:
        # Import Dask distributed
        from dask.distributed import Client

        # Start a distributed client
        client = Client()
    else:
        # Set client to None.
        client = None

    # Load seer from disk.
    seer = load(args.db_dir, client=client)

    # Sample pipeline, starting from previous state.
    seer.sample_pipeline(args.nsamples)

    # Write to file
    r = seer.results
    r.to_csv(args.output)


if __name__ == '__main__':

    main()
