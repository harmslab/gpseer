# imports
import numpy as np

from gpmap import GenotypePhenotypeMap
import epistasis.models

from gpseer import GPSeer

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

    # Get read method for GenotypePhenotype Map
    gpm_constructor = getattr(GenotypePhenotypeMap,
                              'read_{}'.format(args.format))
    # Construct genotype-phenotype map object
    gpm = gpm_constructor(args.input)

    # Handle model parameters
    parameters = {}
    for i in range(0, len(args.parameters), 2):
        key = args.parameters[i]
        val = args.parameters[i + 1]
        parameters[key] = float(val)

    # Create a set of bins
    binsize = args.binsize
    if binsize is None:
        binsize = 0.01 * args.range[1] - args.range[0]
    bins = np.arange(args.range[0], args.range[1],
                     binsize)

    # Construct an epistasis model
    model_constructor = getattr(epistasis.models, args.model)
    model = model_constructor(
        order=args.order, model_type=args.model_type, **parameters)

    if args.preclassify:
        classifier = epistasis.models.EpistasisLogisticRegression(
            order=1,
            threshold=args.threshold,
            model_type=args.model_type)

        # Build a mixed model.
        model = epistasis.models.EpistasisMixedRegression(classifier, model)

    # Initialize a GPSeer engine.
    seer = GPSeer(gpm, model, bins,
                  genotypes='complete',
                  sample_weight=None,
                  client=client,
                  perspective=args.perspective,
                  db_dir=args.db_dir)

    # Sample the posterior probabilities
    seer.sample_pipeline(args.nsamples)

    # Gather results
    r = seer.results

    # Write results to output file.
    r.to_csv(args.output)


if __name__ == '__main__':

    main()
