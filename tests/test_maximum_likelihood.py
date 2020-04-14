import pytest
import os

from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)
from gpseer.maximum_likelihood import (
    main
)

# We'll use the `pytest.mark.datafiles` decorator to
# autoload temporary data files form the example
# directory at the top level of this repo.
@pytest.mark.datafiles(
    'examples/example-train.csv'
)
def test_main(
    logger,
    console_log,
    tmp_path,
    datafiles
):
    infile = datafiles / 'example-train.csv'
    outroot = tmp_path / 'tmp-out'

    main(
        logger,
        str(infile),
        str(outroot),
    )

    expected_outputs = ["_predictions.csv",
                        "_fit-information.csv",
                        "_convergence.csv",
                        "_correlation-plot.pdf",
                        "_phenotype-histograms.pdf"]
    for e in expected_outputs:
        outfile = "{}{}".format(outroot,e)
        assert os.path.isfile(outfile)

    # Assert logging is working
    main_out = "{}_predictions.csv".format(outroot,)
    console = console_log.getvalue()
    assert f"Reading data from {infile}" in console
    assert "Constructing a model..." in console
    assert "Fitting data..." in console
    assert "Predicting missing data..." in console
    assert f"Writing phenotypes to {main_out}" in console
    assert "GPSeer finished!" in console
