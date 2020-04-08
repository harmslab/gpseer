import pytest
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
    outfile = tmp_path / 'output.csv'

    main(
        logger,
        str(infile),
        str(outfile),
    )

    assert outfile.is_file()

    # Assert logging is working
    console = console_log.getvalue()
    assert f"Reading data from {infile}" in console
    assert "Constructing a model..." in console
    assert "Fitting data..." in console
    assert "Predicting missing data..." in console
    assert f"Writing phenotypes to {outfile}" in console
    assert "GPSeer finished!" in console