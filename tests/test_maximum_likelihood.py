import pytest
from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)
from gpseer.maximum_likelihood import (
    run_estimate_ml,
    construct_model
)

# We'll use the `pytest.mark.datafiles` decorator to
# autoload temporary data files form the example
# directory at the top level of this repo.

@pytest.mark.datafiles(
    'examples/example-train.csv'
)
def test_run_estimate_ml(
    logger,
    console_log,
    tmp_path,
    datafiles
):
    infile = datafiles / 'example-train.csv'
    outfile = tmp_path / 'output.csv'

    run_estimate_ml(
        logger,
        str(infile),
        str(outfile),
        wildtype="EEDPT"
    )

    assert outfile.is_file()

    # Assert logging is working
    console = console_log.getvalue()
    assert "Reading input data..." in console
    assert "Finished reading input data." in console
    assert "Constructing a model..." in console
    assert "Fitting model to data..." in console
    assert "Predicting phenotypes..." in console
    assert "Writing phenotypes to file..." in console
    assert "Done!" in console


def test_construct_model():
    # Test default call of construct_model
    model = construct_model()
    assert isinstance(model, EpistasisPipeline)
    assert len(model) == 1
    assert isinstance(model[0], EpistasisLinearRegression)

    # Test default call of construct_model
    model = construct_model(
        threshold=1
    )
    assert isinstance(model, EpistasisPipeline)
    assert len(model) == 2
    assert isinstance(model[0], EpistasisLogisticRegression)
    assert isinstance(model[1], EpistasisLinearRegression)

    # Test default call of construct_model
    model = construct_model(
        threshold=1,
        spline_order=2,
    )
    assert isinstance(model, EpistasisPipeline)
    assert len(model) == 3
    assert isinstance(model[0], EpistasisLogisticRegression)
    assert isinstance(model[1], EpistasisSpline)
    assert isinstance(model[2], EpistasisLinearRegression)
