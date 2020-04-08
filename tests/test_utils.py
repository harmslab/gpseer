from gpseer.utils import construct_model
from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)

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
