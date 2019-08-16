# Import models from epistasis
from epistasis.models import (
      EpistasisPipeline,
      EpistasisLogisticRegression,
      EpistasisSpline,
      EpistasisLinearRegression
)


# Get gpseer's configuration system.
c = get_config()

# Define an epistasis model for this config.
c.GPSeer.model_definition = EpistasisPipeline([
      EpistasisSpline(k=3),
      EpistasisLinearRegression(order=3)
])