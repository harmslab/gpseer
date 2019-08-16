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
c.epistasis_model = EpistasisPipeline([
      #EpistasisLogisticRegression(threshold=5),
      EpistasisSpline(k=3),
      EpistasisLinearRegression(order=3)
])