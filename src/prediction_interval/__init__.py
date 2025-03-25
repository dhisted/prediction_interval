try:
    from prediction_interval._version import version as __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

# Import your machine learning classes
from .models import XGBoostQuantileRegressor, XGBoostCQR, PredictionIntervalResults, XGBoostBootstrap

# Define what is available for import
__all__ = ["PredictionIntervalResults", "XGBoostQuantileRegressor", "XGBoostCQR", "XGBoostBootstrap"]

"""
from .models import ModelA, ModelB: This imports ModelA and ModelB from the models.py file within the same package directory. Adjust the file name if your classes are in a different file.
__all__ = ["ModelA", "ModelB"]: This defines the public API of your package. By listing ModelA and ModelB in __all__, you make them directly accessible when someone imports your package.
With this setup, users can import your classes directly from the package, like so:

from prediction_interval import ModelA, ModelB

model_a = ModelA()
model_b = ModelB()
"""