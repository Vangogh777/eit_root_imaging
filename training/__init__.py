from .logger import TrainLogger
from .loss import (
    MeasurementConsistencyLoss,
    TVRegularizationLoss,
    FrequencyCrossConsistencyLoss,
    BLCCorrectionLoss,
    SmoothnessLoss,
    AdaptiveLossWeighter,
)
from .optimizer import build_optimizer, build_scheduler
from .residual_loss import (
    ResidualMeasurementConsistencyLoss,
    ResidualSparsityLoss,
    ResidualSmoothnessLoss,
    RelativeMSELoss,
)


def __getattr__(name):
    if name == "UnsupervisedTrainer":
        from .unsupervised_loop import UnsupervisedTrainer
        return UnsupervisedTrainer
    raise AttributeError(name)
