from typing import Optional, Sequence

import pytest

from d3rlpy.algos.qlearning.awac import AWACConfig
from d3rlpy.models import (
    MeanQFunctionFactory,
    QFunctionFactory,
    QRQFunctionFactory,
)

from ...testing_utils import create_scaler_tuple
from .algo_test import algo_tester


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize(
    "q_func_factory", [MeanQFunctionFactory(), QRQFunctionFactory()]
)
@pytest.mark.parametrize("scalers", [None, "min_max"])
def test_awac(
    observation_shape: Sequence[int],
    q_func_factory: QFunctionFactory,
    scalers: Optional[str],
) -> None:
    observation_scaler, action_scaler, reward_scaler = create_scaler_tuple(
        scalers
    )
    config = AWACConfig(
        q_func_factory=q_func_factory,
        observation_scaler=observation_scaler,
        action_scaler=action_scaler,
        reward_scaler=reward_scaler,
    )
    awac = config.create()
    algo_tester(awac, observation_shape)  # type: ignore
