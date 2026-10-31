from typing import Any

import torch


class FiniteHistoryOptimizer(torch.optim.Optimizer):

    """
    Checks that the saved and current optimizers have the same number of parameter groups?
    """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("checkpoint parameter group count changed")
        for saved, current in zip(saved_groups, self.param_groups):
        # checks the values are the same in the momentum scheduler
            if saved.get("scheduler_config_sha256") != current.get("scheduler_config_sha256"):
                raise ValueError("checkpoint scheduler configuration differs from the constructor")
        # FP32 parameters let PyTorch own state dtype and device placement.
        super().load_state_dict(state_dict)
