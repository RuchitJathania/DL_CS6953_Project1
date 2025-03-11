from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Expected `optimizer` to be an instance of `torch.optim.Optimizer`, got {type(optimizer)}")

        self.optimizer = optimizer  # Base optimizer (e.g., SGD, Adam)
        self.k = k  # Number of fast optimizer updates before lookahead update
        self.alpha = alpha  # Step size for slow weights

        # Copy parameter groups from the base optimizer
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state

        # Initialize slow weights and counters
        for group in self.param_groups:
            group["counter"] = 0
            for p in group["params"]:
                param_state = self.state[p]
                param_state["slow_param"] = p.clone().detach()

        # ✅ Add defaults dictionary (required to avoid the `AttributeError`)
        defaults = { "k": k, "alpha": alpha }
        super().__init__(self.param_groups, defaults)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)  # Perform inner optimizer step
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] >= self.k:
                for p in group["params"]:
                    param_state = self.state[p]
                    if "slow_param" in param_state:
                        slow_p = param_state["slow_param"]
                        slow_p += (p.data - slow_p) * self.alpha  # Slow weight update
                        p.data.copy_(slow_p)  # Copy slow weights to fast weights
                group["counter"] = 0  # Reset counter
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Ensure Lookahead also clears gradients properly"""
        self.optimizer.zero_grad(set_to_none)

    def state_dict(self):
        base_state_dict = self.optimizer.state_dict()
        lookahead_state_dict = {
            "fast_state": base_state_dict["state"],
            "param_groups": base_state_dict["param_groups"],
            "slow_state": {k: v.clone() for k, v in self.state.items()},
        }
        return lookahead_state_dict

    def load_state_dict(self, state_dict):
        base_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        self.optimizer.load_state_dict(base_state_dict)
        self.state = state_dict["slow_state"]
#
# class Lookahead(Optimizer):
#     def __init__(self, optimizer, k=5, alpha=0.5):
#         self.optimizer = optimizer
#         self.k = k
#         self.alpha = alpha
#         self.param_groups = self.optimizer.param_groups
#         self.state = defaultdict(dict)
#         self.fast_state = self.optimizer.state
#         for group in self.param_groups:
#             group["counter"] = 0
#
#     def update(self, group):
#         for fast in group["params"]:
#             param_state = self.state[fast]
#             if "slow_param" not in param_state:
#                 param_state["slow_param"] = torch.zeros_like(fast.data)
#                 param_state["slow_param"].copy_(fast.data)
#             slow = param_state["slow_param"]
#             slow += (fast.data - slow) * self.alpha
#             fast.data.copy_(slow)
#
#     def update_lookahead(self):
#         for group in self.param_groups:
#             self.update(group)
#
#     def step(self, closure=None):
#         loss = self.optimizer.step(closure)
#         for group in self.param_groups:
#             if group["counter"] == 0:
#                 self.update(group)
#             group["counter"] += 1
#             if group["counter"] >= self.k:
#                 group["counter"] = 0
#         return loss
#
#     def state_dict(self):
#         fast_state_dict = self.optimizer.state_dict()
#         slow_state = {
#             (id(k) if isinstance(k, torch.Tensor) else k): v
#             for k, v in self.state.items()
#         }
#         fast_state = fast_state_dict["state"]
#         param_groups = fast_state_dict["param_groups"]
#         return {
#             "fast_state": fast_state,
#             "slow_state": slow_state,
#             "param_groups": param_groups,
#         }
#
#     def load_state_dict(self, state_dict):
#         slow_state_dict = {
#             "state": state_dict["slow_state"],
#             "param_groups": state_dict["param_groups"],
#         }
#         fast_state_dict = {
#             "state": state_dict["fast_state"],
#             "param_groups": state_dict["param_groups"],
#         }
#         super(Lookahead, self).load_state_dict(slow_state_dict)
#         self.optimizer.load_state_dict(fast_state_dict)
#         self.fast_state = self.optimizer.state
#
#     def add_param_group(self, param_group):
#         param_group["counter"] = 0
#         self.optimizer.add_param_group(param_group)
