

import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(cur_dir)

import logging
import numpy as np
import gymnasium as gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, ModelConfigDict
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from gymnasium.spaces import Box, Dict, Discrete

from uv_net import UVNet
from f_value_net import FValueNet

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class UVModel(TorchModelV2, nn.Module):
    def __init__(self,
                obs_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                num_outputs: int,
                model_config: ModelConfigDict,
                name: str,
                **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        self.agent_num = model_config["agent_num"]
        self.use_gpu = model_config["use_gpu"]


        self.combine_feature_size = model_config["features_size"]# + model_config["value_size"]
        
        self.combine_fc_1 = SlimFC(self.combine_feature_size, self.combine_feature_size * 4, activation_fn="relu")
        self.combine_fc_2 = SlimFC(self.combine_feature_size * 4, self.combine_feature_size * 8, activation_fn="relu")
        self.combine_fc_3 = SlimFC(self.combine_feature_size * 8, self.combine_feature_size * 16, activation_fn="relu")
        self.combine_fc_4 = SlimFC(self.combine_feature_size * 16, self.combine_feature_size * 8, activation_fn="relu")
        self.combine_fc_5 = SlimFC(self.combine_feature_size * 8, self.combine_feature_size * 4, activation_fn="relu")
        self.combine_fc_6 = SlimFC(self.combine_feature_size * 4, self.combine_feature_size, activation_fn="relu")
        self.value_branch = SlimFC(self.combine_feature_size, 1)
        self.logits = SlimFC(self.combine_feature_size, self.num_outputs)

        self._features = None

    def forward(self, input_dict, state, seq_lens):
        features = input_dict["obs"]["observations"]
        action_mask = input_dict["obs"]["action_mask"]
        all_features_1 = self.combine_fc_1(features)
        all_features_2 = self.combine_fc_2(all_features_1)
        all_features_3 = self.combine_fc_3(all_features_2)
        all_features_4 = self.combine_fc_4(all_features_3)
        all_features_5 = self.combine_fc_5(all_features_4)
        self._features = self.combine_fc_6(all_features_5)
        logits = self.logits(self._features)
        inf_mask = torch.clamp(torch.log(action_mask), min=-10000)
        masked_logits = logits + inf_mask
        return masked_logits, state
    
    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])
    





