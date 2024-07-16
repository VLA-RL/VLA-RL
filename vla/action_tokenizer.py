"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
Inspired by https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

class RLbenchActionTokenizer:
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.tokenizer = tokenizer
        eps = 1e-8
        # Transmit X 0-0.5, Y -0.5-0.5, Z 0.5-1.5
        self.x_min = 0
        self.x_max = 0.5 - eps
        self.x_num_bins = 50
        self.x_bins = np.linspace(self.x_min, self.x_max, self.x_num_bins+1)
        self.x_bin_centers = (self.x_bins[:-1] + self.x_bins[1:]) / 2.0
        self.y_min = -0.5
        self.y_max = 0.5 - eps
        self.y_num_bins = 100
        self.y_bins = np.linspace(self.y_min, self.y_max, self.y_num_bins+1)
        self.y_bin_centers = (self.y_bins[:-1] + self.y_bins[1:]) / 2.0
        self.z_min = 0.5
        self.z_max = 1.5 - eps
        self.z_num_bins = 100
        self.z_bins = np.linspace(self.z_min, self.z_max, self.z_num_bins+1)
        self.z_bin_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2.0
        #Rotation -pi theta
        self.rot_min = -np.pi
        self.rot_max = np.pi - eps
        self.rot_num_bins = 100
        self.rot_bins = np.linspace(self.rot_min, self.rot_max, self.rot_num_bins+1)
        self.rot_bin_centers = (self.rot_bins[:-1] + self.rot_bins[1:]) / 2.0
        self.conv = lambda x: np.sin(x) + np.cos(x)
        self.inconv = lambda x: np.arcsin(x / np.sqrt(2)) - np.pi/4

        #gripper 0, 1
        self.grip_num_bins = 2
        self.n_bins = self.x_num_bins + self.y_num_bins + self.z_num_bins + self.rot_num_bins + self.grip_num_bins
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - self.n_bins)#-352 #+1?

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        eps = 1e-8
        x = np.clip(action[0], a_min=self.x_min, a_max=self.x_max-eps)
        y = np.clip(action[1], a_min=self.y_min, a_max=self.y_max-eps)
        z = np.clip(action[2], a_min=self.z_min, a_max=self.z_max-eps)
        # quat = action[3:-1]
        # rot = R.from_quat(quat).as_euler('zyx')
        rot = np.clip(action[3:-1], a_min=self.rot_min, a_max=self.rot_max-eps)
        grip = action[-1]

        x_discretized = np.digitize(x, self.x_bins)
        x_discretized = - x_discretized + self.n_bins #(-352 - -303)
        y_discretized = np.digitize(y, self.y_bins)
        y_discretized = - y_discretized + self.n_bins - self.x_num_bins # (-302 - -203)
        z_discretized = np.digitize(z, self.z_bins)
        z_discretized = - z_discretized + self.n_bins - self.x_num_bins - self.y_num_bins # (-202 - -103)
        rot_discretized = np.digitize(rot, self.rot_bins)
        rot_discretized = - rot_discretized + self.grip_num_bins + self.rot_num_bins# (-102 - -3)
        grip_discretized = int(2 - grip) # (-2 - -1)

        discretized_actions = np.concatenate([[x_discretized, y_discretized, z_discretized], rot_discretized, [grip_discretized]])
        vocabulary_list = (self.tokenizer.vocab_size - discretized_actions)
        # return self.tokenizer.batch_decode(vocabulary_list)
            # Handle single element vs. batch
        if len(discretized_actions.shape) == 1:
            return self.tokenizer.decode(list(vocabulary_list))
        else:
            return self.tokenizer.batch_decode(vocabulary_list.tolist())

    
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids - 1
        x_indice = discretized_actions[0] - self.n_bins + self.x_num_bins
        y_indice = discretized_actions[1] - self.n_bins + self.x_num_bins + self.y_num_bins
        z_indice = discretized_actions[2] - self.n_bins + self.x_num_bins + self.y_num_bins + self.z_num_bins
        rot_indice = discretized_actions[3:-1] - self.grip_num_bins
        grip_indice = discretized_actions[-1]

        x = self.x_bin_centers[x_indice]
        y = self.y_bin_centers[y_indice]
        z = self.z_bin_centers[z_indice]
        rot = self.rot_bin_centers[rot_indice]
        # quat = R.from_euler('zyx', rot).as_quat()
        grip = grip_indice
        continous_actions = np.concatenate([[x, y, z], rot, [grip]])

        return continous_actions
    
    def decode_token_score_to_actions(self, action_score: torch.tensor, soft: bool = True) -> np.ndarray:
        """
            small is large
        """
        device = action_score.device
        x_score = action_score[0,:50]
        y_score = action_score[1,50:150]
        z_score = action_score[2,150:250]
        rot_score = action_score[3:-1,250:350]
        grip_score = action_score[-1,350:]
        if soft:
            x_pred = F.softmax(x_score) @ torch.tensor(self.x_bin_centers, dtype=torch.float32).to(device)
            y_pred = F.softmax(y_score) @ torch.tensor(self.y_bin_centers, dtype=torch.float32).to(device)
            z_pred = F.softmax(z_score) @ torch.tensor(self.z_bin_centers, dtype=torch.float32).to(device)
            rot_pred = F.softmax(rot_score, dim = 1) @ torch.tensor(self.rot_bin_centers, dtype=torch.float32).to(device)
            grip_pred = F.softmax(grip_score) @ torch.tensor([0,1], dtype=torch.float32).to(device)
        else:
            x_pred = self.x_bin_centers[torch.argmax(x_score)]
            y_pred = self.y_bin_centers[torch.argmax(y_score)]
            z_pred = self.z_bin_centers[torch.argmax(z_score)]
            rot_pred = self.rot_bin_centers[torch.argmax(rot_score)]
            grip_pred = torch.argmax(grip_score)

        pred_action = torch.cat([x_pred.unsqueeze(0), y_pred.unsqueeze(0), z_pred.unsqueeze(0), rot_pred, grip_pred.unsqueeze(0)]).to(device)

        return pred_action

    def output_logit_to_continous_action(self, masked_logits: torch.tensor):
        assert masked_logits.shape[1] == 7
        x_pred = (F.softmax(masked_logits[:, 0,:50], dim = 1)@torch.tensor(self.x_bin_centers,dtype=torch.float32).to('cuda')).unsqueeze(1)
        y_pred = (F.softmax(masked_logits[:, 1, 50:150], dim = 1)@torch.tensor(self.y_bin_centers,dtype=torch.float32).to('cuda')).unsqueeze(1)
        z_pred = (F.softmax(masked_logits[:, 2, 150:250], dim = 1)@torch.tensor(self.z_bin_centers,dtype=torch.float32).to('cuda')).unsqueeze(1)
        rot_pred = F.softmax(masked_logits[:,3:-1,250:350], dim = 2)@torch.tensor(self.rot_bin_centers,dtype=torch.float32).to('cuda')
        grip_pred= (F.softmax(masked_logits[:,-1,350:], dim = 1) @ torch.tensor([0,1],dtype=torch.float32).to('cuda')).unsqueeze(1)
        pred_action = torch.cat([x_pred, y_pred, z_pred, rot_pred, grip_pred], dim = 1)
        return pred_action
    


    @property
    def vocab_size(self) -> int:
        return self.n_bins

class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
