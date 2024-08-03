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
from vla.utils import AngleLoss

class RLbenchPoseTokenizer:
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase, dataset_statistics: tuple
    ) -> None:
        self.tokenizer = tokenizer
        lower_bound, upper_bound = dataset_statistics
        eps = 1e-8
        # Transmit X 0-0.5, Y -0.5-0.5, Z 0.5-1.5
        self.x_min = lower_bound[0]
        self.x_max = upper_bound[0] - eps
        self.x_num_bins = 100
        self.x_bins = np.linspace(self.x_min, self.x_max, self.x_num_bins+1)
        self.x_bin_centers = (self.x_bins[:-1] + self.x_bins[1:]) / 2.0
        self.y_min = lower_bound[1]
        self.y_max = upper_bound[1] - eps
        self.y_num_bins = 100
        self.y_bins = np.linspace(self.y_min, self.y_max, self.y_num_bins+1)
        self.y_bin_centers = (self.y_bins[:-1] + self.y_bins[1:]) / 2.0
        self.z_min = lower_bound[2]
        self.z_max = upper_bound[2] - eps
        self.z_num_bins = 100
        self.z_bins = np.linspace(self.z_min, self.z_max, self.z_num_bins+1)
        self.z_bin_centers = (self.z_bins[:-1] + self.z_bins[1:]) / 2.0


        self.rx_min = lower_bound[3]
        self.rx_max = upper_bound[3] - eps
        self.rx_num_bins = 100
        self.rx_bins = np.linspace(self.rx_min, self.rx_max, self.rx_num_bins+1)
        self.rx_bin_centers = (self.rx_bins[:-1] + self.rx_bins[1:]) / 2.0
        self.ry_min = lower_bound[4]
        self.ry_max = upper_bound[4] - eps
        self.ry_num_bins = 100
        self.ry_bins = np.linspace(self.ry_min, self.ry_max, self.ry_num_bins+1)
        self.ry_bin_centers = (self.ry_bins[:-1] + self.ry_bins[1:]) / 2.0
        self.rz_min = lower_bound[5]
        self.rz_max = upper_bound[5] - eps
        self.rz_num_bins = 100
        self.rz_bins = np.linspace(self.rz_min, self.rz_max, self.rz_num_bins+1)
        self.rz_bin_centers = (self.rz_bins[:-1] + self.rz_bins[1:]) / 2.0

        #gripper 0, 1
        self.grip_num_bins = 2
        self.n_bins = self.x_num_bins + self.y_num_bins + self.z_num_bins + self.rx_num_bins + self.ry_num_bins + self.rz_num_bins + self.grip_num_bins
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - self.n_bins)#-352 #+1?

        self.angle_loss = AngleLoss()

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        eps = 1e-8
        x = np.clip(action[0], a_min=self.x_min, a_max=self.x_max-eps)
        y = np.clip(action[1], a_min=self.y_min, a_max=self.y_max-eps)
        z = np.clip(action[2], a_min=self.z_min, a_max=self.z_max-eps)
        x_discretized = np.digitize(x, self.x_bins)
        x_discretized = - x_discretized + self.n_bins +1 #(-402 - -303)
        y_discretized = np.digitize(y, self.y_bins)
        y_discretized = - y_discretized + self.n_bins - self.x_num_bins +1 # (-302 - -203)
        z_discretized = np.digitize(z, self.z_bins)
        z_discretized = - z_discretized + self.n_bins - self.x_num_bins - self.y_num_bins +1# (-202 - -103)
        discretized_actions = np.concatenate([[x_discretized, y_discretized, z_discretized]])
        if len(action) != 3:
            rx = action[3] - np.pi if action[3] > 0 else action[3] + np.pi #Important
            rx = np.clip(rx, a_min=self.rx_min, a_max=self.rx_max-eps)
            rx_discretized = np.digitize(rx, self.rx_bins)
            rx_discretized = - rx_discretized + self.n_bins - self.x_num_bins - self.y_num_bins - self.z_num_bins +1
            ry = np.clip(action[4], a_min=self.ry_min, a_max=self.ry_max-eps)
            ry_discretized = np.digitize(ry, self.ry_bins)
            ry_discretized = - ry_discretized + self.n_bins - self.x_num_bins - self.y_num_bins - self.z_num_bins - self.rx_num_bins +1
            rz = np.clip(action[5], a_min=self.rz_min, a_max=self.rz_max-eps)
            rz_discretized = np.digitize(rz, self.rz_bins)
            rz_discretized = - rz_discretized + self.n_bins - self.x_num_bins - self.y_num_bins - self.z_num_bins - self.rx_num_bins - self.ry_num_bins +1
            discretized_actions = np.concatenate([[x_discretized, y_discretized, z_discretized, rx_discretized, ry_discretized, rz_discretized]])
            if len(action) == 7:
                grip = action[-1]
                grip_discretized = int(2 - grip)
                discretized_actions = np.concatenate([[x_discretized, y_discretized, z_discretized, rx_discretized, ry_discretized, rz_discretized, grip_discretized]])
                
        vocabulary_list = (self.tokenizer.vocab_size - discretized_actions)
            # Handle single element vs. batch
        if len(discretized_actions.shape) == 1:
            return self.tokenizer.decode(list(vocabulary_list))
        else:
            return self.tokenizer.batch_decode(vocabulary_list.tolist())
    
    def decode(self, logits: torch.tensor, soft: bool = False, loss : bool = False) -> np.ndarray: 
        device = logits.device
        x_bins_centers = torch.tensor(self.x_bin_centers, dtype=torch.float32).to(device)
        y_bins_centers = torch.tensor(self.y_bin_centers, dtype=torch.float32).to(device)
        z_bins_centers = torch.tensor(self.z_bin_centers, dtype=torch.float32).to(device)

        rx_bins_centers = torch.tensor(self.rx_bin_centers, dtype=torch.float32).to(device)
        ry_bins_centers = torch.tensor(self.ry_bin_centers, dtype=torch.float32).to(device)
        rz_bins_centers = torch.tensor(self.rz_bin_centers, dtype=torch.float32).to(device)
        grip_bins_centers = torch.tensor([0,1], dtype=torch.float32).to(device)

        
        x_score = logits[:, 0:1, :100]
        y_score = logits[:, 1:2, 100:200]
        z_score = logits[:, 2:3, 200:300]

        x_pred = F.softmax(x_score, dim = -1) @ x_bins_centers if soft else x_bins_centers[x_score.argmax(dim = -1)]
        y_pred = F.softmax(y_score, dim = -1) @ y_bins_centers if soft else y_bins_centers[y_score.argmax(dim = -1)]
        z_pred = F.softmax(z_score, dim = -1) @ z_bins_centers if soft else z_bins_centers[z_score.argmax(dim = -1)]
        rx_pred = torch.tensor([]).to(device)
        ry_pred = torch.tensor([]).to(device)
        rz_pred = torch.tensor([]).to(device)
        gripper_pred = torch.tensor([]).to(device)
        if logits.shape[1] != 3:
            rx_score = logits[:, 3:4, 300:400]
            ry_score = logits[:, 4:5, 400:500]
            rz_score = logits[:, 5:6, 500:600]
            rx_pred = F.softmax(rx_score, dim = -1) @ rx_bins_centers if soft else rx_bins_centers[rx_score.argmax(dim = -1)]
            ry_pred = F.softmax(ry_score, dim = -1) @ ry_bins_centers if soft else ry_bins_centers[ry_score.argmax(dim = -1)]
            rz_pred = F.softmax(rz_score, dim = -1) @ rz_bins_centers if soft else rz_bins_centers[rz_score.argmax(dim = -1)]
            rx_pred = torch.where(rx_pred > 0, rx_pred - torch.pi, rx_pred + torch.pi)
            if logits.shape[1] != 6:
                gripper_score = logits[:, 6:7, 600:]
                gripper_pred = F.softmax(gripper_score, dim = -1) @ grip_bins_centers if soft else grip_bins_centers[gripper_score.argmax(dim = -1)]
        pred_action = torch.cat([x_pred, y_pred, z_pred, rx_pred, ry_pred, rz_pred, gripper_pred], dim = 1).to(device)
        return pred_action
    
    def get_action(self, logits: torch.tensor) -> np.ndarray: 
        grip_bin_centers = np.array([0,1])
        x_score = F.softmax(logits[:, 0:1, :100], dim = -1).squeeze(0).squeeze(0)
        y_score = F.softmax(logits[:, 1:2, 100:200], dim = -1).squeeze(0).squeeze(0)
        z_score = F.softmax(logits[:, 2:3, 200:300], dim = -1).squeeze(0).squeeze(0)

        rx_score = F.softmax(logits[:, 3:4, 300:400], dim = -1).squeeze(0).squeeze(0)
        ry_score = F.softmax(logits[:, 4:5, 400:500], dim = -1).squeeze(0).squeeze(0)
        rz_score = F.softmax(logits[:, 5:6, 500:600], dim = -1).squeeze(0).squeeze(0)
        gripper_score = F.softmax(logits[:, 6:7, 600:], dim = -1).squeeze(0).squeeze(0)

        #sample
        x_pred = self.x_bin_centers[x_score.multinomial(1).item()]
        y_pred = self.y_bin_centers[y_score.multinomial(1).item()]
        z_pred = self.z_bin_centers[z_score.multinomial(1).item()]

        rx_pred = self.rx_bin_centers[rx_score.multinomial(1).item()]
        rx_pred = rx_pred - np.pi if rx_pred > 0 else rx_pred + np.pi
        ry_pred = self.ry_bin_centers[ry_score.multinomial(1).item()]
        rz_pred = self.rz_bin_centers[rz_score.multinomial(1).item()]
        gripper_pred = grip_bin_centers[gripper_score.multinomial(1).item()]

        pred_action = np.array([x_pred, y_pred, z_pred, rx_pred, ry_pred, rz_pred, gripper_pred])
        return pred_action

    def get_mask(self, gt: torch.tensor):
        gripper_start = (gt == 32001).to(torch.int).argmax(dim=1) 
        gripper_end = (gt == 32002).to(torch.int).argmax(dim=1) 
        object_start = (gt == 32008).to(torch.int).argmax(dim=1)
        object_end = (gt == 32009).to(torch.int).argmax(dim=1)
        target_start = (gt == 32010).to(torch.int).argmax(dim=1)
        target_end = (gt == 32011).to(torch.int).argmax(dim=1)
        action_start = (gt == 32014).to(torch.int).argmax(dim=1)
        action_end = (gt == 32015).to(torch.int).argmax(dim=1)

        gripper_mask = torch.zeros_like(gt)
        object_mask = torch.zeros_like(gt)
        target_mask = torch.zeros_like(gt)
        action_mask = torch.zeros_like(gt)

        for i in range(gt.size(0)):
            gripper_mask[i, gripper_start[i]+1:gripper_end[i]] = 1
            object_mask[i, object_start[i]+1:object_end[i]] = 1
            target_mask[i, target_start[i]+1:target_end[i]] = 1
            action_mask[i, action_start[i]+1:action_end[i]] = 1

        item_mask = (gt >= 32003) & (gt <= 32007)
        stage_mask = (gt >= 32012) & (gt <= 32013)

        return gripper_mask.to(torch.bool), item_mask.to(torch.bool), object_mask.to(torch.bool), target_mask.to(torch.bool), stage_mask.to(torch.bool), action_mask.to(torch.bool)

    def get_loss(self, output: torch.tensor, batch: dict, output_start_idx):
        
        nll_loss = output.loss
        output_logits = output.logits[:, output_start_idx:-1]
        device_id = output_logits.device
        batch_size = output_logits.size(0)
        output_gt = batch["labels"][:, 1:].to(device_id)
        gripper_mask, item_mask, object_mask, target_mask, stage_mask, action_mask = self.get_mask(output_gt)

        # gripper_logits = output_logits[gripper_mask][:,self.action_token_begin_idx:self.tokenizer.vocab_size].view(batch_size,-1,self.n_bins)
        item_logits = output_logits[item_mask][:,32003:32008]
        object_logits = output_logits[object_mask][:,self.action_token_begin_idx:self.tokenizer.vocab_size].view(batch_size,-1,self.n_bins)
        target_logits = output_logits[target_mask][:,self.action_token_begin_idx:self.tokenizer.vocab_size].view(batch_size,-1,self.n_bins)
        stage_logits = output_logits[stage_mask][:,32012:32014]
        action_logits = output_logits[action_mask][:,self.action_token_begin_idx:self.tokenizer.vocab_size].view(batch_size,-1,self.n_bins)

        gt_gripper = batch['grippers'].to(device_id)
        gt_item = batch['items'].to(device_id)
        gt_object = batch['objects'].to(device_id)
        gt_target = batch['targets'].to(device_id)
        gt_stage = batch['stages'].to(device_id)
        gt_action = batch['actions'].to(device_id)
        
        #Gripper Loss
        # pred_gripper = self.decode(gripper_logits, soft = True)
        # assert pred_gripper.shape == gt_gripper.shape, f"Gripper shape {pred_gripper.shape} != {gt_gripper.shape}"
        # gripper_position_loss = F.mse_loss(pred_gripper[:,:3], gt_gripper[:,:3].to(torch.float32))
        # gripper_orientation_loss = self.angle_loss(pred_gripper[:,3:6], gt_gripper[:,3:6].to(torch.float32))
        # gripper_open_gt = gt_gripper[:,6].to(torch.int64)
        # gripper_open_loss = F.cross_entropy(gripper_logits[:,6,-2:], gripper_open_gt)

        #Item Loss
        gt_item = gt_item.clone().detach()
        item_loss = F.cross_entropy(item_logits, gt_item)

        #Object Loss
        pred_object = self.decode(object_logits, soft = True)
        assert pred_object.shape == gt_object.shape, f"Object shape {pred_object.shape} != {gt_object.shape}"
        object_position_loss = F.mse_loss(pred_object[:,:3], gt_object[:,:3].to(torch.float32))

        #Target Loss
        pred_target = self.decode(target_logits, soft = True)
        assert pred_target.shape == gt_target.shape, f"Target shape {pred_target.shape} != {gt_target.shape}"
        target_position_loss = F.mse_loss(pred_target[:,:3], gt_target[:,:3].to(torch.float32))

        #Stage Loss
        gt_stage = gt_stage.clone().detach()
        stage_loss = F.cross_entropy(stage_logits, gt_stage)

        #Action Loss
        pred_action = self.decode(action_logits, soft = True)
        assert pred_action.shape == gt_action.shape, f"Action shape {pred_action.shape} != {gt_action.shape}"
        action_position_loss = F.mse_loss(pred_action[:,:3], gt_action[:,:3].to(torch.float32))
        action_orientation_loss = self.angle_loss(pred_action[:,3:6], gt_action[:,3:6].to(torch.float32))
        action_open_gt = gt_action[:,6].to(torch.int64)
        action_open_loss = F.cross_entropy(action_logits[:,6,-2:], action_open_gt)

        loss_dict = {
            'nll_loss':nll_loss,
            # 'gripper_position_loss': gripper_position_loss,
            # 'gripper_orientation_loss': gripper_orientation_loss,
            # 'gripper_open_loss': gripper_open_loss,
            'item_loss': item_loss,
            'object_position_loss': object_position_loss,
            'target_position_loss': target_position_loss,
            'stage_loss': stage_loss,
            'action_position_loss': action_position_loss,
            'action_orientation_loss': action_orientation_loss,
            'action_open_loss': action_open_loss
        }
        return loss_dict

    # def get_loss(self, action_logits: torch.tensor, gt_action: torch.tensor, gripper_logits: torch.tensor, gt_gripper: torch.tensor,
    #              object_logits: torch.tensor, gt_object: torch.tensor, target_logits: torch.tensor, gt_target: torch.tensor, soft: bool = False):
    #     pred_object = self.decode(object_logits, soft = True)
    #     assert pred_object.shape == gt_object.shape, f"Object shape {pred_object.shape} != {gt_object.shape}"
    #     object_position_loss = F.mse_loss(pred_object[:,:3], gt_object[:,:3].to(torch.float32))

    #     pred_target = self.decode(target_logits, soft = True)
    #     assert pred_target.shape == gt_target.shape, f"Target shape {pred_target.shape} != {gt_target.shape}"
    #     target_position_loss = F.mse_loss(pred_target[:,:3], gt_target[:,:3].to(torch.float32))
        
    #     pred_gripper = self.decode(gripper_logits, soft = True)
    #     assert pred_gripper.shape == gt_gripper.shape, f"Gripper shape {pred_gripper.shape} != {gt_gripper.shape}"
    #     gripper_position_loss = F.mse_loss(pred_gripper[:,:3], gt_gripper[:,:3].to(torch.float32))
    #     gripper_orientation_loss = self.angle_loss(pred_gripper[:,3:6], gt_gripper[:,3:6].to(torch.float32))
    #     gripper_open_gt = torch.zeros_like(gripper_logits[:,6,-2:]).scatter_(1, gt_gripper[:,6].unsqueeze(1).to(torch.int64), 1)
    #     gripper_open_loss = F.cross_entropy(gripper_logits[:,6,-2:], gripper_open_gt.to(torch.float32))

    #     pred_action = self.decode(action_logits, soft = True)
    #     assert pred_action.shape == gt_action.shape, f"Action shape {pred_action.shape} != {gt_action.shape}"
    #     action_position_loss = F.mse_loss(pred_action[:,:3], gt_action[:,:3].to(torch.float32))
    #     action_orientation_loss = self.angle_loss(pred_action[:,3:6], gt_action[:,3:6].to(torch.float32))
    #     action_open_gt = torch.zeros_like(action_logits[:,6,-2:]).scatter_(1, gt_action[:,6].unsqueeze(1).to(torch.int64), 1)
    #     action_open_loss = F.cross_entropy(action_logits[:,6,-2:], action_open_gt.to(torch.float32))

    #     loss_dict = {
    #         'object_position_loss': object_position_loss,
    #         'target_position_loss': target_position_loss,
    #         'gripper_position_loss': gripper_position_loss,
    #         'gripper_orientation_loss': gripper_orientation_loss,
    #         'gripper_open_loss': gripper_open_loss,
    #         'action_position_loss': action_position_loss,
    #         'action_orientation_loss': action_orientation_loss,
    #         'action_open_loss': action_open_loss
    #     }

    #     return loss_dict


class RLbenchActionTokenizer:
    def __init__(
            self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.tokenizer = tokenizer
        eps = 1e-8
        # Transmit X 0-0.5, Y -0.5-0.5, Z 0.5-1.5
        self.x_min = -0.5
        self.x_max = 0.5 - eps
        self.x_num_bins = 100
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
        
        #gripper 0, 1
        self.grip_num_bins = 2
        self.n_bins = self.x_num_bins + self.y_num_bins + self.z_num_bins + self.rot_num_bins + self.grip_num_bins
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - self.n_bins)

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
        x_discretized = - x_discretized + self.n_bins +1 #(-352 - -303)
        y_discretized = np.digitize(y, self.y_bins)
        y_discretized = - y_discretized + self.n_bins - self.x_num_bins +1 # (-302 - -203)
        z_discretized = np.digitize(z, self.z_bins)
        z_discretized = - z_discretized + self.n_bins - self.x_num_bins - self.y_num_bins +1# (-202 - -103)
        rot_discretized = np.digitize(rot, self.rot_bins)
        rot_discretized = - rot_discretized + self.grip_num_bins + self.rot_num_bins+1# (-102 - -3)
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
            x_pred = torch.tensor(self.x_bin_centers, dtype=torch.float32).to(device)[torch.argmax(x_score)]
            y_pred = torch.tensor(self.y_bin_centers, dtype=torch.float32).to(device)[torch.argmax(y_score)]
            z_pred = torch.tensor(self.z_bin_centers, dtype=torch.float32).to(device)[torch.argmax(z_score)]
            rot_pred = torch.tensor(self.rot_bin_centers, dtype=torch.float32).to(device)[torch.argmax(rot_score,dim = 1)]
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
