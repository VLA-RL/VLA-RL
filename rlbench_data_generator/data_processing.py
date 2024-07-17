from rlbench.utils import get_stored_demos
from rlbench.observation_config import ObservationConfig, CameraConfig
from utils import _keypoint_discovery
from matplotlib import pyplot as plt
import os, pickle
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from data_generator_config import DataGeneratorConfig

cfg = DataGeneratorConfig

amount = cfg.amount
image_resize = cfg.image_resize
dataset_root = cfg.raw_data_dir
task_names = cfg.task_names
variation = cfg.variation
camera = CameraConfig(image_size=(224,224), depth=False, point_cloud=False, mask=False)
obs_config = ObservationConfig(front_camera=camera)

image_paths = []
language_instructions = []
cur_poses = []
actions = []

for task_name in task_names:
    for variation_number in range(variation): 
        data_path = os.path.join(dataset_root, task_name, f'variation{variation_number}')
        varation_descs_pkl_file = os.path.join(data_path, 'variation_descriptions.pkl')
        with open(varation_descs_pkl_file, 'rb') as f:
            descs = pickle.load(f)
        demos = get_stored_demos(dataset_root = dataset_root, task_name = task_name, obs_config = obs_config, variation_number = variation_number, amount = amount, image_paths = image_resize, random_selection = False)

        for episode_num, demo in enumerate(demos):
            episode_keypoints = _keypoint_discovery(demo)
            for keypoint in episode_keypoints:
                cur_pt = keypoint[0]
                cur_obs = demo[cur_pt]
                key_pt = keypoint[1]
                key_obs = demo[key_pt]
                image_path = os.path.join(data_path, f'episodes/episode{episode_num}', f'front_rgb/{cur_pt}.png')
                image_paths.append(image_path)
                language_instructions.append(descs[0])

                cur_pose = cur_obs.gripper_pose
                cur_rot_euler = Rotation.from_quat(cur_pose[3:]).as_euler('xyz', degrees=False)
                cur_gripper_open = cur_obs.gripper_open
                cur_pose = np.concatenate((cur_pose[:3], cur_rot_euler, [cur_gripper_open]), axis=0)
                cur_poses.append(cur_pose)

                key_pose = key_obs.gripper_pose
                key_rot_euler = Rotation.from_quat(key_pose[3:]).as_euler('xyz', degrees=False)
                # print(rot_euler)
                # quat = normalize_quaternion(pose[3:])
                # if quat[-1] < 0:#TODO: check if this is correct
                #     quat = -quat
                # disc_rot = quaternion_to_discrete_euler(quat, 3.6)
                key_gripper_open = key_obs.gripper_open
                
                action = np.concatenate((key_pose[:3], key_rot_euler, [key_gripper_open]), axis=0)
                actions.append(action)

data = {'image_paths': image_paths, 'language_instructions': language_instructions, 'cur_poses': cur_poses, 'actions': actions}
torch.save(data, os.path.join(dataset_root, f'{task_name}/data.pt'))