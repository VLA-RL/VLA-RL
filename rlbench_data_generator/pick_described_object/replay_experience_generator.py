from PIL import Image
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointVelocity, JointPosition, EndEffectorPoseViaPlanning, EndEffectorPoseViaIK

from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import PutGroceriesInCupboard,PickDescribedObject
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import torch 
import random
import multiprocessing
from multiprocessing import Manager
import os
from pyrep.const import RenderMode


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_data(task, variation_num):
    obs = task._scene.get_observation()
    img = Image.fromarray(obs.front_rgb,'RGB')
    gripper_pose = obs.gripper_pose
    gripper_open = obs.gripper_open
    object_pos = task._task.get_graspable_objects()[variation_num].get_position()
    target_pos = task._task.dropin_box.get_position()
    return img, gripper_pose, gripper_open, object_pos, target_pos

def pose2action(pose, open):
    return np.concatenate([pose[:3], R.from_quat(pose[3:]).as_euler('xyz'), [open]])
    
def run_episode(task, variation_num, episode_num, save_root):
    seed = np.random.randint(0, 100000)
    np.random.seed(seed)
    random.seed(seed)
    
    img_paths = []
    instructions = []
    gripper_poses = []
    items = []
    object_positions = []
    target_positions = []
    stages = []
    actions = []
    rewards = []
    dones = []
    next_img_paths = []
    next_gripper_poses = []

    id = 0
    task.set_variation(variation_num)
    save_dir = save_root + f"variation_{variation_num}/episode_{episode_num}/"
    check_and_make(save_dir)
    desc, _ = task.reset()
    
    #stage 0
    stage = 0
    # task._task.randomize_pose(True)
    img, gripper_pose, gripper_open, object_pos, target_pos = get_data(task, variation_num)

    waypoint_pose = task._task.get_waypoints()[stage].get_waypoint_object().get_pose()
    img_path = save_dir+f"{id}.jpg"
    img.save(img_path)
    img_paths.append(img_path)
    instructions.append(random.sample(desc,1)[0])
    gripper = pose2action(gripper_pose, gripper_open)
    gripper_poses.append(gripper)
    items.append(variation_num)
    object_positions.append(object_pos)
    target_positions.append(target_pos)
    stages.append(stage)
    action = pose2action(waypoint_pose,0)
    actions.append(action)

    task_action = np.concatenate([waypoint_pose, [0]])
    obs, reward, done = task.step(task_action)
    rewards.append(reward)
    dones.append(done)
    id+= 1


    
    #stage 1
    stage = 1
    img, gripper_pose, gripper_open, object_pos, target_pos = get_data(task, variation_num)
    waypoint_pose = task._task.get_waypoints()[stage].get_waypoint_object().get_pose()
    img_path = save_dir+f"{id}.jpg"
    img.save(img_path)
    img_paths.append(img_path)
    next_img_paths.append(img_path)
    instructions.append(random.sample(desc,1)[0])
    gripper = pose2action(gripper_pose, gripper_open)
    gripper_poses.append(gripper)
    next_gripper_poses.append(gripper)
    items.append(variation_num)
    object_positions.append(object_pos)
    target_positions.append(target_pos)
    stages.append(stage)
    action_open = 1
    action = pose2action(waypoint_pose, action_open)
    actions.append(action)
    id += 1 
    
    task_action = np.concatenate([waypoint_pose, [action_open]])
    obs, reward, done = task.step(task_action)
    rewards.append(reward)
    dones.append(done)

    next_img_paths.append(None)
    next_gripper_poses.append(None)
    assert task._scene.task.success()[0] == True, "Task failed"

    return img_paths, instructions, gripper_poses, items, object_positions, target_positions, stages, actions, rewards, dones, next_img_paths, next_gripper_poses

# Define the function to be executed in each process
def process_variation(i, total_episodes,save_root, manager_dict, lock):
    local_train_imgs = []
    local_train_instructions = []
    local_train_gripper = []
    local_train_items = []
    local_train_objects = []
    local_train_targets = []
    local_train_stages = []
    local_train_actions = []
    local_train_rewards = []
    local_train_dones = []
    local_train_next_imgs = []
    local_train_next_grippers = []

    camera = CameraConfig(image_size=(224, 224), depth=False, point_cloud=False, mask=False)
    obs_config = ObservationConfig(left_shoulder_camera=camera, right_shoulder_camera=camera, front_camera=camera, overhead_camera=camera)
    obs_config.front_camera.render_mode = RenderMode.OPENGL
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=True), gripper_action_mode=Discrete()),
        obs_config=obs_config,
        headless=True)
    env.launch()
    task = env.get_task(PickDescribedObject)
    task.set_variation(i)

    j = 0
    while j < total_episodes:
        try:
            img_paths, instructions, gripper_poses, items, object_positions, target_positions, stages, actions, rewards, dones, next_img_paths, next_gripper_poses = run_episode(task, i, j, save_root)
            j += 1
            print(f"variation{i}, epoisode{j} done")
            local_train_imgs += img_paths
            local_train_instructions += instructions
            local_train_gripper += gripper_poses
            local_train_items += items
            local_train_objects += object_positions
            local_train_targets += target_positions
            local_train_stages += stages
            local_train_actions += actions
            local_train_rewards += rewards
            local_train_dones += dones
            local_train_next_imgs += next_img_paths
            local_train_next_grippers += next_gripper_poses

        except Exception as e:
            print(e)
            print(f"variation{i}, epoisode{j} failed")
            continue
            

    with lock:
        manager_dict['train_imgs'] += local_train_imgs
        manager_dict['train_instructions'] += local_train_instructions
        manager_dict['train_grippers'] += local_train_gripper
        manager_dict['train_items'] += local_train_items
        manager_dict['train_objects'] += local_train_objects
        manager_dict['train_targets'] += local_train_targets
        manager_dict['train_stages'] += local_train_stages
        manager_dict['train_actions'] += local_train_actions
        manager_dict['train_rewards'] += local_train_rewards
        manager_dict['train_dones'] += local_train_dones
        manager_dict['train_next_imgs'] += local_train_next_imgs
        manager_dict['train_next_grippers'] += local_train_next_grippers
        

    env.shutdown()

def main():
    save_root = './datasets/pick_described_object_replay/'
    check_and_make(save_root)
    manager = Manager()
    lock = manager.Lock()

    # Create shared lists
    manager_dict = manager.dict({
        'train_imgs': [],
        'train_instructions': [],
        'train_grippers': [],
        'train_items': [],
        'train_objects': [],
        'train_targets': [],
        'train_stages': [],
        'train_actions': [],
        'train_rewards': [],
        'train_dones': [],
        'train_next_imgs': [],
        'train_next_grippers': []
    })

    total_episodes = 100
    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=process_variation, args=(i,total_episodes,save_root, manager_dict, lock))
        processes.append(p)
        p.start()
        print(f"Process {i} started")

    for p in processes:
        p.join()

    print('Data collection done!')
    train_data = {
        'images': manager_dict['train_imgs'],
        'instructions': manager_dict['train_instructions'],
        'grippers': manager_dict['train_grippers'],
        'items': manager_dict['train_items'],
        'objects': manager_dict['train_objects'],
        'targets': manager_dict['train_targets'],
        'stages': manager_dict['train_stages'],
        'actions': manager_dict['train_actions'],
        'rewards': manager_dict['train_rewards'],
        'dones': manager_dict['train_dones'],
        'next_images': manager_dict['train_next_imgs'],
        'next_grippers': manager_dict['train_next_grippers']
    }

    torch.save(train_data, os.path.join(save_root, 'data.pt')) 

if __name__ == '__main__':
    main()