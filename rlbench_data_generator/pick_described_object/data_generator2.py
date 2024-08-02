from PIL import Image
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, JointVelocity, JointPosition, EndEffectorPoseViaPlanning, EndEffectorPoseViaIK

from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import PutGroceriesInCupboard, PickAndLift, StackBlocks, PlaceHangerOnRack, PickDescribedObject, TakeLidOffSaucepan, SetTheTable
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

def randomize_pose(task):
    np.random.seed()
    while True:
        try:
            pos_lower_bound = np.array([-0.2, -0.35, task._scene._workspace_minz])
            pos_upper_bound = np.array([task._scene._workspace_maxx, 0.35, 1.3])
            rot_lower_bound = np.array([-np.pi/2, 0, -np.pi/2])
            rot_upper_bound = np.array([np.pi/2, np.pi/2, np.pi/2])
            pos = np.random.uniform(pos_lower_bound, pos_upper_bound)
            euler = np.random.uniform(rot_lower_bound, rot_upper_bound)
            euler[0] = np.clip(np.random.normal(0, np.pi/6),-np.pi/2,np.pi/2)
            euler[1] = -np.clip(abs(np.random.normal(0,np.pi/6)),0, np.pi/2)
            trans = lambda rx: rx - np.pi if rx > 0 else rx + np.pi 
            euler[0] = trans(euler[0])
            quat = R.from_euler('xyz', euler).as_quat()
            joint_position = task._scene.robot.arm.solve_ik_via_sampling(position=pos, euler=euler,trials=10)
            task.step(np.concatenate([pos, quat, [0]]))
            break
        except Exception as e:
            print(e)
            continue
    return joint_position, pos, euler, quat

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
    # seed = np.random.randint(0, 100000)
    # np.random.seed(seed)
    # random.seed(seed)
    
    img_paths = []
    instructions = []
    gripper_poses = []
    items = []
    object_positions = []
    target_positions = []
    stages = []
    actions = []
    id = 0
    trial_times = 3
    task.set_variation(variation_num)
    save_dir = save_root + f"variation_{variation_num}/episode_{episode_num}/"
    check_and_make(save_dir)
    desc, _ = task.reset()
    for i in range(trial_times):
        #stage 0
        stage = 0
        joint_position, pos, euler, quat = randomize_pose(task)
        img, gripper_pose, gripper_open, object_pos, target_pos = get_data(task, variation_num)
        task_action = np.concatenate([pos, quat, [1]])
        task.step(task_action)
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
        action = pose2action(gripper_pose, 1)
        actions.append(action)
        id += 1

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
        id+= 1

    task_action = np.concatenate([waypoint_pose, [0]])
    task.step(task_action)
    
    #stage 1
    stage = 1
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
    action_open = 1
    action = pose2action(waypoint_pose, action_open)
    actions.append(action)
    id += 1 
    
    for i in range(1):
        randomize_pose(task)
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
        action = pose2action(waypoint_pose, action_open)
        actions.append(action)
        id += 1
    task_action = np.concatenate([waypoint_pose, [action_open]])
    task.step(task_action)

    assert task._scene.task.success()[0] == True, "Task failed"

    return img_paths, instructions, gripper_poses, items, object_positions, target_positions, stages, actions

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

    local_test_imgs = []
    local_test_instructions = []
    local_test_gripper = []
    local_test_items = []
    local_test_objects = []
    local_test_targets = []
    local_test_stages = []
    local_test_actions = []


    camera = CameraConfig(image_size=(448, 448), depth=False, point_cloud=False, mask=False)
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
            img_paths, instructions, gripper_poses, items, object_positions, target_positions, stages, actions = run_episode(task, i, j, save_root)
            j += 1
            print(f"variation{i}, epoisode{j} done")
            if j < total_episodes*0.9:
                local_train_imgs += img_paths
                local_train_instructions += instructions
                local_train_gripper += gripper_poses
                local_train_items += items
                local_train_objects += object_positions
                local_train_targets += target_positions
                local_train_stages += stages
                local_train_actions += actions
            else:
                local_test_imgs += img_paths
                local_test_instructions += instructions
                local_test_gripper += gripper_poses
                local_test_items += items
                local_test_objects += object_positions
                local_test_targets += target_positions
                local_test_stages += stages
                local_test_actions += actions
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

        manager_dict['test_imgs'] += local_test_imgs
        manager_dict['test_instructions'] += local_test_instructions
        manager_dict['test_grippers'] += local_test_gripper
        manager_dict['test_items'] += local_test_items
        manager_dict['test_objects'] += local_test_objects
        manager_dict['test_targets'] += local_test_targets
        manager_dict['test_stages'] += local_test_stages
        manager_dict['test_actions'] += local_test_actions
        

    env.shutdown()

def main():
    save_root = './datasets/pick_described_object2/'
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

        'test_imgs': [],
        'test_instructions': [],
        'test_grippers': [],
        'test_items': [],
        'test_objects': [],
        'test_targets': [],
        'test_stages': [],
        'test_actions': [],
    })

    total_episodes = 20
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
        'actions': manager_dict['train_actions']
    }
    test_data = {
        'images': manager_dict['test_imgs'],
        'instructions': manager_dict['test_instructions'],
        'grippers': manager_dict['test_grippers'],
        'items': manager_dict['test_items'],
        'objects': manager_dict['test_objects'],
        'targets': manager_dict['test_targets'],
        'stages': manager_dict['test_stages'],
        'actions': manager_dict['test_actions']
    }


    torch.save(train_data, os.path.join(save_root, 'train_data.pt')) 
    torch.save(test_data, os.path.join(save_root, 'test_data.pt'))

if __name__ == '__main__':
    main()