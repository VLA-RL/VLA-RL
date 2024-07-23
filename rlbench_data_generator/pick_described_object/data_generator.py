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

import multiprocessing
from multiprocessing import Manager
import os

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

GROCERY_NAMES = [
    "chocolate jello",
    "soup",
    "spam",
    "mustard",
    "sugar",
]

reasoning = [
    "Gripper haven't grasped the {item}",
    "Gripper grasped the {item}, but it is not in the basket",
    "Gripper grasped the {item} and it is above the basket",
]
steps = [
    "Move to the {item} and pick it up.",
    "Move over the basket.",
    "Move down and place the {item} in the basket.",
]
subtasks = [
    "1.Move to the {item} and pick it up. 2.Move over the basket. 3.Place the {item} in the basket.",
    "1.Move over the basket. 2.Place the {item} in the basket.",
    ""
]
prompt = """
{item} POSE: {target_item_pose}
BASKET POSE: {basket_position}
GRIPPER POSE: {gripper_pose}
REASONING: {REASONING}
SUBTASKS: {SUBTASKS}
CURRENT STEP: {STEP}
ACTION: {action}
"""

def run_episode(task, variation_num):
    target_item_poses = []
    waypoints = []
    gripper_poses = []
    front_rgbs = []

    def callable_fun(obs, task, variation_num):
        # target item pose
        target_item_pose = task._task.get_graspable_objects()[variation_num].get_pose()
        target_item_pose = np.concatenate([target_item_pose[:3], R.from_quat(target_item_pose[3:]).as_euler('xyz')])
        target_item_poses.append(target_item_pose)

        #waypoints
        current_pose = obs.gripper_pose
        current_pose = np.concatenate([current_pose[:3], R.from_quat(current_pose[3:]).as_euler('xyz')])
        gripper_poses.append(np.concatenate([current_pose, [obs.gripper_open]]))
        wps = [wp._waypoint.get_position() for wp in task._task._waypoints]
        
        if abs(current_pose[:3] - wps[0]).mean() < 1e-1:
            waypoints.append(0)
        elif abs(current_pose[:3] - wps[1]).mean() < 1e-2:
            waypoints.append(1)
        elif abs(current_pose[:3] - wps[2]).mean() < 1e-2:
            waypoints.append(2)
        else:
            waypoints.append(-1)

        #front rgb
        front_rgbs.append(obs.front_rgb)

    task.reset()
    _ = task._scene.get_demo(callable_each_step=lambda obs: callable_fun(obs,task=task, variation_num=variation_num)) 
    basket_position = task._task.dropin_box.get_position()

    id_0 = int(np.where(np.array(waypoints) == 0)[0].mean())
    id_1 = int(np.where(np.array(waypoints) == 1)[0].mean())
    id_2 = int(np.where(np.array(waypoints) == 2)[0].mean())

    keyframe = lambda a,b, c, gap: [(a-(i+1)*gap, a, c) for i in range((a - b)//gap)]

    keyframes = keyframe(id_0, 0, 0, 8) + keyframe(id_1, id_0, 1, 8) + keyframe(id_2, id_1,  2, 8)

    imgs = []
    instruction = "pick up the %s and place in the basket" % GROCERY_NAMES[variation_num]
    instructions = []
    cots = []
    target_item_poses_ = []
    gripper_poses_ = []
    basket_positions_ = []
    actions = []

    for cur_id, key_id, step in keyframes:
        item = GROCERY_NAMES[variation_num]
        inputs = prompt.format(item = item,
                    target_item_pose = "{target_item_pose}",
                    basket_position = "{basket_position}",
                    gripper_pose = "{gripper_pose}",
                    REASONING = reasoning[step].format(item=item),   
                    STEP = steps[step].format(item=item),
                    SUBTASKS = subtasks[0].format(item=item),
                    action = "{action}")
        imgs.append(front_rgbs[cur_id])
        instructions.append(instruction)
        cots.append(inputs)
        target_item_poses_.append(target_item_poses[cur_id])
        gripper_poses_.append(gripper_poses[cur_id])
        basket_positions_.append(basket_position)
        actions.append(gripper_poses[key_id])
    return imgs, instructions, cots, target_item_poses_, gripper_poses_, basket_positions_, actions


# Define the function to be executed in each process
def process_variation(i, manager_dict, lock):
    local_train_imgs = []
    local_train_instructions = []
    local_train_cots = []
    local_train_target_item_poses = []
    local_train_gripper_poses = []
    local_train_basket_positions = []
    local_train_actions = []

    local_test_imgs = []
    local_test_instructions = []
    local_test_cots = []
    local_test_target_item_poses = []
    local_test_gripper_poses = []
    local_test_basket_positions = []
    local_test_actions = []

    camera = CameraConfig(image_size=(224, 224), depth=False, point_cloud=False, mask=False)
    obs_config = ObservationConfig(left_shoulder_camera=camera, right_shoulder_camera=camera, front_camera=camera, overhead_camera=camera)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=True), gripper_action_mode=Discrete()),
        obs_config=obs_config,
        headless=True)
    env.launch()
    task = env.get_task(PickDescribedObject)
    task.set_variation(i)

    j = 0
    while j < 10:
        try:
            imgs, instructions, cots, target_item_poses, gripper_poses, basket_positions, actions = run_episode(task, i)
            j += 1
            print(f"variation{i}, epoisode{j} done")
            if j < 8:
                local_train_imgs += imgs
                local_train_instructions += instructions
                local_train_cots += cots
                local_train_target_item_poses += target_item_poses
                local_train_gripper_poses += gripper_poses
                local_train_basket_positions += basket_positions
                local_train_actions += actions
            else:
                local_test_imgs += imgs
                local_test_instructions += instructions
                local_test_cots += cots
                local_test_target_item_poses += target_item_poses
                local_test_gripper_poses += gripper_poses
                local_test_basket_positions += basket_positions
                local_test_actions += actions
        except Exception as e:
            print(e)
            print(f"variation{i}, epoisode{j} failed")
            continue
            

    with lock:
        manager_dict['train_imgs'] += local_train_imgs
        manager_dict['train_instructions'] += local_train_instructions
        manager_dict['train_cots'] += local_train_cots
        manager_dict['train_target_item_poses'] += local_train_target_item_poses
        manager_dict['train_gripper_poses'] += local_train_gripper_poses
        manager_dict['train_basket_positions'] += local_train_basket_positions
        manager_dict['train_actions'] += local_train_actions

        manager_dict['test_imgs'] += local_test_imgs
        manager_dict['test_instructions'] += local_test_instructions
        manager_dict['test_cots'] += local_test_cots
        manager_dict['test_target_item_poses'] += local_test_target_item_poses
        manager_dict['test_gripper_poses'] += local_test_gripper_poses
        manager_dict['test_basket_positions'] += local_test_basket_positions
        manager_dict['test_actions'] += local_test_actions

def main():
    manager = Manager()
    lock = manager.Lock()

    # Create shared lists
    manager_dict = manager.dict({
        'train_imgs': [],
        'train_instructions': [],
        'train_cots': [],
        'train_target_item_poses': [],
        'train_gripper_poses': [],
        'train_basket_positions': [],
        'train_actions': [],
        'test_imgs': [],
        'test_instructions': [],
        'test_cots': [],
        'test_target_item_poses': [],
        'test_gripper_poses': [],
        'test_basket_positions': [],
        'test_actions': []
    })

    processes = []
    for i in range(5):
        p = multiprocessing.Process(target=process_variation, args=(i, manager_dict, lock))
        processes.append(p)
        p.start()
        print(f"Process {i} started")

    for p in processes:
        p.join()

    print('Data collection done!')
    train_data = {
        'imgs': manager_dict['train_imgs'],
        'instructions': manager_dict['train_instructions'],
        'cots': manager_dict['train_cots'],
        'target_item_poses': manager_dict['train_target_item_poses'],
        'gripper_poses': manager_dict['train_gripper_poses'],
        'basket_positions': manager_dict['train_basket_positions'],
        'actions': manager_dict['train_actions']
    }
    test_data = {
        'imgs': manager_dict['test_imgs'],
        'instructions': manager_dict['test_instructions'],
        'cots': manager_dict['test_cots'],
        'target_item_poses': manager_dict['test_target_item_poses'],
        'gripper_poses': manager_dict['test_gripper_poses'],
        'basket_positions': manager_dict['test_basket_positions'],
        'actions': manager_dict['test_actions']
    }

    save_dir = './datasets/pick_described_object'
    check_and_make(save_dir)

    torch.save(train_data, os.path.join(save_dir, 'train_data.pt')) 
    torch.save(test_data, os.path.join(save_dir, 'test_data.pt'))

if __name__ == '__main__':
    main()