import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.pick_described_object import PickDescribedObject


class Agent(object):
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]
        return np.concatenate([arm, gripper], axis=-1)


if __name__ == "__main__":
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        ),
        obs_config=ObservationConfig(),
        robot_setup="panda",
        headless=False,
    )
    env.launch()

    task = env.get_task(PickDescribedObject)

    agent = Agent(env.action_shape)
    training_steps = 120
    episode_length = 40
    obs = None
    for i in range(training_steps):
        if i % episode_length == 0:
            print("Reset Episode")
            descriptions, obs = task.reset()
            item_name = task._task.item_name
            item_coord = task._task.item.get_position()
            item_orientation = task._task.item.get_orientation()
            print(item_name)
            print(item_coord)
            print(item_orientation)
        action = agent.act(obs)
        obs, reward, terminate = task.step(action)
        print(reward)

    print("Done")
    env.shutdown()
