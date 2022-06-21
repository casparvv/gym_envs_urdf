from urdfenvs.point_robot_urdf.envs.point_robot_env import PointRobotEnv


class PointRobotVelEnv(PointRobotEnv):
    def __init__(self, render=False, dt=0.01):
        super().__init__(render=render, dt=dt)

    def apply_action(self, action):
        self._robot.apply_velocity_action(action)

    def set_spaces(self):
        (
            self.observation_space,
            self.action_space,
        ) = self._robot.get_velocity_spaces()
