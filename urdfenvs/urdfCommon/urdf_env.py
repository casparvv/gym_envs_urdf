import gym
import time
import numpy as np
import pybullet as p
import warnings
from abc import abstractmethod

from urdfenvs.urdfCommon.plane import Plane
from urdfenvs.sensors.sensor import Sensor
from urdfenvs.urdfCommon.generic_robot import GenericRobot


class WrongObservationError(Exception):
    """Exception when observation lays outside the defined observation space.

    This Exception is initiated when an the observation is not within the
    defined observation space. The purpose of this exception is to give
    the user better information about which specific part of the observation
    caused the problem.
    """

    def __init__(self, msg: str, observation: dict, observationSpace):
        """Constructor for error message.

        Parameters
        ----------

        msg: Default error message
        observation: Observation when mismatch occured
        observationSpace: Observation space of environment
        """
        msg_ext = self.get_wrong_observation(observation, observationSpace)
        super().__init__(msg + msg_ext)

    def get_wrong_observation(self, o: dict, os) -> str:
        """Detecting where the error occured.

        Parameters
        ----------

        o: observation
        os: observation space
        """
        msg_ext = ":\n"
        msg_ext += self.check_dict(o, os)
        return msg_ext

    def check_dict(
        self, o_dict: dict, os_dict, depth: int = 1, tabbing: str = ""
    ) -> str:
        """Checking correctness of dictionary observation.

        This methods searches for the cause for wrong observation.
        It loops over all keys in this dictionary and verifies whether
        observation and observation spaces fit together. If this is not
        the case, the concerned key is checked again. As the observation
        might have nested dictionaries, this function is called
        recursively.

        Parameters
        ----------

        o_dict: observation dictionary
        os_dict: observation space dictionary
        depth: current depth of nesting
        tabbing: tabbing for error message
        """
        msg_ext = ""
        for key in o_dict.keys():
            if not os_dict[key].contains(o_dict[key]):
                if isinstance(o_dict[key], dict):
                    msg_ext += tabbing + key + "\n"
                    msg_ext += self.check_dict(
                        o_dict[key],
                        os_dict[key],
                        depth=depth + 1,
                        tabbing=tabbing + "\t",
                    )
                else:
                    msg_ext += self.check_box(
                        o_dict[key], os_dict[key], key, tabbing
                    )
        return msg_ext

    def check_box(
        self, o_box: np.ndarray, os_box, key: str, tabbing: str
    ) -> str:
        """Checks correctness of box observation.

        This methods detects which value in the observation caused the
        error to be raised. Then it updates the error message msg.

        Parameters
        ----------

        o_box: observation box
        os_box: observation space box
        key: key of observation
        tabbing: current tabbing for error message
        """
        msg_ext = tabbing + "Error in " + key + "\n"
        if isinstance(o_box, float):
            val = o_box
            if val < os_box.low[0]:
                msg_ext += f"{tabbing}\t{key}: {val} < {os_box.low[0]}\n"
            elif val > os_box.high[0]:
                msg_ext += f"{tabbing}\t{key}: {val} > {os_box.high[0]}\n"
            return msg_ext

        for i, val in enumerate(o_box):
            if val < os_box.low[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} < {os_box.low[i]}\n"
            elif val > os_box.high[i]:
                msg_ext += f"{tabbing}\t{key}[{i}]: {val} > {os_box.high[i]}\n"
        return msg_ext

def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array


def filter_shape_dim(
    dim: np.ndarray, shape_type: str, dim_len: int, default: np.ndarray
) -> np.ndarray:
    """
    Checks and filters the dimension of a shape depending
    on the shape, warns were necessary.

    Parameters
    ----------

    dim: the dimension of the shape
    shape_type: the shape type
    dim_len: the number of dimensions should equal dim_len
    default: fallback option for dim

    """

    # check dimensions
    if isinstance(dim, np.ndarray) and np.size(dim) is dim_len:
        pass
    elif dim is None:
        dim = default
    else:
        warnings.warn(
            f"{shape_type} dimension should be of"
            "type (np.ndarray, list) with shape = ({dim_len}, )\n"
            " currently type(dim) = {type(dim)}. Aborting..."
        )
        return default
    return dim


class UrdfEnv(gym.Env):
    """Generic urdf-environment for OpenAI-Gym"""

    def __init__(
        self, robot: GenericRobot, flatten_observation: bool = False,
        render: bool = False, dt: float = 0.01
    ) -> None:
        """Constructor for environment.

        Variables are set and the pyhsics engine is initiated. Either with
        rendering (p.GUI) or without (p.DIRECT). Note that rendering slows
        down the simulation.

        Parameters:

        robot: Robot instance to be simulated
        render: Flag if simulator should render
        dt: Time step for pyhsics engine
        """
        self._dt: float = dt
        self._t: float = 0.0
        self._robot: GenericRobot = robot
        self._render: bool = render
        self._done: bool = False
        self._num_sub_steps: float = 20
        self._obsts: list = []
        self._goals: list = []
        self._flatten_observation: bool = flatten_observation
        self._space_set = False
        if self._render:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)

    def n(self) -> int:
        return self._robot.n()

    def dt(self) -> float:
        return self._dt

    def t(self) -> float:
        return self._t

    @abstractmethod
    def set_spaces(self) -> None:
        """Set observation and action space."""
        pass

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Applies a given action to the robot."""
        pass

    def step(self, action):
        self._t += self.dt()
        # Feed action to the robot and get observation of robot's state
        self.apply_action(action)
        for obst in self._obsts:
            obst.update_bullet_position(p, t=self.t())
        for goal in self._goals:
            goal.update_bullet_position(p, t=self.t())
        p.stepSimulation()
        ob = self._get_ob()

        reward = 1.0

        if self._render:
            self.render()
        return ob, reward, self._done, {}

    def _get_ob(self) -> dict:
        """Compose the observation."""
        observation = self._robot.get_observation()
        if not self.observation_space.contains(observation):
            err = WrongObservationError(
                "The observation does not fit the defined observation space",
                observation,
                self.observation_space,
            )
            warnings.warn(str(err))
        if self._flatten_observation:
            return flatten_observation(observation)
        else:
            return observation

    def add_obstacle(self, obst) -> None:
        """Adds obstacle to the simulation environment.

        Parameters
        ----------

        obst: Obstacle from MotionPlanningEnv
        """
        # add obstacle to environment
        self._obsts.append(obst)
        obst.add_to_bullet(p)

        # refresh observation space of robots sensors
        sensors = self._robot.sensors()
        cur_dict = dict(self.observation_space.spaces)
        for sensor in sensors:
            cur_dict[sensor.name()] = sensor.get_observation_space()
        self.observation_space = gym.spaces.Dict(cur_dict)

        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started"
            )

    def get_obstacles(self) -> list:
        return self._obsts

    def add_goal(self, goal) -> None:
        """Adds goal to the simulation environment.

        Parameters
        ----------

        goal: Goal from MotionPlanningGoal
        """
        self._goals.append(goal)
        goal.add_to_bullet(p)

    def add_walls(
        self,
        dim=np.array([0.2, 8, 0.5]),
        poses_2d=None,
    ) -> None:
        """
        Adds walls to the simulation environment.

        Parameters
        ----------

        dim = [width, length, height]
        poses_2d = [[x_position, y_position, orientation], ...]
        """
        if poses_2d is None:
            poses_2d = [
                [-4, 0.1, 0],
                [4, -0.1, 0],
                [0.1, 4, 0.5 * np.pi],
                [-0.1, -4, 0.5 * np.pi],
            ]
        self.add_shapes(
            shape_type="GEOM_BOX", dim=dim, mass=0, poses_2d=poses_2d
        )

    def add_shapes(
        self,
        shape_type: str,
        dim=None,
        mass: float = 0,
        poses_2d: list = None,
        place_height=None,
    ) -> None:
        """
        Adds a shape to the simulation environment.

        Parameters
        ----------

        shape_type: str
            options are:
                "GEOM_SPHERE",
                "GEOM_BOX",
                "GEOM_CYLINDER",
                "GEOM_CAPSULE"
            .
        dim: np.ndarray or list
            dimensions for the shape, dependent on the shape_type:
                GEOM_SPHERE,    dim=[radius]
                GEOM_BOX,       dim=[width, length, height]
                GEOM_CYLINDER,  dim=[radius, length]
                GEOM_CAPSULE,   dim=[radius, length]
        mass: float
            objects mass (default = 0 : fixed shape)
        poses_2d: list
            poses where the shape should be placed. Each element
            must be of form [x_position, y_position, orientation]
        place_height: float
            z_position of the center of mass
        """
        if poses_2d is None:
            poses_2d = [[-2, 2, 0]]
        # convert list to numpy array
        if isinstance(dim, list):
            dim = np.array(dim)

        # create collisionShape
        if shape_type == "GEOM_SPHERE":
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_SPHERE", 1, default=np.array([0.5])
            )
            shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=dim[0])
            default_height = dim[0]

        elif shape_type == "GEOM_BOX":
            if dim is not None:
                dim = 0.5 * dim
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_BOX", 3, default=np.array([0.5, 0.5, 0.5])
            )
            shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
            default_height = dim[2]

        elif shape_type == "GEOM_CYLINDER":
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_CYLINDER", 2, default=np.array([0.5, 1.0])
            )
            shape_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=dim[0], height=dim[1]
            )
            default_height = 0.5 * dim[1]

        elif shape_type == "GEOM_CAPSULE":
            # check dimensions
            dim = filter_shape_dim(
                dim, "GEOM_CAPSULE", 2, default=np.array([0.5, 1.0])
            )
            shape_id = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=dim[0], height=dim[1]
            )
            default_height = dim[0] + 0.5 * dim[1]

        else:
            warnings.warn(
                "Unknown shape type: {shape_type}, aborting..."
            )
            return

        if place_height is None:
            place_height = default_height

        # place the shape at poses_2d
        for pose in poses_2d:
            p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=shape_id,
                baseVisualShapeIndex=-1,
                basePosition=[pose[0], pose[1], place_height],
                baseOrientation=p.getQuaternionFromEuler([0, 0, pose[2]]),
            )

        if self._t != 0.0:
            warnings.warn(
                "Adding an object while the simulation already started"
            )

    def add_sensor(self, sensor: Sensor) -> None:
        """Adds sensor to the robot.

        Adding a sensor requires an update to the observation space.
        This seems to require a conversion to dict and back to
        gym.spaces.Dict.
        """
        self._robot.add_sensor(sensor)
        cur_dict = dict(self.observation_space.spaces)
        cur_dict[sensor.name()] = sensor.get_observation_space()
        self.observation_space = gym.spaces.Dict(cur_dict)
        
    def show_lidar(self, sensor_data, q, body_ids_old, number_lidar_rays) -> np.ndarray:
        """Shows LiDAR rays/obstacles in the simulation environment.
        Use obs_type to either plot the rays, the obstacles created at the end of the rays
         or both to visualize both the rays and obstacles.

        Below is the function show_lidar_spheres, which only shows the spheres at the end of the rays. 
         The spheres only function is faster than this one.
        To do: find a way to also update the positions and shapes of the rays since the current implementation is too slow. 
         However, this seems not possible due to the visualShape length changing, this one is not 'updateable'.
        
        Parameters
        ----------
        
        sensor_data: Configuration space positions of LiDAR rays, relative to the robot.
        q: Configuration space state of the robot.
        body_ids_old: The ids of the previously added ray visualizations, used to remove old ones.
        number_lidar_rays: The total number of rays the LiDAR uses.
        
        Returns
        ----------
        
        body_ids: The new ray visualization ids.
        """
        obs_type = "both" # ray, obs or both
        # Remove previously added ray/obstacle visualizations.
        if body_ids_old is not None:
            # body_ids_old is None before creating the first visualizations.
            for body_id_old in body_ids_old:
                p.removeBody(int(body_id_old))
        body_ids = np.zeros(number_lidar_rays)
        if obs_type == "both":
            body_ids = np.zeros(number_lidar_rays*2)
        # Reshape and add z-values to the sensor data.
        q_obs = sensor_data.reshape(number_lidar_rays, 2)
        q_obs = np.append(q_obs, np.zeros((number_lidar_rays, 1)), axis = 1)
        # Calculate the angles of the rays/obstacles.
        angles = np.arange(number_lidar_rays)/number_lidar_rays*2*np.pi
        if obs_type == "ray" or obs_type == "both":
            # Calculate the length of the ray cylinders.
            xs = np.linalg.norm(q_obs, axis=1)
        if obs_type == "obs" or obs_type == "both":
            # Create a sphere visual shape for every obstacle at the end of the rays.
            shape_id_sphere = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.9,0.3,0,0.8]
            )
        for n in range(number_lidar_rays):
            if obs_type == "ray" or obs_type == "both":
                # Create a cylinder visual shape for each ray.
                shape_id_ray = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=0.02, length=xs[n], rgbaColor=[0.9,0,0.1,0.8]
                )
                # Create a Pybullet visualization for each ray using the visual shape.
                body_id_ray = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=shape_id_ray,
                    basePosition=q + np.divide(q_obs[n], 2) + [0, 0, 0.1],
                    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2 + angles[n]]),
                    )
                body_ids[n] = int(body_id_ray)
            
            if obs_type == "obs" or obs_type == "both":
                # Create a Pybullet visualization for each sphere using the visual shape.
                body_id_sphere = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=shape_id_sphere,
                    basePosition=q + q_obs[n] + [0, 0, 0.1],
                    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2 + angles[n]]),
                    )
                if obs_type == "both":
                    body_ids[number_lidar_rays + n] = int(body_id_sphere)
                else:
                    body_ids[n] = int(body_id_sphere)
        return body_ids
    
    def show_lidar_spheres(self, sensor_data, q, body_ids_old, number_lidar_rays) -> np.ndarray:
        """Shows sphere obstacles at the end of LiDAR rays in the simulation environment.

        Parameters
        ----------
        
        sensor_data: Configuration space positions of LiDAR rays, relative to the robot.
        q: Configuration space state of the robot.
        body_ids_old: The ids of the previously added ray visualizations.
        number_lidar_rays: The total number of rays the LiDAR uses.
        
        Returns
        ----------
        
        body_ids: The new ray visualization ids.
        """
        if body_ids_old is None:
            body_ids = np.zeros(number_lidar_rays)
            # Create a sphere visual shape for every obstacle at the end of the rays.
            shape_id_sphere = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.9,0.3,0,0.8]
            )
        # Reshape and add z-values to the sensor data.
        q_obs = sensor_data.reshape(number_lidar_rays, 2)
        q_obs = np.append(q_obs, np.zeros((number_lidar_rays, 1)), axis = 1)
        # Calculate the angles of the rays/obstacles.
        angles = np.arange(number_lidar_rays)/number_lidar_rays*2*np.pi 
        if body_ids_old is None:
            # The visualizations are created at the first time this function is executed.
            for n in range(number_lidar_rays):
                # Create a Pybullet visualization for each sphere using the visual shape.
                body_id_sphere = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=shape_id_sphere,
                    basePosition=q + q_obs[n] + [0, 0, 0.1],
                    baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2 + angles[n]]),
                    )
                body_ids[n] = int(body_id_sphere)
        else:
            # The visualizations are updated after the first time the function is executed, improving the speed over the show_lidar function.
            body_ids = body_ids_old
            for n in range(number_lidar_rays):
                p.resetBasePositionAndOrientation(int(body_ids[n]), q + q_obs[n] + [0, 0, 0.1], p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2 + angles[n]]))
        return body_ids

    def check_initial_state(self, pos: np.ndarray, vel: np.ndarray) -> tuple:
        """Filters initial state of the robot and returns a valid state."""

        if not isinstance(pos, np.ndarray) or not pos.size == self._robot.n():
            pos = np.zeros(self._robot.n())
        if not isinstance(vel, np.ndarray) or not vel.size == self._robot.n():
            vel = np.zeros(self._robot.n())
        return pos, vel

    def reset(self, pos: np.ndarray = None, vel: np.ndarray = None) -> dict:
        """Resets the simulation and the robot.

        Parameters
        ----------

        pos: np.ndarray: Initial joint positions of the robot
        vel: np.ndarray: Initial joint velocities of the robot
        """
        self._t = 0.0
        pos, vel = self.check_initial_state(pos, vel)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._num_sub_steps
        )
        self._robot.reset(pos=pos, vel=vel)
        if not self._space_set:
            self.set_spaces()
            self._space_set = True
        self.plane = Plane()
        p.setGravity(0, 0, -10.0)
        p.stepSimulation()
        return self._robot.get_observation()

    def render(self) -> None:
        """Rendering the simulation environment.

        As rendering is done rather by the self._render flag,
        only the sleep statement is called here. This speeds up
        the simulation when rendering is not desired.

        """
        time.sleep(self.dt())

    def close(self) -> None:
        p.disconnect()
