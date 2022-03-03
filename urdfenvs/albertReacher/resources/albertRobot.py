import pybullet as p
import gym
import os
import math
from urdfpy import URDF
import numpy as np

from urdfenvs.urdfCommon.differentialDriveRobot import DifferentialDriveRobot


class AlbertRobot(DifferentialDriveRobot):
    def __init__(self):
        n = 9
        fileName = os.path.join(os.path.dirname(__file__), 'albert.urdf')
        super().__init__(n, fileName)
        self._r = 0.08
        self._l = 0.494

    def setJointIndices(self):
        self.urdf_joints = [10, 11, 14, 15, 16, 17, 18, 19, 20]
        self.robot_joints = [24, 25, 8, 9, 10, 11, 12, 13, 14]
        self.castor_joints = [22, 23]

    def setAccLimits(self):
        accLimit = np.array([1.0, 1.0, 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
        self._limitAcc_j[0, :] = -accLimit
        self._limitAcc_j[1, :] = accLimit

