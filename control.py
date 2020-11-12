import json
import math
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize

from homogeneous_transform import *


class RobotModel:
    def getNbJoints(self):
        """
        Returns
        -------
        length : int
            The number of joints for the robot
        """
        return len(self.getJointsNames())

    @abstractmethod
    def getJointsNames(self):
        """
        Returns
        -------
        joint_names : string array
            The list of names given to robot joints
        """

    @abstractmethod
    def getJointsLimits(self):
        """
        Returns
        -------
        np.array
            The values limits for the robot joints, each row is a different
            joint, column 0 is min, column 1 is max
        """

    @abstractmethod
    def getOperationalDimensionNames(self):
        """
        Returns
        -------
        joint_names : string array
            The list of names of the operational dimensions
        """

    @abstractmethod
    def getBaseFromToolTransform(self, joints):
        """
        Parameters
        ----------
        joints_position : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The transformation matrix from base to tool
        """

    @abstractmethod
    def computeMGD(self, joint):
        """
        Parameters
        ----------
        joints_position : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The coordinate of the effectors in the operational space
        """


class RTRobot(RobotModel):
    """
    Model a robot with a 2 degrees of freedom: 1 rotation and 1 translation

    The operational space of the robot is 2 dimensional because it can only move inside a plane
    """
    def __init__(self):
        # TODO initialize some properties or transforms which are independent
        #      of joints configuration
        self.L0 = 1.05
        self.L1 = 0.2
        self.L2 = 0.25
        self.T_0_1 = translation([0,0,self.L0])
        self.T_1_2 = translation([self.L1,0,0])
        self.T_2_E = translation([0.0,-self.L2,0])
        self.max_q1 = 0.25 # [m]
    def getJointsNames(self):
        return ["q0", "q1"]

    def getJointsLimits(self):
        return np.array([[-np.pi,np.pi],[0,self.max_q1]],dtype = np.double)
    def getOperationalDimensionNames(self):
        return ["x","y"]

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ translation(joints[1] * np.array([1,0,0]))
        return T_0_1 @ T_1_2.dot(self.T_2_E)


    def computeMGD(self, joints):
        return None

class RRRRobot(RobotModel):
    """
    Model a robot with 3 degrees of freedom along different axis
    """
    def __init__(self):
        # TODO initialize some properties or transforms which are independent
        #      of joints configuration
        self.L0 = 1.01
        self.L1 = 0.4
        self.L2 = 0.3
        self.L3 = 0.31
        self.T_0_1 = translation([0,0,self.L0])
        self.T_1_2 = translation([0,self.L1,0])
        self.T_2_3 = translation([0.0,self.L2,0])
        self.T_3_E = translation([0.0,self.L3,0])


    def getJointsNames(self):
        return ["q0", "q1", "q2"]

    def getJointsLimits(self):
        #TODO use meaningful limits based on URDF content
        return np.array([[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y","z"]

    def getBaseFromToolTransform(self, joints):
        #TODO compute MGD
        max_xy = self.L1 + self.L2 + self.L3
        min_z = self.L0 - self.L2 - self.L3
        max_z = self.L0 + self.L2 + self.L3
        return np.array([[-max_xy,max_xy],[-max_xy,max_xy],[min_z,max_z]])


    def computeMGD(self, joints):
        #TODO compute MGD
        return None

def getRobotModel(robot_name):
    robot = None
    if robot_name == "rt":
        robot = RTRobot()
    elif robot_name == "rrr":
        robot = RRRRobot()
    else:
        raise RuntimeError("Unknown robot name: '" + robot_name + "'")
    return robot
