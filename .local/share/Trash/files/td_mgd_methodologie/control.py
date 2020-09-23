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
        pass

    def getJointsNames(self):
        return ["q0", "q1"]

    def getJointsLimits(self):
        #TODO use meaningful limits based on URDF content
        return np.array([[-10,10],[-5,5]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y"]

    def getBaseFromToolTransform(self, joints):
        return None

    def computeMGD(self, joints):
        return None

class RRRRobot(RobotModel):
    """
    Model a robot with 3 degrees of freedom along different axis
    """
    def __init__(self):
        # TODO initialize some properties or transforms which are independent
        #      of joints configuration
        pass

    def getJointsNames(self):
        return ["q0", "q1", "q2"]

    def getJointsLimits(self):
        #TODO use meaningful limits based on URDF content
        return np.array([[-10,10],[-10,10],[-10,10]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y","z"]

    def getBaseFromToolTransform(self, joints):
        #TODO compute MGD
        return None

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
