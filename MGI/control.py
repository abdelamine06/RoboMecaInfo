import json
import math
import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize

from homogeneous_transform import *

class Robotself:
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
    def getOperationalDimensionLimits(self):
        """
        Returns
        -------
        limits : np.array(x,2)
            The values limits for the operational dimensions, each row is a
            different dimension, column 0 is min, column 1 is max
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

    @abstractmethod
    def computeJacobian(self, joints):
        """
        Parameters
        ----------
        joints : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        np.array
            The jacobian of the robot for given joints values
        """

    @abstractmethod
    def analyticalMGI(self, target):
        """
        Parameters
        ----------
        joints : np.array
            The values of the joints of the robot in joint space

        Returns
        -------
        nb_solutions : int
            The number of solutions for the given target, -1 if there is an
            infinity of solutions
        joint_pos : np.ndarray shape(X,) or None
            One of the joint configuration which allows to reach the provided
            target. If no solution is available, returns None.
        """

    def computeMGI(self, joints, target, method, leg):
        """
        Parameters
        ----------
        joints : np.ndarray shape(X,)
            The current position of joints in angular space
        target : np.ndarray shape(X,)
            The target in operational space
        method : str
            The method used to compute MGI, available choices:
            - analyticalMGI
            - jacobianInverse
            - jacobianTransposed
        """

        if method == "analyticalMGI":
            nb_sols, sol = self.analyticalMGI(target)
            return sol
        elif method == "jacobianInverse":
            return self.solveJacInverse(joints, target,leg)
        elif method == "jacobianTransposed":
            return self.solveJacTransposed(joints, target)
        raise RuntimeError("Unknown method: " + method)

    def solveJacInverse(self, joints, target,leg):
        
        pos = self.computeMGD(joints)
        if leg==0:
            pos = pos[0:3]
            J = self.computeJacobian(joints)
            distance = target - pos
            epsilon = np.linalg.inv(J) @ distance
        else :
            J = self.computeJacobian(joints)
            realDistance = [0,0,0,0]
            distance = target - pos[0:3]
            realDistance[0:3] = distance
            realDistance[3] = 0

            newJ = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=np.double)
            newJ[0,:] = J[0,:]
            newJ[1,:] = J[1,:]
            newJ[2,:] = J[2,:]
            newJ[3,:] = [pos[3],pos[3],pos[3],pos[3]]
            epsilon = np.linalg.inv(newJ) @ realDistance

        return epsilon + joints




    def solveJacTransposed(self, joints, target):
        
        def distance(joints, self, target):
            pos = self.computeMGD(joints)
            distance = target - pos[0:3]
            return distance.transpose().dot(distance)

        def jacobian_distance(joints, self, target):
            J = self.computeJacobian(joints)
            pos = self.computeMGD(joints)
            return -2 * J[0:3].transpose().dot(target - pos[0:3])
        
        result = minimize(distance, joints, (self, target),
                                      jac=jacobian_distance)

        if result.success:
            return result.x
        return joints

class RTRobot(Robotself):
    """
    self a robot with a 2 degrees of freedom: 1 rotation and 1 translation

    The operational space of the robot is 2 dimensional because it can only move inside a plane
    """
    def __init__(self):
        self.L0 = 1.05
        self.L1 = 0.2
        self.L2 = 0.25
        self.T_0_1 = translation([0,0,self.L0])
        self.T_1_2 = translation([self.L1,0,0])
        self.T_2_E = translation([0.0,-self.L2,0])

    def getJointsNames(self):
        return ["q0", "q1"]

    def getJointsLimits(self):
        return np.array([[-np.pi,np.pi],[0,0.25]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y"]

    def getOperationalDimensionLimits(self):
        #TODO: refine limits
        dimension = math.sqrt(np.square(self.L1) + np.square(2*self.L2))
        return np.array([[-dimension,dimension],[-dimension,dimension]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ translation(joints[1] * np.array([1,0,0]))
        return T_0_1 @ T_1_2.dot(self.T_2_E)

    def computeMGD(self, joints):
        tool_pos = self.getBaseFromToolTransform(joints) @ np.array([0,0,0,1])
        return tool_pos[:2]

    def analyticalMGI(self, target):
        Distance = np.linalg.norm(target)
        if np.square(Distance) < np.square(self.L2):
            return 0, None
        d2 = math.sqrt(np.square(Distance) - np.square(self.L2)) - self.L1
        if d2 < 0 or d2 > self.L2:
            return 0,None
        direction  = math.atan2(target[1], target[0])
        final = math.atan2(self.L2,self.L1+d2)
        teta1 = direction + final
        return 1,np.array([teta1,d2])

    def computeJacobian(self, joints):
        T_0_1 = self.T_0_1 @ drot_z(joints[0])
        T_1_2 = self.T_1_2 @ translation(joints[1] * np.array([1,0,0]))
        dQ0 = T_0_1 @ T_1_2.dot(self.T_2_E)
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ dtranslation(joints[1] *np.array([1,0,0]))
        dQ1 = T_0_1 @ T_1_2.dot(self.T_2_E)
        J = np.array([[0,0],[0,0]],dtype=np.double)
        J[:,0] = dQ0[:2,3]
        J[:,1] = dQ1[:2,3]
        return J

class RRRRobot(Robotself):
    """
    self a robot with 3 degrees of freedom along different axis
    """
    def __init__(self):
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
        return np.array([[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y","z"]

    def getOperationalDimensionLimits(self):
        #TODO: refine limits
        dim_x = self.L1+self.L2+self.L3
        dim_y = dim_x
        dim_z = self.L2+self.L3
        return np.array([[-dim_x,dim_x],[-dim_y,dim_y],[self.L0-dim_z,self.L0+dim_z]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ rot_x(joints[1])
        T_2_3 = self.T_2_3 @ rot_x(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, joints):
        tool_pos = self.getBaseFromToolTransform(joints) @ np.array([0,0,0,1])
        return tool_pos[:3]

    def analyticalMGI(self, target):
        Distance = math.sqrt((self.L1**2 + self.L2**2 - 2*self.L1*self.L2*math.cos(self.L1/ self.L2)))
        
        teta2 = math.acos((Distance**2 - self.L1**2 - self.L2**2)/(2*self.L1*self.L2))
        
        
        alpha1 = math.atan(Distance)
        beta1 = math.acos((Distance**2 + self.L1**2 - self.L2**2)/(2*self.L1*math.sqrt(Distance**2)))
        teta1 = alpha1 + beta1


        PWx = self.L1 * math.cos(teta1) + self.L2 * math.cos(teta1 + teta2)
        PWy = self.L1 * math.sin(teta1) + self.L2 * math.sin(teta1 + teta2)

        b = target[1] - PWy
        a = target[0] - PWx

        x_origine_phi = -b/a
        Distance_phi = math.sqrt(PWy**2 + (PWx-x_origine_phi)**2)
        angle_phi = 180 - math.cos(Distance_phi/x_origine_phi)

        teta3 = angle_phi - teta1 - teta2


        return 1, np.array([teta1,teta2,teta3])


    def computeJacobian(self, joints):
        d0 = self.T_0_1 @ drot_z(joints[0])
        d0 = d0 @ self.T_1_2 @ rot_x(joints[1])
        d0 = d0 @ self.T_2_3 @ rot_x(joints[2])
        d0 = d0 @ self.T_3_E
        d1 = self.T_0_1 @ rot_z(joints[0])
        d1 = d1 @ self.T_1_2 @ drot_x(joints[1])
        d1 = d1 @ self.T_2_3 @ rot_x(joints[2])
        d1 = d1 @ self.T_3_E
        d2 = self.T_0_1 @ rot_z(joints[0])
        d2 = d2 @ self.T_1_2 @ rot_x(joints[1])
        d2 = d2 @ self.T_2_3 @ drot_x(joints[2])
        d2 = d2 @ self.T_3_E
        J = np.array([[0,0,0],[0,0,0],[0,0,0]],dtype=np.double)
        J[:,0] = d0[:3,3]
        J[:,1] = d1[:3,3]
        J[:,2] = d2[:3,3]
        return J


class LegRobot(Robotself):
    """
    self of a simple robot leg with 4 degrees of freedom
    """
    def __init__(self):
        #TODO: implement
        self.L0 = 1.01
        self.L1 = 0.4
        self.L2 = 0.3
        self.L3 = 0.3
        self.L4 = 0.21
        self.T_0_1 = translation([0,0,self.L0])
        self.T_1_2 = translation([0,self.L1,0])
        self.T_2_3 = translation([0,self.L2,0])
        self.T_3_4 = translation([0,self.L3,0])
        self.T_4_E = translation([0,self.L4,0])

    def getJointsNames(self):
        return ["q0", "q1", "q2","q3"]

    def getJointsLimits(self):
        return np.array([[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi],[-np.pi,np.pi]],dtype = np.double)

    def getOperationalDimensionNames(self):
        return ["x","y","z"]

    def getOperationalDimensionLimits(self):
        dim_x = self.L1+self.L2+self.L3+self.L4
        dim_y = dim_x
        dim_z = self.L2+self.L3+self.L4
        return np.array([[-dim_x,dim_x],[-dim_y,dim_y],[self.L0-dim_z,self.L0+dim_z]])


    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ rot_x(joints[1])
        T_2_3 = self.T_2_3 @ rot_x(joints[2])
        T_3_4 = self.T_2_3 @ rot_x(joints[3])
        correctionDecalage = translation([0.02,-0.01,0]) 
        return T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ self.T_4_E @ correctionDecalage

    def computeMGD(self, joints):
        T = self.getBaseFromToolTransform(joints)
        mgd = T[:4] @ np.array([0,0,0,1])
        mgd[3] = T[1,2]
        return mgd

    def analyticalMGI(self, target):
        #TODO: implement
        raise NotImplementedError()

    def computeJacobian(self, joints):
        d0 = self.T_0_1 @ drot_z(joints[0])
        d0 = d0 @ self.T_1_2 @ rot_x(joints[1])
        d0 = d0 @ self.T_2_3 @ rot_x(joints[2])
        d0 = d0 @ self.T_3_4 @ rot_x(joints[3])
        d0 = d0 @ self.T_4_E
        d1 = self.T_0_1 @ rot_z(joints[0])
        d1 = d1 @ self.T_1_2 @ drot_x(joints[1])
        d1 = d1 @ self.T_2_3 @ rot_x(joints[2])
        d1 = d1 @ self.T_3_4 @ rot_x(joints[3])
        d1 = d1 @ self.T_4_E
        d2 = self.T_0_1 @ rot_z(joints[0])
        d2 = d2 @ self.T_1_2 @ rot_x(joints[1])
        d2 = d2 @ self.T_2_3 @ drot_x(joints[2])
        d2 = d2 @ self.T_3_4 @ rot_x(joints[3])
        d2 = d2 @ self.T_4_E
        d3 = self.T_0_1 @ rot_z(joints[0])
        d3 = d3 @ self.T_1_2 @ rot_x(joints[1])
        d3 = d3 @ self.T_2_3 @ rot_x(joints[2])
        d3 = d3 @ self.T_3_4 @ drot_x(joints[3])
        d3 = d3 @ self.T_4_E
        J = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=np.double)
        J[:,0] = d0[:3,3]
        J[:,1] = d1[:3,3]
        J[:,2] = d2[:3,3]
        J[:,3] = d3[:3,3]
        return J    

def getRobotModel(robot_name):
    robot = None
    if robot_name == "rt":
        robot = RTRobot()
    elif robot_name == "rrr":
        robot = RRRRobot()
    elif robot_name == "leg":
        robot = LegRobot()
    else:
        raise RuntimeError("Unknown robot name: '" + robot_name + "'")
    return robot
