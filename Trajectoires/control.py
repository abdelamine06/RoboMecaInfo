import json
import math
import numpy as np
from abc import ABC, abstractmethod
from scipy import optimize

from homogeneous_transform import *

tol = 1e-9

def cosineLaw(x,y, L1, L2):
    """
    Parameters
    ----------
    x : double
    y : double
    L1: double
    L2: double

    Returns
    -------
    solutions : list((double,double))
        The list of couples (alpha, beta) that allows to reach the provided
        target
    """
    solutions = []
    dist = math.sqrt(x**2+y**2)
    if (dist < abs(L1 - L2)) or dist > (L1 + L2):
        return solutions
    phi = math.atan2(y, x)
    alpha = math.acos((L1**2+dist**2 - L2**2) / (2*L1* dist))
    beta = math.acos((L1**2+ L2**2 - dist**2) / (2*L1* L2))
    solutions.append(np.array([phi+alpha, beta - np.pi]))
    if abs(alpha) > tol:
        solutions.append(np.array([ phi-alpha, np.pi - beta]))
    return solutions

import sys
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
    def f(self):
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
    def computeDJacobian(self, joints, djoints):
        """
        Calcule la dérivée de la jacobienne par rapport au temps
        Parameters :
        joints : np.array
            The values of the joints of the robot in joint space
        djoints : np.array
            The velocities of the joints of the robot in joint space

        Returns
        -------
        np.array
            The time derivative of the jacobian of the robot for given joints values
        """
        #TODO:remove code vvv
        #On veut calculer dJ/dt => dJ/dq * dq/dt
        #https://en.wikipedia.org/wiki/Chain_rule
        J2 = self.computeJacobian(joints, 2)
        dJ = np.zeros(J2.shape)
        for i in range(J2.shape[0]):
            for j in range(J2.shape[1]):
                for k in range(len(djoints)):
                    dJ[i,j] += J2[i,j] * djoints[k]
        #print(np.abs(J2@djoints)-np.abs(dJ), file=sys.stderr)

        return J2 @ djoints
        #return dJ

    @abstractmethod
    def analyticalMGI(self, target):
        """
        Parameters
        ----------
        joints : np.arraynd shape(n,)
            The current values of the joints of the robot in joint space
        target : np.arraynd shape(m,)
            The target in operational space

        Returns
        -------
        nb_solutions : int
            The number of solutions for the given target, -1 if there is an
            infinity of solutions
        joint_pos : np.ndarray shape(X,) or None
            One of the joint configuration which allows to reach the provided
            target. If no solution is available, returns None.
        """

    def computeMGI(self, joints, target, method, seed = None):
        """
        Parameters
        ----------
        joints : np.ndarray shape(n,)
            The current position of joints in angular space
        target : np.ndarray shape(m,)
            The target in operational space
        method : str
            The method used to compute MGI, available choices:
            - analyticalMGI
            - jacobianInverse
            - jacobianTransposed
        seed : None or int
            The seed used for inner random components if needed
        """
        if method == "analyticalMGI":
            nb_sols, sol = self.analyticalMGI(target)
            return sol
        elif method == "jacobianInverse":
            return self.solveJacInverse(joints, target, seed)
        elif method == "jacobianTransposed":
            return self.solveJacTransposed(joints, target)
        raise RuntimeError("Unknown method: " + method)

    def solveJacInverse(self, joints, target, seed = None):
        """
        Parameters
        ----------
        joints: np.ndarray shape(n,)
            The initial position for the search in angular space
        target: np.ndarray shape(n,)
            The wished target for the tool in operational space
        seed: None or int
            Since the method comport some random part, the seed can be specified
            to obtain reproductible results.
        """
        max_iterations = 500
        max_step_size = 0.1
        tol = 10**-6
        for i in range(max_iterations):
            pos = self.computeMGD(joints)
            error = target - pos
            if np.linalg.norm(error) < tol:
                break
            try:
                J_inv = np.linalg.inv(self.computeJacobian(joints))
                step = J_inv @ error
                step_size = np.linalg.norm(step)
                if step_size > max_step_size:
                    step = step / step_size * max_step_size
                joints = joints + step
            except np.linalg.LinAlgError:
                noise_level = 1e-1
                joints = joints + np.random.default_rng(seed).uniform(
                    -noise_level, noise_level,
                    joints.shape[0])
        return joints

    def solveJacTransposed(self, joints, target):
        limits = self.getJointsLimits()
        cost_func = lambda x : np.linalg.norm(self.computeMGD(x) - target, 2)
        jac_func = lambda x : - 2 * (self.computeJacobian(x).transpose() @
                                     (target - self.computeMGD(x)))
        res = optimize.minimize(cost_func, joints,
                                jac= jac_func,
                                bounds = optimize.Bounds(limits[:,0], limits[:,1]))
        return res.x

class RTRobot(RobotModel):


    """
    Model a robot with a 2 degrees of freedom: 1 rotation and 1 translation

    The operational space of the robot is 2 dimensional because it can only move inside a plane
    """
    def __init__(self):
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

    def getOperationalDimensionLimits(self):
        max_dist = np.sqrt((self.L1 + self.max_q1)**2 + self.L2**2)
        return np.array([[-1,1],[-1,1]]) * max_dist

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ translation(joints[1] * np.array([1,0,0]))
        return T_0_1 @ T_1_2.dot(self.T_2_E)

    def computeMGD(self, joints):
        tool_pos = self.getBaseFromToolTransform(joints) @ np.array([0,0,0,1])
        return tool_pos[:2]

    def analyticalMGI(self, target):
        dist = np.linalg.norm(target)
        min_dist = np.sqrt(self.L1**2 + self.L2**2)
        max_dist = np.sqrt((self.L1 + self.max_q1)**2 + self.L2**2)
        if dist < min_dist or dist > max_dist:
            return 0, None
        # Using basic geometry to get distance of joint q1
        q1 = np.sqrt(dist**2 - self.L2**2) - self.L1
        # q0 = dir_to_target + target_offset
        dir_to_target = math.atan2(target[1], target[0])
        dir_offset = math.atan2(self.L2, self.L1+q1)
        q0 = dir_to_target + dir_offset
        return 1, np.array([q0,q1])

    def computeJacobian(self, joints):
        J = np.zeros((2,2), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        J[:,0] = (self.T_0_1 @ d_rot_z(joints[0]) @ self.T_1_2 @
                  translation(joints[1] * np.array([1,0,0])) @ self.T_2_E)[:2,3]
        J[:,1] = (self.T_0_1 @ rot_z(joints[0]) @ self.T_1_2 @
                  d_translation(np.array([1,0,0])) @ self.T_2_E)[:2,3]
        return J
    def computeDJacobian(self, joints, djoints):
        dJ = np.zeros((2,2), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        dJ[:,0] = (self.T_0_1 @ d_rot_z_dt(joints[0], d=1, dq=djoints[0]) @ self.T_1_2 @
                  d_translation_dt(joints[1] * np.array([1,0,0]), d=0, dq=djoints[1]) @ self.T_2_E)[:2,3]
        dJ[:,1] = (self.T_0_1 @ d_rot_z_dt(joints[0], 0, djoints[0]) @ self.T_1_2 @
                  d_translation_dt(np.array([1,0,0]), d=1, dq=djoints[1]) @ self.T_2_E)[:2,3]
        return dJ

class RRRRobot(RobotModel):
    """
    Model a robot with 3 degrees of freedom along different axis
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
        max_xy = self.L1 + self.L2 + self.L3
        min_z = self.L0 - self.L2 - self.L3
        max_z = self.L0 + self.L2 + self.L3
        return np.array([[-max_xy,max_xy],[-max_xy,max_xy],[min_z,max_z]])

    def getBaseFromToolTransform(self, joints):
        T_0_1 = self.T_0_1 @ rot_z(joints[0])
        T_1_2 = self.T_1_2 @ rot_x(joints[1])
        T_2_3 = self.T_2_3 @ rot_x(joints[2])
        return T_0_1 @ T_1_2 @ T_2_3 @ self.T_3_E

    def computeMGD(self, joints):
        tool_pos = self.getBaseFromToolTransform(joints) @ np.array([0,0,0,1])
        return tool_pos[:3]

    def analyticalMGI(self, target):
        # When X and Y of target are 'almost' zero, there is an infinity of solutions
        singularity = np.linalg.norm(target[:2]) < tol
        # First: use q0 to align target along y-axis:
        # - There's 2 potential solutions:
        theta = 0
        if not singularity:
            theta = math.atan2(target[1], target[0]) - np.pi/2
        solutions = []
        for q0 in [theta, theta + np.pi]:
            target_in_0 = np.zeros(4, dtype=np.double)
            target_in_0[:3] = target
            target_in_0[3] = 1
            # Put target in the proper referential:
            # only 2 rotations and 2 translations remaining
            target_in_2a = (invert_transform(self.T_1_2) @ rot_z(-q0) @
                            invert_transform(self.T_0_1)  @ target_in_0)
            for q12 in cosineLaw(target_in_2a[1], target_in_2a[2], self.L2, self.L3):
                solutions.append(np.array([q0,q12[0],q12[1]]))
        if len(solutions) == 0:
            return 0, None
        if singularity:
            return -1, solutions[0]
        return len(solutions), solutions[0]

    def computeJacobian(self, joints,d=1):
        J = np.zeros((3,3), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        J[:,0] = (self.T_0_1 @ d_rot_z(joints[0],d) @ self.T_1_2 @
                  rot_x(joints[1]) @ self.T_2_3 @ rot_x(joints[2]) @
                  self.T_3_E)[:3,3]
        J[:,1] = (self.T_0_1 @ rot_z(joints[0]) @ self.T_1_2 @
                  d_rot_x(joints[1],d) @ self.T_2_3 @ rot_x(joints[2]) @
                  self.T_3_E)[:3,3]
        J[:,2] = (self.T_0_1 @ rot_z(joints[0]) @ self.T_1_2 @
                  rot_x(joints[1]) @ self.T_2_3 @ d_rot_x(joints[2],d) @
                  self.T_3_E)[:3,3]
        return J

    def computeDJacobian(self, joints, djoints):
        dJ = np.zeros((3,3), dtype=np.double)
        # Derivation by joint[i] + picking up (x,y) from 4x4 matrix
        dJ[:,0] = (self.T_0_1 @ d_rot_z_dt(joints[0],1,djoints[0]) @
                    self.T_1_2 @ d_rot_x_dt(joints[1], 0, djoints[1]) @
                    self.T_2_3 @ d_rot_x_dt(joints[2],0,djoints[2]) @
                    self.T_3_E)[:3,3]
        dJ[:,1] = (self.T_0_1 @ d_rot_z_dt(joints[0],0,djoints[0]) @
                    self.T_1_2 @ d_rot_x_dt(joints[1],1,djoints[1]) @
                    self.T_2_3 @ d_rot_x_dt(joints[2],0,djoints[2]) @
                    self.T_3_E)[:3,3]
        dJ[:,2] = (self.T_0_1 @ d_rot_z_dt(joints[0],0,djoints[0]) @
                    self.T_1_2 @ d_rot_x_dt(joints[1],0,djoints[1]) @
                    self.T_2_3 @ d_rot_x_dt(joints[2],1, djoints[2]) @
                    self.T_3_E)[:3,3]
        #print(dJ-super().computeDJacobian(joints, djoints), file=sys.stderr)
        return dJ#super().computeDJacobian(joints, djoints)

class LegRobot(RobotModel):
    """
    Model of a simple robot leg with 4 degrees of freedom
    """
    def __init__(self):
        self.L0 = 1.01
        self.L1 = 0.4
        self.L2 = 0.3
        self.L3 = 0.3
        self.L4 = 0.2
        self.link_offset=0.02
        self.T_0_1 = translation([0,0,self.L0])
        self.T_1_2 = translation([self.link_offset,self.L1,0])
        self.T_2_3 = translation([-self.link_offset,self.L2,0])
        self.T_3_4 = translation([self.link_offset,self.L3,0])
        self.T_4_E = translation([0.0,self.L4,0])

    def getJointsNames(self):
        return ["q0", "q1", "q2", "q3"]

    def getJointsLimits(self):
        angle_lim = np.array([-np.pi, np.pi])
        L = np.zeros((4,2))
        for d in range(4):
            L[d,:] = angle_lim
        return L

    def getOperationalDimensionNames(self):
        return ["x","y","z", "r32"]

    def getOperationalDimensionLimits(self):
        xy_max = math.sqrt((self.L1 + self.L2 + self.L3 + self.L4)**2 + self.link_offset**2)
        z_offset = math.sqrt((self.L2 + self.L3 + self.L4)**2 + self.link_offset**2)
        z_min = self.L0 - z_offset
        z_max = self.L0 + z_offset
        return np.array([[-xy_max,xy_max],[-xy_max,xy_max],[z_min,z_max],[-1,1]])

    def getBaseFromToolTransform(self, joints):
        return (self.T_0_1 @ rot_z(joints[0]) @
                self.T_1_2 @ rot_x(joints[1]) @
                self.T_2_3 @ rot_x(joints[2]) @
                self.T_3_4 @ rot_x(joints[3]) @
                self.T_4_E)

    def extractMGD(self, T):
        """
        T : np.arraynd shape(4,4)
           An homogeneous transformation matrix
        """
        return np.append(T[:3,3],T[2,1])

    def computeMGD(self, joints):
        return self.extractMGD(self.getBaseFromToolTransform(joints))

    def analyticalMGI(self, target):
        solutions = []
        # Due to the link_offset, elements near 'z-axis' are unreachable
        XY_norm = np.linalg.norm(target[:2])
        if XY_norm < self.link_offset:
            return 0, None
        # q0 is the only element which can 'align' the direction of the tool
        # with respect to X,Y. Due to the link offset, it is more complex than
        # doing only atan2(Y,X)
        alpha = math.atan2(target[1],target[0]) - np.pi/2
        beta = math.atan2(self.link_offset, XY_norm)
        # By symetry we have two solutions, note beta sign changing
        for q0 in [alpha + beta, np.pi + alpha - beta]:
            # In referential post q1, target should be in [0.02, Y_in_1, Z_in_1]
            target_pos_in_q0 = np.concatenate((target[:3],[1]))
            target_pos_in_q1 = rot_z(-q0) @ invert_transform(self.T_0_1) @ target_pos_in_q0
            Y_in_1 = target_pos_in_q1[1]
            Z_in_1 = target_pos_in_q1[2]
            # Now we have aligned the elements, we also know that:
            # sin(q1+q2+q3) = target[3] (aka r_3,2)
            alpha = math.asin(target[3])
            for q123 in [alpha, np.pi - alpha]:
                # Target origin of 3 in basis 1 is determined by q123
                Y3_in_1 = Y_in_1 - math.cos(q123) * self.L4
                Z3_in_1 = Z_in_1 - math.sin(q123) * self.L4
                q12_solutions = cosineLaw(Y3_in_1 - self.L1, Z3_in_1, self.L2, self.L3)
                for q12 in q12_solutions:
                    q3 = q123 - q12[0] - q12[1]
                    solutions.append([q0, q12[0], q12[1], q3])
        nb_sols = len(solutions)
        if nb_sols == 0:
            return 0, None
        return nb_sols, solutions[0]

    def computeJacobian(self, joints,d=1):
        J = np.zeros((4,4),dtype=np.double)
        J[:,0] = self.extractMGD(self.T_0_1 @ d_rot_z(joints[0],d) @
                                 self.T_1_2 @ rot_x(joints[1]) @
                                 self.T_2_3 @ rot_x(joints[2]) @
                                 self.T_3_4 @ rot_x(joints[3]) @
                                 self.T_4_E)
        J[:,1] = self.extractMGD(self.T_0_1 @ rot_z(joints[0]) @
                                 self.T_1_2 @ d_rot_x(joints[1],d) @
                                 self.T_2_3 @ rot_x(joints[2]) @
                                 self.T_3_4 @ rot_x(joints[3]) @
                                 self.T_4_E)
        J[:,2] = self.extractMGD(self.T_0_1 @ rot_z(joints[0]) @
                                 self.T_1_2 @ rot_x(joints[1]) @
                                 self.T_2_3 @ d_rot_x(joints[2],d) @
                                 self.T_3_4 @ rot_x(joints[3]) @
                                 self.T_4_E)
        J[:,3] = self.extractMGD(self.T_0_1 @ rot_z(joints[0]) @
                                 self.T_1_2 @ rot_x(joints[1]) @
                                 self.T_2_3 @ rot_x(joints[2]) @
                                 self.T_3_4 @ d_rot_x(joints[3],d) @
                                 self.T_4_E)
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
