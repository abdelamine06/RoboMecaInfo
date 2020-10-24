#!/usr/bin/env python3

import argparse
import json
import math
import numpy as np
import pybullet as p
import pybullet_data
import sys
from time import sleep, time
import control as ctrl
import homogeneous_transform as ht


class UserDebugParameter:
    def __init__(self, name, initial_value, limits):
        self.name = name
        self.value = initial_value
        self.id = p.addUserDebugParameter(
            name, limits[0], limits[1], initial_value)

    def read(self, p):
        self.value = p.readUserDebugParameter(self.id)


class Simulation:
    """
    A class interfacing the pybullet simulation and the robot model

    Attributes
    ----------
    robot_name : str
        The name of the robot active in the simulation
    robot_model : ctrl.RobotModel
        The physical model of the robot
    tool_pos_id : int
        The pybullet identifier for the tool_pos object
    robot : int
        The pybullet identifier for the robot object
    dt : float
        The time for each simulation step
    t : float
        The simulation time elapsed since beginning of the simulation
    log : file stream or None
        The file to which log data are sent, None if log has never been opened
    last_tick : float or None
        Time in seconds since simulation was ticked for last time, None if it
        has never been ticked
    last_tool_pos : np.ndarray shape(3,)
        Last position of the tool according to robot model and joints position
        in world basis
    operational_pos : np.ndarray shape(N,)
        Position of the tool in the operational space, number of dimensions
        depends on the robot_model
    tool_pos : np.ndarray shape(3,)
        Current position of the tool according to robot model and joints in
        world basis
    tool_orientation : np.ndarray shape(4,)
        The quaternion describing the orientation of the tool in the world
        referential
    user_parameters : list of UserDebugParameter
        The parameters that can be used for the simulation
    joints : np.ndarray shape(4,)
        The last values of the robot joints
    """

    def __init__(self, robot_name, log_path, dt):
        self.robot_name = robot_name
        self.dt = dt
        self.robot_model = ctrl.getRobotModel(robot_name)
        self.launchSimulation()
        self.addUserDebugParameters()
        self.initMemory()
        self.logStart(log_path)

    def __del__(self):
        if self.log is not None:
            self.log.close()

    def launchSimulation(self):
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        # Loading ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        # Loading tool position urdf
        self.tool_pos_id = p.loadURDF(
            "resources/tool_position.urdf", [0, 0, 0], useFixedBase=True)
        # Loading robot
        self.robot = p.loadURDF(
            "resources/{:}_robot.urdf".format(self.robot_name), useFixedBase=True)
        urdf_nb_joints = p.getNumJoints(self.robot)
        robot_model_nb_joints = self.robot_model.getNbJoints()
        if (robot_model_nb_joints != urdf_nb_joints):
            raise RuntimeError("Mismatch in number of joints : urdf has {:} and robot_model has {:}".format(
                urdf_nb_joints, robot_model_nb_joints))

        # Time management
        p.setRealTimeSimulation(0)
        self.t = 0
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

    def initMemory(self):
        self.last_tick = None
        self.last_tool_pos = None
        self.operational_pos = None
        self.tool_pos = None
        self.tool_orientation = None

    def addUserDebugParameters(self):
        self.user_parameters = []
        joint_names = self.robot_model.getJointsNames()
        joint_limits = self.robot_model.getJointsLimits()
        for i in range(self.robot_model.getNbJoints()):
            initial_value = (joint_limits[i, :].mean())
            self.user_parameters.append(UserDebugParameter(
                joint_names[i], initial_value, joint_limits[i, :]))

    def updateStatus(self):
        self.joints = np.array([0.0] * self.robot_model.getNbJoints())
        for i in range(self.robot_model.getNbJoints()):
            self.joints[i] = p.getJointState(self.robot, i)[0]
        for param in self.user_parameters:
            param.read(p)

    def updateMGD(self):
        self.operational_pos = self.robot_model.computeMGD(
            self.joints)
        world_from_tool = self.robot_model.getBaseFromToolTransform(
            self.joints)
        # TODO update tool_pos based on 'world_from_tool'
        self.tool_pos = np.array([0,0,0])
        # TODO update tool_orientation based on 'world_from_tool'
        self.tool_orientation = np.array([0,0,0,1])

    def tick(self):
        for i in range(self.robot_model.getNbJoints()):
            ctrl_mode = p.POSITION_CONTROL
            p.setJointMotorControl2(
                self.robot, i, ctrl_mode, self.user_parameters[i].value)

        # Make sure that time spent is not too high
        now = time()
        if not self.last_tick is None:
            tick_time = now - self.last_tick
            sleep_time = self.dt - tick_time
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                print("Time budget exceeded: {:}".format(
                    tick_time), file=sys.stderr)
        self.last_tick = time()
        self.t += self.dt
        p.stepSimulation()

        self.updateStatus()
        self.updateMGD()
        if self.tool_pos is not None:
            self.updateToolPos()
            # TODO Draw trajectory of the tool based on addUserDebugLine
            self.last_tool_pos = self.tool_pos

        self.logStep()

    def setTargetPos(self, pos):
        p.resetBasePositionAndOrientation(
            self.tool_target_id, pos, [0, 0, 0, 1])

    def updateToolPos(self):
        p.resetBasePositionAndOrientation(
            self.tool_pos_id, self.tool_pos, self.tool_orientation)

    def logStart(self, path):
        if path is None:
            self.log = None
            return
        self.log = open(path, "w")
        self.log.write("{:},{:},{:},{:}\n".format(
            "t", "name", "source", "value"))

    def logStep(self):
        # Skip logging step if not activated
        if self.log is None:
            return
        for i in range(self.robot_model.getNbJoints()):
            name = self.user_parameters[i].name
            target = self.user_parameters[i].value
            self.log.write("{:},{:},{:},{:}\n".format(
                self.t, name, "target", target))
        # TODO Add the real value of the joints read from pybullet
        # TODO Write the position of the tool in operational space


if __name__ == "__main__":
    # Reading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="rt",
                        help="The name of the robot to be used: rt, rrr")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Duration of a simulation step [s]")
    parser.add_argument("--duration", type=float, default=-1,
                        help="Duration of the simulation [s]")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to the output log file")
    args = parser.parse_args()

    # Launching simulation
    simulation = Simulation(args.robot, args.log, args.dt)
    simulation.updateStatus()
    while args.duration < 0 or simulation.t < args.duration:
        simulation.tick()
        simulation.updateStatus()
