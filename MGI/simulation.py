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
    mode : str
        The mode used for the simulation, it can be chosen among:
        - mgd :: target is provided in joint space
        - analyticalMGI :: target provided in operational space, joint target is
                           retrieved through analyticalMGI
        - jacobianInverse :: target provided in operational space, joint target is
                             retrieved through jacobianInverse method
        - jacobianTransposed :: target provided in operational space, joint target is
                                retrieved through jacobianTransposed
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
    joints : np.ndarray shape(X,)
        The last values of the robot joints
    joints_target : np.ndarray shape(X,)
        The current target values for the robot joints
    operational_target : np.ndarray shape(X,)
        The target in operational space, None if mode is mgd
    """

    def __init__(self, robot_name, log_path, dt, mode, target):
        self.log = None
        self.robot_name = robot_name
        self.dt = dt
        self.mode = mode
        self.joints_target = None
        self.operational_target = None
        self.robot_model = ctrl.getRobotModel(robot_name)
        self.launchSimulation()
        self.addUserDebugParameters(target)
        self.initMemory()
        self.logStart(log_path)
        self.computeTime=0

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

    def addUserDebugParametersList(self, names, limits, initial_values = None):
        """
        Add to the list of current user parameters the parameters specified

        Parameters
        ----------
        names: list(str)
            The names of the debug parameters to add
        limits: np.ndarray shape(X,2)
            The limits for the debug parameters, each row concern another
            parameter, first column is min, second column is max
        initial_values: np.ndarray shape(X,) or None
            The initial values for the debug parameters. If None is used, value
            is chosen to be the middle of the limits
        """
        if len(names) != limits.shape[0]:
            raise RuntimeError("Incompatible sizes for names and limits: " +
                               str(len(names)) + " and " +
                               str(limits.shape[0]))
        if initial_values is not None:
            if initial_values.shape[0] != limits.shape[0]:
                raise RuntimeError("Incompatible sizes for initial_values and limits: " +
                                   str(initial_values.shape[0]) + " and " +
                                   str(limits.shape[0]))
        else:
            initial_values = limits.mean(axis=1)
        for i in range(len(names)):
            self.user_parameters.append(UserDebugParameter(
                names[i], initial_values[i], limits[i,:]))

    def addUserDebugParameters(self, target):
        self.user_parameters = []
        if self.mode == "mgd":
            self.addUserDebugParametersList(
                self.robot_model.getJointsNames(),
                self.robot_model.getJointsLimits(),
                target)
        else:
            self.addUserDebugParametersList(
                self.robot_model.getOperationalDimensionNames(),
                self.robot_model.getOperationalDimensionLimits(),
                target)

    def updateStatus(self):
        """
        Updates joints position and user parameters from simulations
        """
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
        self.tool_pos = (world_from_tool @ np.array([0, 0, 0, 1]))[:3]
        self.tool_orientation = ht.get_quat(world_from_tool)

    def getDebugAsArray(self):
        l = []
        for p in self.user_parameters:
            l.append(p.value)
        return np.array(l)

    def updateTargets(self):
        """
        Update joints_target and operational_target
        """
        if self.mode == "mgd":
            self.joints_target = self.getDebugAsArray()
            self.operational_target = None
        else:
            self.operational_target = self.getDebugAsArray()
            leg = 0
            if(self.robot_name == "leg"):
                leg = 1
            self.joints_target = self.robot_model.computeMGI(
                self.joints,
                self.operational_target,
                self.mode,
                leg)

    def tick(self):
        if self.joints_target is not None:
            for i in range(self.robot_model.getNbJoints()):
                ctrl_mode = p.POSITION_CONTROL
                p.setJointMotorControl2(
                    self.robot, i, ctrl_mode, self.joints_target[i])

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
        # Get compute time when updating robot model
        start_time = time()
        self.updateTargets()
        self.computeTime = time()-start_time
        
        if self.tool_pos is not None:
            self.updateToolPos()
            if not self.last_tool_pos is None:
                p.addUserDebugLine(self.last_tool_pos,
                                   self.tool_pos, [0, 0, 0], 1.0, 1.0)
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
        self.log.write("{:},{:},{:},{:},{:},{:}\n".format(
            "t", "name", "source", "value", "mode", "computeTime"))

    def logStep(self):
        # Skip logging step if not activated
        if self.log is None:
            return
        for i in range(self.robot_model.getNbJoints()):
            name = self.robot_model.getJointsNames()[i]
            measure = self.joints[i]
            self.log.write("{:},{:},{:},{:},{:},{:}\n".format(
                self.t, name, "measure", measure, self.mode,self.computeTime))
            if self.joints_target is not None:
                target = self.joints_target[i]
                self.log.write("{:},{:},{:},{:},{:},{:}\n".format(
                    self.t, name, "target", target, self.mode,self.computeTime))
        # Only log operational dimensions if available
        if self.operational_pos is not None:
            op_dimensions = self.robot_model.getOperationalDimensionNames()
            for i in range(len(op_dimensions)):
                name = op_dimensions[i]
                measure = self.operational_pos[i]
                self.log.write("{:},{:},{:},{:},{:},{:}\n".format(
                    self.t, name, "measure", measure,self.mode,self.computeTime))
                if self.operational_target is not None:
                    op_target = self.operational_target[i]
                    self.log.write("{:},{:},{:},{:},{:},{:}\n".format(
                        self.t,name, "operational_target",op_target, self.mode, self.computeTime
                    ))
        
        #TODO: write the duration of the last step

if __name__ == "__main__":
    # Reading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="mgd",
                        choices=["mgd","analyticalMGI","jacobianInverse","jacobianTransposed"],
                        help="The target specification mode for the simulator")
    parser.add_argument("--robot", type=str, default="rt",
                        help="The name of the robot to be used: rt, rrr")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Duration of a simulation step [s]")
    parser.add_argument("--duration", type=float, default=-1,
                        help="Duration of the simulation [s]")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to the output log file")
    parser.add_argument("--target",
                        type=lambda s: np.array([float(item) for item in s.split(',')]),
                        default=None,
                        help="The initial target to use ")
    args = parser.parse_args()

    # Launching simulation
    simulation = Simulation(args.robot, args.log, args.dt, args.mode, args.target)
    simulation.updateStatus()
    while args.duration < 0 or simulation.t < args.duration:
        simulation.tick()
        simulation.updateStatus()
