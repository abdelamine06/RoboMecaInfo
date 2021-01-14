#!/usr/bin/env python3

from abc import ABC, abstractmethod
import argparse
import json
import numpy as np

import model
import math

def buildController(dic):
    controller_name = dic.get("type")
    params = dic.get("params")
    if controller_name == "PIDController":
        return PIDController(params)
    elif controller_name == "OpenLoopEffortController":
        return OpenLoopEffortController(params)
    elif controller_name == "FeedForwardController":
        return FeedForwardController(params)
    elif controller_name == "OpenLoopPendulumEffortController":
        return OpenLoopPendulumEffortController(params)
    raise RuntimeError("Unknown controller name: {:}".format(controller_name))

def buildRobotController(dic):
    controllers = []
    for entry in dic.get("controllers"):
        controllers.append(buildController(entry))
    return RobotController(controllers)

def oneIn(list):
    for i in range(len(list)):
        element = list[i]
        if element is not None:
            return element, i
    return None, None

class Controller():
    """
    Implement a simple 1D controller
    """
    def __init__(self, params):
        self.cmd_max = params["cmd_max"]

        self.prev_vel = 0.0
        self.prev_t = None
        self.prev_vel = 0
        self.prev_pos = 0
        self.prev_acc = 0

    @abstractmethod
    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        """
        Parameters
        ----------
        measured_pos : float
        measured_vel : float
        ref_pos : float
        ref_vel : float
        ref_acc : float

        Returns
        -------
        cmd : float
            The command computed by the controller
        """
    def computeAcc(self, t, vel):
        if self.prev_vel is None or self.prev_t is None:
            return None
        return (vel-self.prev_vel)/(t-self.prev_t)
class PIDController(Controller):

    def __init__(self, params):
        
        """
        Parameters
        ----------
        params: dictionary
            Classic members: kp, kd, ki
        """
        super().__init__(params)
        assert("kp" in params)
        assert("ki" in params)
        assert("kd" in params)

        #Todo: configurer ça depuis simulation?
        self.constant_dt = 0.02

        self.kp = params["kp"]
        self.ki = params["ki"] 
        self.kd = params["kd"] 

        self.last_input = 0.0

        # Quand activé, on passe en  
        # proportionnal on measurement. Ca marche pas mal. 
        # http://brettbeauregard.com/blog/2017/06/introducing-proportional-on-measurement/
        #Parametres : 10 100 0.5 100 !
        self.proportionnal_on_measurement = False
        self.init_pos = None

        self.i_total = 0.0
        self.a = 0
    
    @property
    def i_max(self):
        return self.cmd_max/self.ki
    
    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        
        if self.init_pos is None:
            self.init_pos = measured_pos
        
        #On est a temps constant, pas besoin de prendre en compte la différence de temps ! 
        #Ca veut rien dire ptdr ref, id_ref = oneIn([ref_pos, ref_vel, ref_acc])
        
        ref = measured_pos

        acc = measured_vel-self.prev_vel if measured_vel is not None else None

        retval = 0.0

       
        err = (ref_pos - measured_pos)
        #Proportionnel
        if not self.proportionnal_on_measurement :
            p = self.kp * err
        else:
            p = -self.kp * (measured_pos - self.init_pos)
        #Intégral
        if self.ki != 0.0:
            self.i_total += err 
            if abs(self.i_total) > self.i_max:
                self.i_total = self.i_max if self.i_total > self.i_max else -self.i_max 
            i = self.ki * self.constant_dt * self.i_total
        else:
            i = 0.0
        #Dérivé
        if self.kd != 0.0:
            d = self.kd * self.constant_dt * (ref_vel - measured_vel) / self.constant_dt
            self.prev_err = err
        else:
            d = 0.0

        self.prev_vel = measured_vel

        retval = p+i+d
        if abs(retval) > self.cmd_max:
            retval = self.cmd_max if retval > self.cmd_max else -self.cmd_max

        return retval

class OpenLoopEffortController(Controller):
    """
    An open-loop controller which uses an effort proportional to acceleration
    """
    def __init__(self, params):
        super().__init__(params)
        self.mass = 1
        self.friction = 0.01
        assert("k_acc" in params)

        self.k_acc = params["k_acc"]
        

    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        if self.prev_t is None:
            self.prev_t = t 
            self.prev_acc = ref_acc
            return 0.0
        inertie = self.mass*self.prev_acc
        gravite = 0
        #frotement_visqueux = self.friction*ref_vel
        #frotement_sec = np.tan(0)*np.cos(0)*ref_vel
        frotement_visqueux = 0
        frotement_sec = 0
        u = inertie + gravite + frotement_visqueux + frotement_sec
        x = self.k_acc*u
        self.prev_t = t 
        self.prev_acc = ref_acc
        return x


        
class OpenLoopPendulumEffortController(Controller):
    """
    An open-loop controller which compensate gravity and acceleration for the
    Pendulum
    """
    def __init__(self, params):
        super().__init__(params)
        assert("mass" in params)    
        assert("dist" in params)

        self.size = params["mass"]
        self.mass = params["mass"]
        self.friction = 0.01


    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        if self.prev_t is None:
            self.prev_t = t 
            self.prev_acc = ref_acc
            self.prev_vel = ref_vel
            self.prev_pos = ref_pos
            return 0.0
        inertie = self.mass*self.prev_acc
        pos = self.prev_pos +self.prev_vel*(t-self.prev_t)
        gravite = self.mass*math.cos(pos)*self.size*(t-self.prev_t)*(-9.81)
        #frotement_visqueux = self.friction*prev_vel
        #frotement_sec = np.tan(0)*np.cos(0)*prev_vel
        frotement_visqueux = 0
        frotement_sec = 0
        u = inertie + gravite + frotement_visqueux + frotement_sec

        self.prev_t = t 
        self.prev_acc = ref_acc
        self.prev_vel = ref_vel
        self.prev_pos = ref_pos

        return u
        
class FeedForwardController(Controller):
    """
    A controller combining a PIDController and a model
    """
    def __init__(self, params):
        super().__init__(params)
        if params["model"]["type"]=="OpenLoopPendulumEffortController" :
            self.open_loop = OpenLoopPendulumEffortController(params["model"]["params"])
        else :
            self.open_loop = OpenLoopEffortController(params["model"]["params"])

        self.pid = PIDController(params["pid"]["params"])


    def step(self, t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        pid_val = self.pid.step(t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc)   
        open_loop_val = self.open_loop.step(t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc)   
        return pid_val + open_loop_val  

class RobotController():
    """
    A controller across multiple independent dimensions
    """
    def __init__(self, controllers):
        self.controllers = controllers

    def step(self,  t, measured_pos, measured_vel, ref_pos, ref_vel, ref_acc):
        N = len(self.controllers)
        cmd = np.zeros(N)
        if ref_vel is None:
            ref_vel = np.zeros(N)
        if ref_acc is None:
            ref_acc = np.zeros(N)
        for i in range(N):
            cmd[i] = self.controllers[i].step(t, measured_pos[i], measured_vel[i], ref_pos[i], ref_vel[i], ref_acc[i])
        return cmd
