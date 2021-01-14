#!/usr/bin/env python3

import numpy as np
import json
import math
import argparse
from abc import ABC, abstractmethod
import traceback
import math
import control as ctrl
import sys
"""
    Start : ça sert à quoi? Si on prend t < start, c'est censé donner quoi? et t > end?
    Demander si on doit partir d'une vélocité nulle ou pas ?
    Dans naturalCubicSpline : la vitesse au démarrage risque de ne pas être nulle??
"""

def buildTrajectory(type_name, start, knots, parameters = None):
    if type_name == "ConstantSpline":
        return ConstantSpline(knots, start)
    if type_name == "LinearSpline":
        return LinearSpline(knots, start)
    if type_name == "CubicZeroDerivativeSpline":
        return CubicZeroDerivativeSpline(knots, start)
    if type_name == "CubicWideStencilSpline":
        return CubicWideStencilSpline(knots, start)
    if type_name == "CubicCustomDerivativeSpline":
        return CubicCustomDerivativeSpline(knots, start)
    if type_name == "NaturalCubicSpline":
        return NaturalCubicSpline(knots, start)
    if type_name == "PeriodicCubicSpline":
        return PeriodicCubicSpline(knots, start)
    if type_name == "TrapezoidalVelocity":
        if parameters is None:
            raise RuntimeError("Parameters can't be None for TrapezoidalVelocity")
        return TrapezoidalVelocity(knots, parameters["vel_max"], parameters["acc_max"], start)
    raise RuntimeError("Unknown type: {:}".format(type_name))

def buildTrajectoryFromDictionary(dic):
    return buildTrajectory(dic["type_name"], dic["start"], np.array(dic["knots"]), dic.get("parameters"))

def buildRobotTrajectoryFromDictionary(dic):
    model = ctrl.getRobotModel(dic["model_name"])
    return RobotTrajectory(model, np.array(dic["targets"]), dic["trajectory_type"],
                           dic["target_space"],dic["planification_space"],
                           dic["start"], dic.get("parameters"))

class Trajectory:
    """
    Describe a one dimension trajectory. Provides access to the value
    for any degree of the derivative

    Parameters
    ----------
    start: float
        The time at which the trajectory starts
    end: float or None
        The time at which the trajectory ends, or None if the trajectory never
        ends
    """
    def __init__(self, start = 0):
        """
        The child class is responsible for setting the attribute end, if the
        trajectory is not periodic
        """
        self.start = start
        self.end = None

    @abstractmethod
    def getVal(self, t, d):
        """
        Computes the value of the derivative of order d at time t.

        Notes:
        - If $t$ is before (resp. after) the start (resp. after) of the
         trajectory, returns:
          - The position at the start (resp. end) of the trajectory if d=0
          - 0 for any other value of $d$

        Parameters
        ----------
        t : float
            The time at which the position is requested
        d : int >= 0
            Order of the derivative. 0 to access position, 1 for speed,
            2 for acc, etc...

        Returns
        -------
        x : float
            The value of derivative of degree d at time t.
        """

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end
    def __str__(self):
        return "Trajectory start:"+str(self.getStart())+",end:"+str(self.getEnd())+"\n"+str(self.knots)

class Spline(Trajectory):
    """
    Attributes
    ----------
    knots : np.ndarray shape (N,2+)
        The list of timing for all the N via points:
        - Column 0 represents time points
        - Column 1 represents the position
        - Additional columns might be used to specify other elements
          (e.g derivative)
    coeffs : np.ndarray shape(N-1,K+1)
        A list of n-1 polynomials of degree $K$
    """

    def __init__(self, knots, start = 0):
        super().__init__(start)
        knots = np.array(knots)
        times = knots[:, 0]
        # Zip va créer toutes les paires consécutives de la liste et all va vérifier que chaque paire
        # a son premier élément inférieur au second
        if not all(i < j for i, j in zip(times, times[1:])):
            raise RuntimeError('Invalid times')

        self.knots = knots
        self.end = self.knots[-1][0]
        self.start_index = max((i for i, x in enumerate(self.knots[:, 0]) if x <= start))
        self.updatePolynomials()

    @abstractmethod
    def updatePolynomials(self):
        """
        Updates the polynomials (self.coeffs) based on the knots and the
        interpolation method
        """

    def getDegree(self):
        """
        Returns
        -------
        d : int
            The degree of the polynomials used in this spline
        """
        return self.coeffs.shape[1] - 1

    @classmethod
    def derivative(cls, a, power, d=1):
        """
            :Exemple
            >>> Spline.derivative(2, 1, 1)
            (2, 0)
            >>> Spline.derivative(2, 2, 1)
            (4, 1)
            >>> Spline.derivative(2, 2, 2)
            (4, 0)
            >>> Spline.derivative(7, 6, 45)
            (0, 0)
        """
        if d > power:
            return (0, 0)

        if power < 0 or d < 0:
            raise ValueError("Impossible de dériver avec des trucs négatifs")

        if d == 0:
            return (a, power)

        a = (power * a, power - 1) if power > 0 else (0, 0)
        return cls.derivative(*a, d-1)

    def getVal(self, t, d = 0):
        if t <= self.start:
            if d == 0:
                # Tant qu'on est avant le start on la position sera égale à la position du point de départ (aucun mouvement)
                return self.knots[self.start_index][1]
            else:
                # Tant qu'on est avant le départ, la vitesse et l'accélération sont nulles
                return 0.0
        if not self.end is None and t >= self.end:
            if d == 0:
                return self.knots[-1][1]
            else:
                return 0.0

        if self.end is not None and t > self.end:
            # Après le dernier point, la position est égale à la position du dernier point
            # La vitesse et l'accélération sont nulles
            return self.knots[-1][1] if d == 0 else 0.0

        if not self.end is None and t > self.end:
            raise ValueError('Houlala le t il est pas bien', self.start, self.end, t)
        #gérer self.end
        real_t = t
        if self.end is None:
            real_t = t % self.knots[-1][0]
            index = max((i for i, x in enumerate(self.knots[:, 0]) if x <= real_t))
        elif t >= self.end:
            index = len(self.coeffs) - 1
        else:
            index = max((i for i, x in enumerate(self.knots[:, 0]) if x <= real_t))

        #index = max((i for i, x in enumerate(self.knots[:, 0]) if x <= t)) if t != self.end else len(self.coeffs) - 1
        poly = self.coeffs[index]
        #TODO: remplacer ça par un moyen propre de virer le ti quand il n'est pas nécessaire.
        # Les splines cubiques(k=4) n'ont pas besoin du ti
        ti = self.knots[index][0] if self.getDegree() < 3 else 0

        retval = 0.

        for j, elem in enumerate(poly):
            a, power = Spline.derivative(elem, j, d)
            retval += a * (real_t - ti)**power

        return retval

class ConstantSpline(Spline):

    def updatePolynomials(self):
        self.coeffs = np.array([[knot[1]] for knot in self.knots[:-1]])

class LinearSpline(Spline):

    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0] - 1, 2))
        for i in range(len(self.knots)-1):
            ti = self.knots[i][0]
            ti1 = self.knots[i+1][0]
            xi = self.knots[i][1]
            xi1 = self.knots[i+1][1]

            self.coeffs[i][0] = xi
            self.coeffs[i][1] = (xi1-xi)/(ti1-ti)

class CubicZeroDerivativeSpline(Spline):
    """
    Update polynomials ensuring derivative is 0 at every knot.
    """
    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0] - 1, 4))
        for i in range(len(self.knots)-1):
            ti = self.knots[i][0]
            ti1 = self.knots[i+1][0]
            xi = self.knots[i][1]
            xi1 = self.knots[i+1][1]

            A = np.array([
              [ti**3, ti**2, ti, 1],
              [ti1**3, ti1**2, ti1, 1],
              [3*ti**2, 2*ti, 1, 0],
              [3*ti1**2, 2*ti1, 1, 0]])
            B = np.array([xi, xi1, 0, 0])
            self.coeffs[i] = np.flip(np.linalg.inv(A) @ B)

class CubicWideStencilSpline(Spline):
    """
    Update polynomials based on a larger neighborhood following the method 1
    described in http://www.math.univ-metz.fr/~croisil/M1-0809/2.pdf
    """

    def __init__(self, knots, start = 0):
        super().__init__(knots, start)

        if len(knots) < 4 or len(knots) - self.start_index < 2 :
            raise RuntimeError('CubicWideStencilSpline requires at least 4 points')

    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0] - 1, 4))

        for i in range(1, len(self.knots)-2):
            tim1 = self.knots[i-1][0]
            ti = self.knots[i][0]
            ti1 = self.knots[i+1][0]
            ti2 = self.knots[i+2][0]

            xim1  =self.knots[i-1][1]
            xi = self.knots[i][1]
            xi1 = self.knots[i+1][1]
            xi2 = self.knots[i+2][1]
            A = np.array([
              [ti**3, ti**2, ti, 1],
              [ti1**3, ti1**2, ti1, 1],
              [tim1**3,tim1**2,tim1,1],
              [ti2**3,ti2**2,ti2,1]])
            B = np.array([xi, xi1, xim1, xi2])
            x = np.flip(np.linalg.inv(A) @ B)
            self.coeffs[i] = x
            if i == 1:
                self.coeffs[i-1] = x
            if i == len(self.knots) - 3:
                self.coeffs[i+1] = x

#The custom derivative will be specified
class CubicCustomDerivativeSpline(Spline):
    """
    For this splines, user is requested to specify the velocity at every knot.
    Therefore, knots is of shape (N,3)
    """
    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0] - 1, 4))
        for i in range(len(self.knots)-1):
            ti = self.knots[i][0]
            ti1 = self.knots[i+1][0]
            xi = self.knots[i][1]
            xi1 = self.knots[i+1][1]

            vi = self.knots[i][2]
            vi1 = self.knots[i+1][2]

            A = np.array([
              [ti**3, ti**2, ti, 1],
              [ti1**3, ti1**2, ti1, 1],
              [3*ti**2, 2*ti, 1, 0],
              [3*ti1**2, 2*ti1, 1, 0]])
            B = np.array([xi, xi1, vi, vi1])
            self.coeffs[i] = np.flip(np.linalg.inv(A) @ B)

class NaturalCubicSpline(Spline):

    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0] - 1, 4))
        n = len(self.knots) - 1
        m = np.zeros((4*n,n*4))
        B = np.zeros((m.shape[1],1))

        offset = n * 2
        test = []
        for i in range(n):
            ti = self.knots[i][0]
            ti1 = self.knots[i+1][0]
            xi = self.knots[i][1]
            xi1 = self.knots[i+1][1]
            m[2*i, i*4:i*4+4] = np.array([ti**3,ti**2,ti,1])
            m[2*i+1,i*4:i*4+4] = np.array([ti1**3,ti1**2,ti1,1])
            if(i < n-1):
                m[offset+2*i, i*4:i*4+8] = np.array([3*ti1**2,  2*ti1,  1,  0,  -3*ti1**2,  -2*ti1,  -1, 0])
                m[offset+2*i+1,i*4:i*4+8] = np.array([6*ti1,     2,      0,  0,  -6*ti1,      -2,     0,  0])

            B[2*i:2*i+2, 0] = np.array([xi, xi1])

        t0 = self.knots[0][0]
        td = self.knots[-1][0]
        #print(td)
        m[-2, 0:2] = np.array([6*t0,2])
        m[-1, -4:] = np.array([6*td,2,0,0])


        x = np.linalg.inv(m) @ B
        for i in range(n):
            self.coeffs[i] = [val for val in np.flip(x[4*i:4*i+4])]#charmant

# x0 x1 x2 -> (x0 x1) (x1 x2)

class PeriodicCubicSpline(Spline):
    def __init__(self, knots, start = 0):
        super().__init__(knots, start)
        self.end = None
    """
    Describe global splines where position, 1st order derivative and second
    derivative are always equal on both sides of a knot. This i
    """
    def updatePolynomials(self):
        self.coeffs = np.zeros((self.knots.shape[0] - 1, 4))
        n = len(self.knots) - 1
        m = np.zeros((4*n,n*4))
        B = np.zeros((m.shape[1],1))

        offset = n * 2
        test = []
        for i in range(n):
            ti = self.knots[i][0]
            ti1 = self.knots[i+1][0]
            xi = self.knots[i][1]
            xi1 = self.knots[i+1][1]
            m[2*i, i*4:i*4+4] = np.array([ti**3,ti**2,ti,1])
            m[2*i+1,i*4:i*4+4] = np.array([ti1**3,ti1**2,ti1,1])
            if(i < n-1):
                m[offset+2*i, i*4:i*4+8] = np.array([3*ti1**2,  2*ti1,  1,  0,  -3*ti1**2,  -2*ti1,  -1, 0])
                m[offset+2*i+1,i*4:i*4+8] = np.array([6*ti1,     2,      0,  0,  -6*ti1,      -2,     0,  0])

            B[2*i:2*i+2, 0] = np.array([xi, xi1])

        t0 = self.knots[0][0]
        td = self.knots[-1][0]
        m[-2, 0:4] = np.array([3*t0**2,  2*t0,  1,  0])
        m[-2, -4:] = np.array([-3*td**2,  -2*td,  -1, 0])
        m[-1, 0:4] = np.array([6*t0,2,0,0])
        m[-1, -4:] = np.array([-6*td,-2,0,0])

        x = np.linalg.inv(m) @ B
        for i in range(n):
            self.coeffs[i] = [val for val in np.flip(x[4*i:4*i+4])]#charmant

    def getEnd(self):
        return self.knots[-1][0] *10

class TrapezoidalVelocity(Trajectory):

    def getKnot(self,id):
        try:
            return self.knots[id][0]
        except:
            return self.knots[id]

    def __init__(self, knots, vMax, accMax, start):
        self.parameters = []
        self.accMax = float(accMax)
        self.vMax = float(vMax)
        self.knots = knots
        self.start = start
        elapsed = 0.0
        for i in range(len(knots)-1):
            D = self.getKnot(i+1)-self.getKnot(i)
            Dabs = float(abs(D)) #ヽ(◉◡◔)ﾉ
            ta = 0.0
            if Dabs > (vMax**2)/accMax:
                ta = vMax/accMax
            else:
                ta = math.sqrt(Dabs/accMax)
            Da = accMax*(ta**2)/2.0
            T = 2.0*ta+(Dabs-2*Da)/vMax
            self.parameters.append((elapsed, D, Dabs, ta, Da, T))
            elapsed += T
        self.end = float(elapsed)


    def getVal(self, t, d=0):
        if t < self.start:
            if d == 0:
                t = self.start
            else:
                return 0.0
        if t > self.end:
            if d == 0:
                t = self.end
            else:
                return 0
        index = 0
        #if t > self.end:
        #    return 0
        for i in range(len(self.parameters)):
            if t >= self.parameters[i][0]:
                index = i

        if index >= len(self.knots)-1:
            return 0
        #return index
        elapsed, D, Dabs,ta, Da, T = self.parameters[index]

        sign = 1 if D >= 0 else -1
        offset = elapsed
        t_local = t - offset
        xend = self.getKnot(index+1)
        xsrc = self.getKnot(index)
        #TODO: simplifier ça vvvv

        max_speed_reached = self.accMax * ta
        if d == 0:
            if t_local <= ta:
                return xsrc + sign*(self.accMax*(t_local**2)/2)
            elif t_local > T-ta:
                return xend - sign*(self.accMax*((T-t_local)**2)/2)
            else:
                return xsrc + sign*(Da+max_speed_reached*(t_local-ta))
        elif d == 1:
            if t_local <= ta:
                return sign*self.accMax*t_local
            elif t_local > T-ta:
                return sign*(max_speed_reached - self.accMax*(t_local-(T-ta)))
            else:
                return sign*max_speed_reached
        elif d == 2:
            if t_local <= ta:
                return sign*self.accMax
            elif t_local > T-ta:
                return -1* sign*self.accMax
            else:
                return 0
        elif d> 2:
            return 0
        else:
            raise Exception("Dérivée négative, c'est pas ouf.")

#TODO: tester synchronicité des trajectoires trapézoidales.
class RobotTrajectory:
    """
    Represents a multi-dimensional trajectory for a robot.

    Attributes
    ----------
    model : control.RobotModel
        The model used for the robot
    planification_space : str
        Two space in which trajectories are planified: 'operational' or 'joint'
    trajectories : list(Trajectory)
        One trajectory per dimension of the planification space
    start : float
        The beginning of the trajectory [s]
    end : float
        The end of the trajectory [s]
    """

    supported_spaces = ["operational", "joint"]

    def __init__(self, model, targets, trajectory_type,
                 target_space, planification_space,
                 start = 0, parameters = None):
        """
        model : ctrl.RobotModel
            The model of the robot concerned by this trajectory
        targets : np.ndarray shape(m,n) or shape(m,n+1)
            The multi-dimensional knots for the trajectories. One row concerns one
            target. Each column concern one of the dimension of the target space.
            For trajectories with specified time points (e.g. splines), the first
            column indicates time point.
            For trajectories with specified derivative, rows are used two by two :
            the first one is the time point, the second row is the derivative.
        target_space : str
            The space in which targets are provided: 'operational' or 'joint'
        trajectory_type : str
            The name of the class to be used with trajectory
        planification_space : str
            The space in which trajectories are defined: 'operational' or 'joint'
        start : float
            The start of the trajectories [s]
        parameters : dictionary or None
            A dictionary containing extra-parameters for trajectories
        """
        self.model = model

        self.target_space = target_space
        self.planification_space = planification_space

        self.correct_spaces = (planification_space in RobotTrajectory.supported_spaces) and (target_space in RobotTrajectory.supported_spaces)
        self.time_problem = False
        self.incorrect_points = []

        nb_joints = model.getNbJoints()

        self.times = []
        self.targets_split = []

        #Longueur attendue d'une cible si la cible ne contient que de la position
        #(3 si la cible est dans l'espace opérationnel, nb_joints dans l'espace articulaire)
        nb_dims = len(model.getOperationalDimensionNames()) if target_space == "operational" else nb_joints

        #Si il y a un nombre impair d'éléments, on considère que le premier est le temps
        offset = 0
        if len(targets[0])%nb_dims==1:
            offset = 1
            self.times = [target[0] for target in targets]

        #Si une cible de dimension 3 contient 7 éléments, on considère que :
        #1 élément est le temps
        #3 éléments sont la position
        #3 éléments sont la vitesse
        #(avec la position et la vitesse qui s'alternent)
        nb_subarray = int((len(targets[0])-offset)/nb_dims)
        #tableau de tableau [[positions], [velocites], [accelerations], [jerks], etc...]
        self.targets_split = []
        for subarray_index in range(nb_subarray):
            new_subarray = []
            for i in range(len(targets)):
                new_target = []
                new_target = targets[i][offset+subarray_index::nb_subarray]
                new_subarray.append(new_target)
            self.targets_split.append(new_subarray)

        print("Got following targets : ", self.targets_split, file=sys.stderr)
        print("Times:",self.times, file=sys.stderr)

        #Cibles converties dans l'espace correct
        self.target_correct_space =  [[] for i in range(nb_subarray)]
        #NOTE AVANT DE MODIFIER : NE PAS FAIRE [[]]*nb_subarray : ca copie le meme tableau, et ca mene a des comportements
        #bien nuls du style "j'append à un des tableaux et tous les tableaux se retrouve appended"

        #On convertit ou non les cibles dans l'espace des trajectoires
        if target_space == planification_space:
            self.target_correct_space = self.targets_split
        elif target_space == "operational":
            for i in range(len(self.targets_split[0])):
                target = self.targets_split[0][i]
                new_target = self.model.computeMGI(None,target,"analyticalMGI")#position
                self.target_correct_space[0].append(new_target)
                if new_target is None:
                    self.incorrect_points.append(i)
                elif nb_subarray > 1:
                    velocity = self.targets_split[1][i]
                    new_velocity = self.model.computeJacobian(new_target) @ velocity #velocité
                    self.target_correct_space[1].append(new_velocity)
        elif target_space == "joint":
            for i in range(len(self.targets_split[0])):
                target = self.targets_split[0][i]
                new_target =self.model.computeMGD(target)
                self.target_correct_space[0].append(new_target)
                if nb_subarray > 1:
                    velocity = self.targets_split[1][i]
                    new_velocity = np.linalg.inv(self.model.computeJacobian(target)) @ velocity #velocité

        self.check()

        #On construit les trajectoires
        self.trajectories = []
        self.trajectories = self.build_trajectories(
            self.target_correct_space,
            self.times,
            nb_dims,
            start,trajectory_type,parameters)

        self.end = self.trajectories[0].getEnd()
        for trajectory in self.trajectories:
            if self.end < trajectory.getEnd():
                self.end = trajectory.getEnd()

        self.start= 0
        for traj in self.trajectories:
            print(traj, file=sys.stderr)
        self.test = np.array([0.,0.,0.])
    def build_trajectories(self, target_split, times, nb_dim, start, trajectory_type, parameters):

        new_trajs =  []
        for i in range(nb_dim):
            knots = []
            for j in range(len(target_split[0])):
                knot = []
                if len(times)>0 :
                    knot = [times[j]]
                for subarray_id in range(len(target_split)):
                    knot.append(target_split[subarray_id][j][i])
                knots.append(knot)
            new_trajectory = buildTrajectory(trajectory_type,
                                            start,
                                            knots,
                                            parameters)
            new_trajs.append(new_trajectory)
        return new_trajs

    def check(self):
        #TODO:generate more precise error message
        if len(self.incorrect_points) > 0 or not self.correct_spaces or self.time_problem:
            raise ValueError("Incorrect points, problem in spaces or problem in time")
        return True

    def getVal(self, t, dim, degree, space):
        """
        Parameters
        ----------
        t : float
            The time at which the value is requested
        dim : int
            The dimension index
        degree : int
            The degree of the derivative requested (0 means position)
        space : str
            The space in which the value is requested: 'operational' or 'joint'

        Returns
        -------
        v : float
            The value of derivative of order degree at time t on dimension dim
            of the chosen space
        """

        retarray = None
        if degree == 0:
            if space == "joint":
                retarray = self.getJointTarget(t)
            elif space == "operational":
                retarray = self.getOperationalTarget(t)
        if degree == 1:
            if space == "joint":
                retarray = self.getJointVelocity(t)
            elif space == "operational":
                retarray = self.getOperationalVelocity(t)
        if degree == 2:
            if space == "joint":
                retarray = self.getJointAcceleration(t)
            elif space == "operational":
                retarray = self.getOperationalAcceleration(t)
        if retarray is None or dim >= len(retarray):
            return None
        return retarray[dim]

    def getOperationalTarget(self, t):
        """
        Returns:
        target : np.ndarray shape(N,)
            The target for the robot in operational space at time t
        """

        values = np.array([trajectory.getVal(t, 0) for trajectory in self.trajectories])
        if self.planification_space == "operational":
            return values
        else:
            x = self.model.computeMGD(values)
            if x is None:
                raise Exception("Couldn't compute MGD")
            return x

    def getJointTarget(self, t):
        """
        Returns:
        target : np.ndarray shape(N,)
            The target for the robot in joint space at time t
        """
        values = np.array([trajectory.getVal(t, 0) for trajectory in self.trajectories])
        if self.planification_space == "joint":
            return values
        else:
            #print("Computing mgi of :", values, file=sys.stderr)
            q = self.model.computeMGI(None, values, "analyticalMGI")
            #print("a",q,file=sys.stderr)
            if q is None:
                raise Exception("Couldn't compute MGI")
            return q
        return None

    def getOperationalVelocity(self, t):
        """
        Returns:
        target : np.ndarray shape(N,)
            The theoric velocity of the robot in operational space at time t
        """
        if self.planification_space == "operational":
            return np.array([trajectory.getVal(t, 1) for trajectory in self.trajectories])

        q = self.getJointTarget(t)
        jacobian = self.model.computeJacobian(q)

        dq = np.array([trajectory.getVal(t, 1) for trajectory in self.trajectories])
        return jacobian @ dq

    def getJointVelocity(self, t):
        """
        Returns:
        target : np.ndarray shape(N,)
            The theoric velocity if the robot in joint space at time t
        """
        if self.planification_space == "joint":
            return np.array([trajectory.getVal(t, 1) for trajectory in self.trajectories])

        q = self.getJointTarget(t)
        jacobian = self.model.computeJacobian(q)

        dx = np.array([trajectory.getVal(t, 1) for trajectory in self.trajectories])
        dq = np.linalg.inv(jacobian) @ dx
        return dq

    def getOperationalAcceleration(self, t):
        """
        Returns:
        target : np.ndarray shape(N,)
            The theoric velocity of the robot in operational space at time t
        """
        #\dot{x} = J @ \dot{q}
        #d/dt \dot{x}  = d/dt (J @ \dot{q})
        #\ddot{x} = \dot{J}\dot{q} + J\ddot{q}
        if self.planification_space == "operational":
            return np.array([trajectory.getVal(t, 2) for trajectory in self.trajectories])

        q = self.getJointTarget(t)
        dq = np.array([trajectory.getVal(t, 1) for trajectory in self.trajectories])
        ddq = np.array([trajectory.getVal(t, 2) for trajectory in self.trajectories])

        jacobian = self.model.computeJacobian(q)
        djacobian = self.model.computeDJacobian(q, dq)

        return jacobian @ ddq + djacobian @ dq

    def getJointAcceleration(self, t):
        """
        Returns:
        target : np.ndarray shape(N,)
            The theoric velocity if the robot in joint space at time t
        """

        if self.planification_space == "joint":
            return np.array([trajectory.getVal(t, 2) for trajectory in self.trajectories])
        #dddot{q} = pinv(J)(\ddot{x}-\dot{J}\dot{q})
        q = self.getJointTarget(t)
        dq = self.getJointVelocity(t)
        ddx = np.array([trajectory.getVal(t, 2) for trajectory in self.trajectories])

        jacobian = self.model.computeJacobian(q)
        djacobian = self.model.computeDJacobian(q, dq)
        ddq = np.linalg.pinv(jacobian) @ (ddx - djacobian @ dq)

        return ddq#(speed-pspeed)/0.02

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--robot", help="Consider robot trajectories and not 1D trajectories",
                        action="store_true")
    parser.add_argument("--degrees",
                        type=lambda s: np.array([int(item) for item in s.split(',')]),
                        default=[0,1,2],
                        help="The degrees of derivative to plot")
    parser.add_argument("trajectories", nargs="+", type=argparse.FileType('r'))
    args = parser.parse_args()
    trajectories = {}
    tmax = 0
    tmin = 10**10
    for t in args.trajectories:
        try:
            if args.robot:
                trajectories[t.name] = buildRobotTrajectoryFromDictionary(json.load(t))
            else:
                trajectories[t.name] = buildTrajectoryFromDictionary(json.load(t))
            tmax = max(tmax, trajectories[t.name].getEnd())
            tmin = min(tmin, trajectories[t.name].getStart())
        except KeyError as e:
            print("Error while building trajectory from file {:}:\n{:}".format(t.name, traceback.format_exc()))
            exit()
    order_names = ["position", "velocity", "acceleration", "jerk"]
    print("source,t,order,variable,value")
    for source_name, trajectory in trajectories.items():
        for t in np.arange(tmin - args.margin, tmax + args.margin, args.dt):
            for degree in args.degrees:
                order_name = order_names[degree]
                if (args.robot):
                    space_dims = {
                        "joint" : trajectory.model.getJointsNames(),
                        "operational" : trajectory.model.getOperationalDimensionNames()
                    }
                    for space, dim_names in space_dims.items():
                        for dim in range(len(dim_names)):

                            #print(dim_names, file=sys.stderr)
                            v = trajectory.getVal(t, dim, degree, space)
                            if v is not None:
                                print("{:},{:},{:},{:},{:}".format(source_name,t,order_name,dim_names[dim],v))
                            #else:
                            #    print("eh", space, file=sys.stderr)
                else:
                    v = trajectory.getVal(t,degree)
                    print("{:},{:},{:},{:},{:}".format(source_name,t,order_name,"x",v))
