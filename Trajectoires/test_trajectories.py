import inspect
import random
import sys
import unittest

import numpy as np

from trajectories import *

atol = 1e-06

def sign(nb):
    if nb < 0-atol:
        return -1
    if nb > 0+atol:
        return 1
    if -atol < nb < atol:
        return 0

def get_non_continuous_points(traj, start, end, d):
    non_continuous_points = 0

    for t in np.arange(start + 0.2, end + 0.2, 0.2):
        xi = traj.getVal(t, d)
        xi1 = traj.getVal(t + 0.2, d)

        diff = abs(xi - xi1)
        if diff > 1:
            non_continuous_points += 1
    
    return non_continuous_points




class TestTrajectory(unittest.TestCase):

    def test_getVal(self):
        knots = np.array([[0, 2], [1, 4], [2, 3], [4, -1]])
        traj = LinearSpline(knots, 1.4)

        # La vitesse et l'accélération doivent être nulles avant le start et après le end
        for d in [1, 2]:
            self.assertEqual(0.0, traj.getVal(1.38, d))
            self.assertEqual(0.0, traj.getVal(4.05, d))


class TestConstantSpline(unittest.TestCase):

    def setUp(self):
        self.knots = np.array([[0, 2], [1, 4], [3, 8]])

    def test_getDegree(self):
        traj = ConstantSpline(self.knots, 0)
        self.assertEqual(0, traj.getDegree())

    def test_position(self):
        traj = ConstantSpline(self.knots, 0)
        for knot in self.knots:
            self.assertEqual(knot[1], traj.getVal(knot[0]))

    def test_getVal(self):
        traj = ConstantSpline(self.knots, 0.0)

        for i in range(len(self.knots) - 1):
            expected = self.knots[i][1]
            a = self.knots[i][0]
            b = self.knots[i + 1][0]

            # On génère 1000 ti aléatoirements
            for j in range(1000):
                t = random.uniform(a, b - 0.01)
                # Pour tout ti compris entre deux points (t0 et t1), xi doit être égal à x0
                self.assertEqual(expected, traj.getVal(t))

                # La vitesse et l'accélération doivent être nulles
                self.assertEqual(0.0, traj.getVal(t, 1))
                self.assertEqual(0.0, traj.getVal(t, 2))


class TestLinearSpline(unittest.TestCase):

    def setUp(self):
        self.knots = np.array([[0, 2], [1, 4], [2, 3], [4, -1]])
        self.traj = LinearSpline(self.knots)

    def test_getDegree(self):
        traj = LinearSpline(self.knots)
        self.assertEqual(1, traj.getDegree())

    def test_getVal(self):
        traj = LinearSpline(self.knots)

        for i in range(len(self.knots) - 1):
            a = self.knots[i][0]
            b = self.knots[i + 1][0]

            # On génère 1000 ti aléatoirements
            for j in range(1000):
                t1 = random.uniform(a, b - 0.01)
                t2 = random.uniform(a, b - 0.01)

                if t1 != t2:
                    # Pour ti et tj différents, on doit avoir deux valeurs différentes
                    self.assertNotEqual(traj.getVal(t1), traj.getVal(t2))

                # L'accélération doit être nulle
                self.assertEqual(0.0, traj.getVal(t1, 2))

    def test_continuous(self):
        start = self.knots[0][0]
        end = self.knots[-1][0]

        # Position continue
        result = get_non_continuous_points(self.traj, start, end, 0)
        self.assertEqual(result, 0)

        # Vitesse non continue
        result = get_non_continuous_points(self.traj, start, end, 1)
        self.assertGreater(result, 0)


class TestCubicZeroDerivativeSpline(unittest.TestCase):

    def test_getDegree(self):
        knots = np.array([[0, 2], [1, 3], [3, 7],[7, 0]])
        start = 0
        spline = CubicZeroDerivativeSpline(knots, start)
        self.assertEqual(3, spline.getDegree())

    def test_zero_speed(self):
        knots = np.array([[0, 2], [1, 3], [3, 7],[7, 0]])
        start = 0
        spline = CubicZeroDerivativeSpline(knots, start)
        for i in range(len(knots)):
            self.assertTrue(-atol < spline.getVal(knots[i][0], 1) < atol )

    def test_infinite_acceleration(self):
        knots = np.array([[0, 2], [1, 3], [3, 7],[7, 0]])
        start = 0
        spline = CubicZeroDerivativeSpline(knots, start)
        for i in range(len(knots)):
            before = spline.getVal(knots[i][0]-0.02, 2)
            after = spline.getVal(knots[i][0]+0.02, 2)
            self.assertTrue(abs(before-after) > 2)

    def test_continuous(self):
        knots = np.array([[0, 2], [1, 3], [3, 7],[7, 0]])
        start = 0.5
        end = 7
        spline = CubicZeroDerivativeSpline(knots, start)

        # Position continue
        result = get_non_continuous_points(spline, start, end, 0)
        self.assertEqual(result, 0)
        self.assertEqual(2, spline.getVal(-50, ))

        # Vitesse continue
        result = get_non_continuous_points(spline, start, end, 1)
        self.assertEqual(result, 0)

        # Accélération non continue
        result = get_non_continuous_points(spline, start, end, 2)
        self.assertGreaterEqual(result, 0)


class TestCubicWideStencilSpline(unittest.TestCase):

    def setUp(self):
        self.knots = np.array([[0, 2], [1, 3], [3, 7],[7, 0]])

    def test_getDegree(self):
        spline = CubicWideStencilSpline(self.knots, 0)
        self.assertEqual(3, spline.getDegree())

    def test_continuous(self):
        start = 0
        end = 7
        spline = CubicZeroDerivativeSpline(self.knots, start)

        # Position continue
        result = get_non_continuous_points(spline, start, end, 0)
        self.assertEqual(0, result)

        # Vitesse continue
        result = get_non_continuous_points(spline, start, end, 0)
        self.assertEqual(0, result)

        # Accélération ???


class AutoTest(unittest.TestCase):

    def setUp(self):
        self.t_range = (0, 10)
        self.knots_range = (2, 8)

    def test_random_knots(self):
        # PeriodicCubicSpline pas au point
        splines_class = [ConstantSpline, LinearSpline, CubicZeroDerivativeSpline, NaturalCubicSpline, CubicWideStencilSpline]

        for k in range(900):
            # Génération de n knots, nombre compris dans les bornes de knots_range
            knots_number = random.randint(*self.knots_range)
            knots = np.zeros((knots_number, 2))

            # Génération des knots, les t sont bornés et uniques.
            times = sorted(random.sample(range(*self.t_range), knots_number))

            # Valeurs associées aux t entre -10 et 10 (arbitraire)
            values = random.choices(range(-10, 10), k=knots_number)

            # Sélection d'un start aléatoire placé sur un point de contrôle
            start_index = random.choice(range(0, knots_number))
            start = times[start_index]

            for i in range(knots_number):
                knots[i] = np.array([times[i], values[i]])
            
            print(knots)
            # Pour chaque spline présente dans la liste, on va regarder si la trajectoire passe bien par
            # les knots au bon moment
            for spline_class in splines_class:
                if spline_class == CubicWideStencilSpline and (len(knots) < 4 or len(knots) - start_index < 2):
                    continue
                print(spline_class, start)

                traj = spline_class(knots, start=start)

                for knot in knots[start_index:]:
                    value = traj.getVal(knot[0])

                    print('expected %s, got %s for t=%s' % (knot[1], value, knot[0]))
                    self.assertTrue(np.isclose(knot[1], value))


if __name__ == '__main__':
    import doctest

    trajectories_module = __import__('trajectories')
    test_suite = doctest.DocTestSuite(trajectories_module)
    unittest.TextTestRunner().run(test_suite)
    unittest.main()
