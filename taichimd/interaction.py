import taichi as ti
from .consts import *

class HessianNotImplemented(NotImplementedError):
    pass

class ForceNotImplemented(NotImplementedError):
    pass

@ti.data_oriented
class Interaction:

    def fill_params(self, *args):
        return list(args)

@ti.data_oriented
class ExternalPotential(Interaction):

    def energy(self, r):
        raise NotImplementedError

    def force(self, r):
        raise ForceNotImplemented

    def hessian(self, r):
        raise HessianNotImplemented

@ti.data_oriented
class QuadraticWell(ExternalPotential):

    def __init__(self, k, center):
        self.k = k
        self.center = ti.Vector(center)

    @ti.func
    def energy(self, r):
        return self.k * ((r - self.center) ** 2).sum() / 2

    @ti.func
    def force(self, r):
        return -self.k * (r - self.center)

    @ti.func
    def hessian(self, r):
        return self.k * IDENTITY

@ti.data_oriented
class InverseSquare(ExternalPotential):

    def __init__(self, k, center):
        self.k = k
        self.center = center

    @ti.func
    def energy(self, r):
        return self.k / (r - self.center).norm()

    @ti.func
    def force(self, r):
        dr = r - self.center
        dr2 = (dr ** 2).sum()
        return -self.k * dr / dr.norm() / dr2


'''
Pair interactions
'''
@ti.data_oriented
class PairInteraction(Interaction):

 
    def energy(self, r2, args):
        raise NotImplementedError

    def derivative(self, r2, args):
        raise ForceNotImplemented

    def second_derivative(self, r2, args):
        raise HessianNotImplemented

    @ti.func
    def force(self, r, r2, args):
        return 2. * self.derivative(r2, args) * r
    
    @ti.func
    def hessian(self, r, r2, args):
        return -4. * r.outer_product(r) * self.second_derivative(r2, args) \
            - 2 * IDENTITY * self.derivative(r2, args)


@ti.data_oriented
class LennardJones(PairInteraction):
    
    n_params = 3

    def __init__(self, rcut=0):
        self.rcut = rcut
        self.rc2 = rcut ** 2
        self.irc6 = 1 / rcut ** 6
        self.irc12 = self.irc6 ** 2
   
    @ti.func
    def energy(self, r2, args):
        u = 0.0
        if ti.static(self.rcut <= 0) or 0 < r2 < self.rc2:
            s12, s6, e = args[0], args[1], args[2]
            u = 4. * e * (s12 * (1 / r2 ** 6 - self.irc12)
                - s6 * (1 / r2 ** 3 - self.irc6))
        return u

    @ti.func
    def derivative(self, r2, args):
        s12, s6, e = args[0], args[1], args[2]
        return 12. * e / r2 \
            * (-2. * s12 / r2 ** 6 + s6 / r2 ** 3) 
    @ti.func
    def second_derivative(self, r2, args):
        s12, s6, e = args[0], args[1], args[2]
        return 24. * e / r2 ** 2 \
            * (7. * s12 / r2 ** 6 - 2. * s6 / r2 ** 3)

    def fill_params(self, sigma, epsilon):
        s6 = sigma ** 6
        s12 = s6 ** 2
        return [s12, s6, epsilon]

    def mix(self, v1, v2, rule="arithmetic"): #mixing rule
        s_i, e_i = v1[0], v1[1]
        s_j, e_j = v2[0], v2[1]
        if rule=="geometric":
            return self.fill_params( ti.sqrt(s_i * s_j), ti.sqrt(e_i * e_j))
        elif rule=="arithmetic":
            return self.fill_params( (s_i + s_j) / 2., ti.sqrt(e_i * e_j))
        else:
            raise NotImplementedError

@ti.data_oriented
class Coulomb(PairInteraction):

    n_params = 1
    
    
    @ti.func
    def energy(self, r2, args):
        k = args[0]
        return k / ti.sqrt(r2)

    @ti.func
    def derivative(self, r2, args):
        k = args[0]
        r = ti.sqrt(r2)
        return -k / (2 * r * r2)

    @ti.func
    def second_derivative(self, r2, args):
        k = args[0]
        r = ti.sqrt(r2)
        return 3. * k / (4 * r * r2 * r2)


@ti.data_oriented
class HarmonicBond(PairInteraction):

    n_params = 2

    @ti.func
    def energy(self, r2, args):
        k, r0 = args[0], args[1]
        r = ti.sqrt(r2)
        return k * (r - r0) ** 2 / 2

    @ti.func
    def derivative(self, r2, args):
        k, r0 = args[0], args[1]
        return k * (1 - r0 / ti.sqrt(r2)) / 2

    @ti.func
    def second_derivative(self, r2, args):
        k, r0 = args[0], args[1]
        r = ti.sqrt(r2)
        return k * r0 / (2 * r * r2)

#TODO: what is this for a pair? k * r ** 2, for each atom?
@ti.data_oriented
class ParabolicPotential(PairInteraction):

    n_params = 1


    @ti.func
    def energy(self, r2, args):
        k = args[0]
        r = ti.sqrt(r2)
        return k * r ** 2 / 2

    @ti.func
    def derivative(self, r2, args):
        k = args[0]
        return k / 2

    @ti.func
    def second_derivative(self, r2, args):
        return 0.

'''
harmonic bond bending
true harmonic: e = k * (acos(x) - theta0) ** 2
second order approx: e = (x - x0) ** 2 / (1 - x0 ** 2), x0 = cos(theta0)
third order approx: e = (x - x0) ** 2 / (1 - x0 ** 2) + x0 ** 2 * (x - x0) ** 3 / 
(1 - x0 ** 2) ** 2

calculate force:
x = cos<r1-r0, r2-r0> = (r1 - r0)^T.(r2 - r0) / (|r1 - r0|*|r2 - r0|)  

d|r|/dr = d(sqrt(r^T.r))/dr = e[r] = r / |r|

let v = (r1 - r0)^T.(r2 - r0) = (r1^T.r2 - r0^T.(r1 + r2) + r0^2)

du/dr_1 = (r2 - r0)
du/dr_2 = (r1 - r0)
du/dr_0 = (2r0 - r1 - r2)

let u = |r1 - r0|*|r2 - r0|

dv/dr_1 = |r2 - r0|/|r1 - r0| * (r1 - r0)
dv/dr_2 = |r1 - r0|/|r2 - r0| * (r2 - r0)
dv/dr_0 = -|r1 - r0|/|r2 - r0| * (r2 - r0) - |r2 - r0|/|r1 - r0| * (r1 - r0)

dx = vdu - udv / u**2

dx/dr_1 = [|r1 - r0|*|r2 - r0|*(r2 - r0) - (r1 - r0)^T.(r2 - r0)*|r2 - r0|/|r1 - r0| * (r1 - r0)] / u**2
        = [(r2 - r0) - (r1 - r0)^T.(r2 - r0) / (r1 - r0) ** 2 * (r1 - r0)]/u

dx/dr_2 is the same

dx/dr_0 = -dx/dr_1-dx/dr_2

'''

class AngleInteraction(Interaction):

    '''
    Bond bending potentials operate on the angle cosine
    between two bond vectors [r1<--r0-->r2],
    U = f(cosx, args)
    f* = dU/dr* = f'(cosx) * d(cosx)/dr*
    '''

    @ti.func
    def energy(self, cosx, args):
        raise NotImplementedError
    @ti.func
    def derivative(self, cosx, args):
        raise NotImplementedError


class HarmonicAngle(AngleInteraction):

    n_params = 2

    @ti.func
    def energy(self, cosx, args):
        k, theta0 = args[0], args[1]
        return 1/2. * k * (ti.acos(cosx) - theta0) ** 2
    
    @ti.func
    def derivative(self, cosx, args):
        k, theta0 = args[0], args[1]
        return - k * (ti.acos(cosx) - theta0) / ti.sqrt(1 - cosx ** 2)

#TODO: dihedrals and impropers

