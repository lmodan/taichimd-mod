import taichi as ti
import numpy as np
from .common import *
from .grid import NeighborList, NeighborTable

@ti.data_oriented
class ForceField(Module):
    is_conservative = False
    
    def calc_force(self):
        raise NotImplementedError


@ti.data_oriented
class ClassicalFF(ForceField):

    mixture = True
    #Modan: mixing rules?

    is_conservative = True

    MAX_ATOMTYPES = 256
    MAX_BONDTYPES = 256
    MAX_ANGLETYPES = 256
    MAX_DIHEDRALTYPES = 256
    MAX_IMPROPERTYPES = 256

    MAX_BONDS = 16
    MAX_ANGLES = 16
    MAX_DIHEDRALS = 16
    MAX_IMPROPERS = 16

    #def __init__(self, nonbond=None, bonded=None,
    #            bending=None, torsional=None,
    #            external=None):
    def __init__(self, masses=None, nonbond=None, bonds=None, angles=None, dihedrals=None, impropers=None, external=None):

        self.masses = masses
        self.nonbond = nonbond

        #Modan: reassigning the potential terms
        #self.bonded = bonded
        #self.bending = bending
        #self.torsional = torsional

        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.impropers = impropers

        # external potential is independent of atom type
        self.external = external

    #def set_params(self, nonbond=None, 
    #        bonded=None, bending=None, torsional=None):
    def set_params(self, masses=None, nonbond=None, bonds=None, angles=None, dihedrals=None, impropers=None):

        self.masses_params_d = masses
        self.nonbond_params_d = nonbond
        
        #Modan: reassigning the potential terms
        #self.bonded_params_d = bonded
        #self.bending_params_d = bending
        #self.torsional_params_d = torsional
        self.bonds_params_d = bonds
        self.angles_params_d = angles
        self.dihedrals_params_d = dihedrals
        self.impropers_params_d = impropers
        
    def register(self, system):
        #TODO: find out how to update topology on-the-fly
        # initialize data structures
        if ti.static(system.is_atomic and self.bonds != None):
            print("Warining: the simulation system does not have bonding definitions. Bonds/angles/dihedrals/impropers potentials will not be used.")

        if ti.static(self.nonbond != None):
            self.nonbond_params = ti.Vector.field(self.nonbond.n_params, dtype=ti.f32)
            ti.root.dense(ti.ij, (self.MAX_ATOMTYPES, self.MAX_ATOMTYPES)).place(self.nonbond_params)
        if ti.static(not system.is_atomic and self.bonds != None):
            self.bonds_np = []
            self.bond = ti.Vector.field(3, dtype=ti.i32) #a single bond as type, atom1, atom2
            self.bonds_params = ti.Vector.field(self.bonds.n_params, dtype=ti.f32)
            ti.root.dense(ti.i, self.MAX_BONDS 
                    * system.n_particles).place(self.bond) #1D array of [type, atom1, atom2] with MAX_BOND*nparticles dimensions
            ti.root.pointer(ti.i, self.MAX_BONDTYPES).place(self.bonds_params)
        if ti.static(not system.is_atomic and self.angles != None):
            self.angles_np = []
            self.angle = ti.Vector.field(4, dtype=ti.i32)
            self.angles_params = ti.Vector.field(self.angles.n_params, dtype=ti.f32)
            ti.root.dense(ti.i, self.MAX_ANGLES 
                    * system.n_particles).place(self.angle)
            ti.root.pointer(ti.i, self.MAX_ANGLETYPES).place(self.angles_params)
        if ti.static(not system.is_atomic and self.dihedrals != None):
            self.dihedrals_np = []
            self.dihedral = ti.Vector.field(5, dtype=ti.i32)
            self.dihedrals_params = ti.Vector.field(self.dihedrals.n_params, dtype=ti.f32)
            ti.root.dynamic(ti.i, self.MAX_DIHEDRALS 
                    * system.n_particles).place(self.dihedral)
            ti.root.pointer(ti.i, self.MAX_DIHEDRALTYPES).place(self.dihedrals_params)
        if ti.static(not system.is_atomic and self.impropers != None):
            self.impropers_np = []
            self.improper = ti.Vector.field(5, dtype=ti.i32)
            self.impropers_params = ti.Vector.field(self.impropers.n_params, dtype=ti.f32)
            ti.root.dynamic(ti.i, self.MAX_IMPROPERS 
                    * system.n_particles).place(self.impropers)
            ti.root.pointer(ti.i, self.MAX_IMPROPERTYPES).place(self.impropers_params)
        #inta is used to turn off 1-2 and 1-3 interactions, probably better to use floats if 1-4 interaction should be counted as 0.5 
        if ti.static(not system.is_atomic):
            self.intra = ti.field(dtype=ti.i32) # boolean
            ti.root.bitmasked(ti.ij, (system.n_particles, system.n_particles)).place(self.intra)
        return super().register(system)


    def populate_tables(self, i0, m, n):
        sys = self.system
        if ti.static(not sys.is_atomic and self.bonds != None):
            table = np.tile(np.array(m.bond), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.bonds_np.append(table.reshape(-1, 3))
        if ti.static(not sys.is_atomic and self.angles != None):
            table = np.tile(np.array(m.angle), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.angles_np.append(table.reshape(-1, 4))
        if ti.static(not sys.is_atomic and self.dihedrals != None):
            table = np.tile(np.array(m.dihedral), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.dihedrals_np.append(table.reshape(-1, 5))
        if ti.static(not sys.is_atomic and self.impropers != None):
            table = np.tile(np.array(m.improper), (n, 1, 1))
            table[:, :, 1:] += (i0 + np.arange(n) * m.natoms).reshape(-1, 1, 1)
            self.impropers_np.append(table.reshape(-1, 5))
        if ti.static(not sys.is_atomic):
            self.set_intra(i0, n, m.natoms, m.intra)

    
    def set_intra(self, i0, nmolec, natoms, imat):
        for i in range(nmolec):
            istart = i0 + i * natoms
            iend = istart + natoms
            for l in range(natoms):
                for m in range(natoms):
                    self.intra[istart + l, istart + m] = imat[l, m]


    def build(self):
        sys = self.system
        # set nonbond parameters
        if ti.static(self.nonbond != None):
            mass_np = []
            for aid in range(sys.type.shape[0]): # cannot enumerate directly with peratom types
                atyp = sys.type[aid]
                if not atyp in sys.masses_d:
                    mass = self.masses_params_d[atyp]
                    sys.masses_d[atyp] = mass #atomtypes start from 1
                mass_np.append(mass)
            sys.masses_np=np.array(mass_np)

            for k, v in self.nonbond_params_d.items():
                self.nonbond_params[k, k] = self.nonbond.fill_params(*v)  
                for k2, v2 in self.nonbond_params_d.items():
                    v_mix = self.nonbond.mix(v, v2)
                    self.nonbond_params[k, k2] = v_mix
                    self.nonbond_params[k2, k] = v_mix
        # build bonds table and set parameters
        if ti.static(not sys.is_atomic and self.bonds != None):
            for k, v in self.bonds_params_d.items():
                self.bonds_params[k] = self.bonds.fill_params(*v)
            self.bonds_np = np.vstack(self.bonds_np)
            self.bond.from_numpy(self.bonds_np) #self.bond as bond list
            self.nbonds = self.bonds_np.shape[0]
        # build angles table
        if ti.static(not sys.is_atomic and self.angles != None):
            for k, v in self.angles_params_d.items():
                self.angles_params[k] = self.angle.fill_params(*v)
            self.angles_np = np.vstack(self.angles_np)
            self.angle.from_numpy(self.angles_np) #self.angle as internally saved angles list
            # workaround
            self.nangles = self.angles_np.shape[0]
        # build dihedrals table
        if ti.static(not sys.is_atomic and self.dihedrals != None):
            for k, v in self.dihedrals_params_d.items():
                self.dihedrals_params[k] = self.dihedrals.fill_params(*v)
            self.dihedral.from_numpy(np.vstack(self.dihedrals_np)) #self.dihedral as dihedrals list
        # build impropers table
        if ti.static(not sys.is_atomic and self.impropers != None):
            for k, v in self.impropers_params_d.items():
                self.impropers_params[k] = self.impropers.fill_params(*v)
            self.improper.from_numpy(np.vstack(self.impropers_np)) #self.improper as impropers list

    @ti.func
    def force_nonbond(self, i, j):
        sys = self.system
        not_excl = True
        if ti.static(not sys.is_atomic):
            not_excl = self.intra[i, j] == 0
        if i < j and not_excl:
            d = sys.calc_distance(sys.position_unwrap[j], sys.position_unwrap[i])
            r2 = (d ** 2).sum()
            itype = sys.type[i]
            jtype = sys.type[j]
            
            imass = sys.masses[i]
            jmass = sys.masses[j]
                        
            params = self.nonbond_params[itype, jtype]
            uij = self.nonbond.energy(r2, params) #modded from __call__ in pair-wise interactions
            if uij != 0.0:
                sys.ep[None] += uij
                force = self.nonbond.force(d, r2, params) 
                # += performs atomic add
                sys.force[i] += force / imass
                sys.force[j] -= force / jmass
                if ti.static(sys.integrator.requires_hessian):
                    h = self.nonbond.hessian(d, r2, params)
                    sys.hessian[i, j] = h
                    sys.hessian[j, i] = h
                    sys.hessian[i, i] -= h
                    sys.hessian[j, j] -= h

    @ti.func
    def calc_force(self):
        sys = self.system
        sys.ep[None] = 0.0
        for i in sys.force:
            sys.force[i].fill(0.0)
            if ti.static(self.external != None):
                sys.ep[None] += self.external(sys.position_unwrap[i])
                sys.force[i] += self.external.force(sys.position_unwrap[i])
                if ti.static(sys.integrator.requires_hessian):
                    sys.hessian[i, i] += self.external.hessian(sys.position_unwrap[i])

        if ti.static(self.nonbond != None):
            if ti.static(not sys.grids == None and hasattr(sys, "n_neighbors")):
                ti.block_dim(64) #for loop decorator for parallelization
                for i in sys.n_neighbors:
                    for j in range(sys.n_neighbors[i]):
                        self.force_nonbond(i, sys.neighbors[i, j])
            elif ti.static(not sys.grids == None and hasattr(sys, "neighbors")):
                for i, j in sys.neighbors:
                    self.force_nonbond(i, j)
            else:
                for i, j in ti.ndrange(sys.n_particles, sys.n_particles):
                    self.force_nonbond(i, j)
 
        if ti.static(not sys.is_atomic and self.bonds != None):
            for x in range(self.nbonds):
                bondtype, i, j = self.bond[x][0], self.bond[x][1], self.bond[x][2]
                imass = sys.masses[i]
                jmass = sys.masses[j]
                params = self.bonds_params[bondtype]
                d = sys.calc_distance(sys.position_unwrap[j], sys.position_unwrap[i])
                if ti.static(self.bonds == None):
                    raise NotImplementedError("Rigid bonds are not implemented yet!") 
                else:
                    #harmonic bonds
                    r2 = (d ** 2).sum()
                    sys.ep[None] += self.bonds.energy(r2, params)#modded from __call__ in pair-wise interactions
                    force = self.bonds.force(d, r2, params)
                    sys.force[i] += force / imass
                    sys.force[j] -= force / jmass
                    if ti.static(sys.integrator.requires_hessian):
                        h = self.bonds.hessian(d, r2, params)
                        sys.hessian[i, j] = h
                        sys.hessian[j, i] = h
                        sys.hessian[i, i] -= h
                        sys.hessian[j, j] -= h

        if ti.static(not sys.is_atomic and self.angles != None):
            if ti.static(sys.integrator.requires_hessian):
                raise NotImplementedError("Hessian not supported for angle potentials!")
            for x in range(self.nangles):
                angletype, i, j, k = self.angle[x][0], self.angle[x][1], self.angle[x][2], self.angle[x][3]
                imass = sys.masses[i]
                jmass = sys.masses[j]
                kmass = sys.masses[k]
                params = self.angles_params[bendtype]
                v1 = sys.calc_distance(sys.position_unwrap[i], sys.position_unwrap[j])
                v2 = sys.calc_distance(sys.position_unwrap[k], sys.position_unwrap[j])
                if ti.static(self.angles == None):
                    raise NotImplementedError("Rigid angles are not implemented yet!") 
                else:
                    l1 = v1.norm()
                    l2 = v2.norm()
                    d = v1.dot(v2)
                    cosx = d / (l1 * l2)
                    #print(bendtype, cosx, ti.acos(cosx), params[1], self.bending(cosx, params))

                    #TODO: what is this angle definition?
                    sys.ep[None] += self.angles(cosx, params)
                    d_cosx = self.angles.derivative(cosx, params)
                    u = 1 / l1 / l2
                    f1 = (v2 - d / l1 ** 2 * v1) * u
                    f2 = (v1 - d / l2 ** 2 * v2) * u
                    sys.force[i] -= f1 * d_cosx / imass
                    sys.force[k] -= f2 * d_cosx / jmass
                    sys.force[j] += (f1 + f2) * d_cosx / kmass

        # TODO: implementation of dihedrals and impropers

                        
