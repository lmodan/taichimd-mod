import os, sys
import taichi as ti
sys.path.append(os.getcwd())
from taichimd import COLOR_MOLECULES
import numpy as np

from taichimd.system import MolecularDynamics
from taichimd.interaction import *
from taichimd.molecule import Molecule
from taichimd.forcefield import ClassicalFF
from taichimd.ui import *

from taichimd.integrator import NVTIntegrator

nchain = 10
boxlength = 160
dt=0.0005
l0 = 1.54
lchain = 100
pos = np.zeros((nchain, lchain, DIM))
pos[:, :, 0] = np.arange(lchain) * l0
for i in range(nchain):
    pos[i, :, 1] += boxlength / 1.2 / nchain * i
pos = pos.reshape(nchain * lchain, DIM)
# center the molecule
pos -= np.mean(pos, axis=0)
temperature = 573.0
# intramolecular exclusion matrix
# exclude intramolecular interaction when
# two atoms are less than 4 bonds apart
adj = np.eye(lchain, k=1) + np.eye(lchain, k=-1)
intra = (adj + adj@adj + adj@adj@adj).astype(bool)

gui=True

ti.init(arch=ti.cuda)

mol = Molecule([1] * lchain, 
    bond=[[1, i, i+1] for i in range(lchain - 1)],
    intra=intra)
ff = ClassicalFF(nonbond=LennardJones(rcut=14), bonds=HarmonicBond())
ff.set_params(nonbond={1:[3.95,46.0]}, bonds={1:[96500.0/2, l0]})
md = MolecularDynamics({mol: nchain}, boxlength, dt, ff, NVTIntegrator, temperature=temperature, renderer=MDRenderer if gui else None)
md.read_restart(pos, centered=True)
md.randomize_velocity(keep_molecules=False)
md.run(irender=64)