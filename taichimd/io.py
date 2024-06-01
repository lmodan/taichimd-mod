import taichi as ti
from .consts import *
from .common import *
import math
import numpy as np

#IO module compatible with LAMMPSdata and LAMMPStrj

@ti.data_oriented
class IO:

    #dataformats = ["LMPDAT"]#["LMPDAT", "LMPTRJ", "PDB"]
    def __init__(self, system, forcefield, dataformat="LMPDAT"):
        self.supported_formats = ["LMPDAT", "LMPTRJ"]
        self.dataformat = dataformat
        self.system = system
        self.forcefield = forcefield

@ti.data_oriented
class LMPdata(IO):
    def __init__(self, system, forcefield, dataformat="LMPDAT", triclinic=False):

        self.comment = "Created with TaichiMD"
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.masses = {}
        self.box = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.natoms = 0
        self.nbonds = 0
        self.nangles = 0
        self.ndihedrals = 0
        self.nimpropers = 0
        self.istriclinic = triclinic
        self.system = system
        self.forcefield = forcefield
        self.dataformat = dataformat

    def readfromFields(self):
        Lx, Ly, Lz = self.system.boxlength
        self.box = [[-Lx/2., -Ly/2., -Lz/2.],[Lx/2., Ly/2., Lz/2.], [0.0, 0.0, 0.0]]
        positions_unwrap = self.system.position_unwrap.to_numpy()
        for idx, atom in enumerate(positions_unwrap):
            if DIM == 2: #DIM in consts
                z = 0.0
            else:
                z = atom[2]
            x, y = atom[:2]
            molID = self.system.molIDperatom[idx]
            #molID from "composition.keys()", atom type from forcefield params in system.forcefield
            #mol, type, q, x, y, z
            typ = self.system.type[idx]
            q = 0.0 #hardcoded for CG simulations
            self.atoms[idx+1] = [molID, typ, q, x, y, z]

        if hasattr(self.forcefield, "bonds_np"):
            for idx, bond in enumerate(self.forcefield.bonds_np):
                typ, a1, a2 = bond
                self.bonds[idx+1] = [typ, a1+1, a2+1]
        if hasattr(self.forcefield, "angles_np"):
            for idx, angle in enumerate(self.forcefield.angles_np):
                typ, a1, a2, a3 = angle
                self.angles[idx+1] = [typ, a1+1, a2+1, a3+1]
        if hasattr(self.forcefield, "dihedrals_np"):
            for idx, dihedral in enumerate(self.forcefield.dihedrals_np):
                typ, a1, a2, a3, a4 = dihedral
                self.dihedrals[idx+1] = [typ, a1, a2, a3, a4]
        if hasattr(self.forcefield, "impropers_np"):
            for idx, improper in enumerate(self.forcefield.impropers_np):
                typ, a1, a2, a3, a4 = improper
                self.impropers[idx+1] = [typ, a1, a2, a3, a4]
            
    def validateData(self):
        self.natoms = len(self.atoms)
        self.nbonds = len(self.bonds)
        self.nangles = len(self.angles)
        self.ndihedrals = len(self.dihedrals)
        self.nimpropers = len(self.impropers)
        if self.natoms:
            self.natomtypes = np.max([atom[1] for i,(k,atom) in enumerate(self.atoms.items())])
        if self.nbonds:
            self.nbondtypes = np.max([atom[0] for i,(k,atom) in enumerate(self.bonds.items())])
        else:
            self.nbondtypes = 0
        if self.nangles:
            self.nangletypes = np.max([atom[0] for i,(k,atom) in enumerate(self.angles.items())])
        else:
            self.nangletypes = 0
        if self.ndihedrals:
            self.ndihedraltypes = np.max([atom[0] for i,(k,atom) in enumerate(self.dihedrals.items())])
        else:
            self.ndihedraltypes = 0
        if self.nimpropers:
            self.nimpropertypes = np.max([atom[0] for i,(k,atom) in enumerate(self.impropers.items())])
        else:
            self.nimpropertypes = 0
        for atom in range(self.natomtypes):
            atom += 1 #start from type 1 rather than 0
            if not atom in self.masses:
                self.masses[atom] = 1.0
                
        if not all(tilt == 0 for tilt in self.box[2]):
            self.istriclinic = True
    def writeLMPDAT(self, output_data="data.lammpsdat"):
        self.readfromFields()
        self.validateData()
        with open(output_data, 'w') as of:
            of.write("{}\n\n".format(self.comment))
            maxnum = max([self.natoms, self.nbonds, self.nangles,
                          self.ndihedrals, self.nimpropers])
            mw = int(math.log(maxnum, 10) + 1)
            of.write("{:{mw}d} atoms\n".format(self.natoms, mw=mw))
            of.write("{:{mw}d} bonds\n".format(self.nbonds, mw=mw))
            of.write("{:{mw}d} angles\n".format(self.nangles, mw=mw))
            of.write("{:{mw}d} dihedrals\n".format(self.ndihedrals, mw=mw))
            of.write("{:{mw}d} impropers\n\n".format(self.nimpropers, mw=mw))
            maxnum = max([self.natomtypes, self.nbondtypes, self.nangletypes,
                          self.ndihedraltypes, self.nimpropertypes])
            mw = int(math.log(maxnum, 10) + 1)
            of.write("{:{mw}d} atom types\n".format(self.natomtypes, mw=mw))
            of.write("{:{mw}d} bond types\n".format(self.nbondtypes, mw=mw))
            of.write("{:{mw}d} angle types\n".format(self.nangletypes, mw=mw))
            of.write("{:{mw}d} dihedral types\n".format(self.ndihedraltypes, mw=mw))
            of.write("{:{mw}d} improper types\n\n".format(self.nimpropertypes, mw=mw))
            of.write("{:.5f}  {:.5f} xlo xhi\n".format(self.box[0][0], self.box[1][0]))
            of.write("{:.5f}  {:.5f} ylo yhi\n".format(self.box[0][1], self.box[1][1]))
            of.write("{:.5f}  {:.5f} zlo zhi\n".format(self.box[0][2], self.box[1][2]))
            
            if self.istriclinic:
                of.write("{:.5f}  {:.5f}  {:.5f} xy xz yz\n\n".format(self.box[2][0], self.box[2][1], self.box[2][2]))
            of.write("\nMasses\n\n")
            mw = int(math.log(self.natomtypes, 10) + 1)
            ks=sorted(self.masses)
            for i, k in enumerate(ks):
                m=self.masses[k]
                of.write("{:{mw}d} {:7.4f}\n".format(k,m,mw=mw))
            of.write("\nAtoms\n\n")
            wid = int(math.log(self.natoms, 10) + 1)
            wmol = int(math.log(max([a[0] for i,(k,a) in enumerate(self.atoms.items())]), 10) + 1)
            wtype = int(math.log(max([a[1] for i,(k,a) in enumerate(self.atoms.items())]), 10) + 1)
            ks=sorted(self.atoms)
            for i, k in enumerate(ks):
                a=self.atoms[k]
                of.write("{:{wid}d} {:{wmol}d} {:{wtype}d} {:=9.6f} {:=9.6f} {:=9.6f} {:=9.6f}\n".format(k, *a,
                         wid=wid, wmol=wmol, wtype=wtype))
                         
            if self.nbonds:
                of.write("\nBonds\n\n")
                wid = int(math.log(self.nbonds, 10) + 1)
                wtype = int(math.log(self.nbondtypes, 10) + 1)
                wb = int(math.log(self.natoms, 10) + 1)
                ks=sorted(self.bonds)
                for i, k in enumerate(ks):
                    b=self.bonds[k]
                    of.write("{:{wid}d} {:{wtype}d} {:{wb}d} {:{wb}d}\n".format(k, *b,
                             wid=wid, wtype=wtype, wb=wb))
                             
            if self.nangles:
                of.write("\nAngles\n\n")
                wid = int(math.log(self.nangles, 10) + 1)
                wtype = int(math.log(self.nangletypes, 10) + 1)
                ks=sorted(self.angles)
                for i, k in enumerate(ks):
                    a=self.angles[k]
                    of.write("{:{wid}d} {:{wtype}d} {:{wb}d} {:{wb}d} {:{wb}d}\n".format(k, *a,
                             wid=wid, wtype=wtype, wb=wb))

            if self.ndihedrals:
                of.write("\nDihedrals\n\n")
                wid = int(math.log(self.ndihedrals, 10) + 1)
                wtype = int(math.log(self.ndihedraltypes, 10) + 1)
                ks=sorted(self.dihedrals)
                for i, k in enumerate(ks):
                    d=self.dihedrals[k]
                    of.write("{:{wid}d} {:{wtype}d} {:{wb}d} {:{wb}d} {:{wb}d} {:{wb}d}\n".format(k, *d,
                             wid=wid, wtype=wtype, wb=wb))
            if self.nimpropers:
                of.write("\nImpropers\n\n")
                wid = int(math.log(self.nimpropers, 10) + 1)
                wtype = int(math.log(self.nimpropertypes, 10) + 1)
                ks=sorted(self.impropers)
                for i, k in enumerate(ks):
                    imp=self.impropers[k]
                    of.write("{:{wid}d} {:{wtype}d} {:{wb}d} {:{wb}d} {:{wb}d} {:{wb}d}\n".format(k, *imp, wid=wid, wtype=wtype, wb=wb))
            
    def writeLMPTRJ(self, output_dump="dump.lammpstrj"):
        self.readfromFields()
        with open(output_dump, 'a') as fw:
            fw.write("ITEM: TIMESTEP\n")
            fw.write("{}\n".format(self.system.cframe))
            fw.write("ITEM: NUMBER OF ATOMS\n")
            fw.write("{}\n".format(len(self.atoms))) # since data is not "validated", box size and number of atoms are subject to change, say in deforming boxes or GCMC simulations
            if not self.istriclinic:
                fw.write("ITEM: BOX BOUNDS pp pp pp\n") # for nontriclinic box
                fw.write("{:18E} {:18E}\n".format(self.box[0][0], self.box[1][0]))
                fw.write("{:18E} {:18E}\n".format(self.box[0][1], self.box[1][1]))
                fw.write("{:18E} {:18E}\n".format(self.box[0][2], self.box[1][2]))
            else:
                raise NotImplementedError("LMPTRJ with triclinic simulation box not yet implemented!")
            fw.write("ITEM: ATOMS id mol type xu yu zu\n")
            for i,(k,a) in enumerate(self.atoms.items()):
                molID, typ, q, x, y, z = a
                fw.write("{aid} {molid} {atyp} {x:9e} {y:9e} {z:9e}\n".format(aid=k, molid=molID, atyp = typ, x=x, y=y, z=z))
