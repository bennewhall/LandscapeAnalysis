#Test file to illustrate how to generate 2D surfaces using custom Hamiltonians
#
# Only works for 2D currently but should be clear how to generalize 
# (sees line: 77, 78)
#
# <James.D.Whitfield@Dartmouth.edu>

# Should be able to read-write Hamiltonians into rhf objects using this library
import readwriteH

import pyscf 
from pyscf import gto
import numpy as np
import scipy
import matplotlib.pyplot as plt                                                 
from matplotlib import cm 
from scipy import fftpack                                                      
from mpl_toolkits.mplot3d import Axes3D                                         
import matplotlib.pyplot as plt

#Plotting control parameters
sample_points = 20
sample_max    = -2*np.pi
sample_min    = 2*np.pi

Z=1
################################################################################
print( """ 
This file reads input from [hcore, ovlp, eri] where the file formatting is 
"idx1 idx2 idx3 idx4 val" or "idx1 idx2 val". The first line should have
'M' followed by the number of basis functions.  Output is [pes.dat] which
can be plotted using the gnu_plot script.
""")
do_write=True
#change to do_write to True to generate example input files

with open('pes.dat', 'w') as f: 
  if(do_write): 
    diatomic = pyscf.gto.Mole()
    diatomic.basis = 'sto-3g'
    x=2.5 
    y=2*x
    diatomic.atom = [[Z, (0, 0, 0)], [Z, (x, 0, 0)], [Z, (y, 0, 0)]]
    diatomic.verbose = 0
    diatomic.charge = 1
    diatomic.build()
  
    rhf = pyscf.scf.RHF(diatomic)
  
    readwriteH.writeH(rhf)
  else:    
    rhf=readwriteH.readH()
 

  dm1 = rhf.init_guess_by_1e()
  h1e = rhf.get_hcore()                                           
  s1e = rhf.get_ovlp()   
  mo_energy, core_guess_coeff = rhf.eig(h1e, s1e)                         
  core_guess_occ = rhf.get_occ(mo_energy, core_guess_coeff)

  def _rotate_mo(mo_coeff, mo_occ, dx):

    dr = pyscf.scf.hf.unpack_uniq_var(dx, mo_occ)
    u=scipy.linalg.expm(dr)
    return np.dot(mo_coeff, u)
 
  def E_hf(x):
    #rotate by scaled eigenvector
    x_mo_coeff = _rotate_mo(core_guess_coeff, core_guess_occ, x)
    dm=rhf.make_rdm1(x_mo_coeff,core_guess_occ)
 
    vhf = rhf.get_veff(dm= dm)  
 
    return rhf.energy_tot(dm,vhf=vhf)                                           

  NG = sample_points
  X1=np.linspace(sample_min,sample_max)
  X2=np.linspace(sample_min,sample_max)      
                                    
  for x1 in X1:                                                                 
    print("",file=f)        
    for x2 in X2:              
            x = np.array([x1,x2])                                           
            energy = E_hf(x)
            print(x1,"\t",x2,"\t",energy, file = f)
