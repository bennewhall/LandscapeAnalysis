#Sahil Gulania
#James Whitfield  
import pyscf 
from pyscf import gto
import numpy as np
from hessian import *
import pdb
import scipy
import matplotlib.pyplot as plt                                                 
from matplotlib import cm 
from scipy import fftpack                                                      
from mpl_toolkits.mplot3d import Axes3D                                         
import matplotlib.pyplot as plt
from openfermion.ops._givens_rotations import (                                 
    fermionic_gaussian_decomposition, givens_decomposition_square)

Z=1
Z1=1
Z2=1

with open('pes.dat', 'w') as f: 
  diatomic = pyscf.gto.Mole()
  diatomic.basis = 'sto-3g'
  x=2.5 
  y=2*x
  diatomic.atom = [[Z1, (0, 0, 0)], [Z2, (x, 0, 0)], [Z1, (y, 0, 0)]]
  diatomic.verbose = 0
  diatomic.charge = 1
  diatomic.build()
 
  rhf = pyscf.scf.RHF(diatomic)
  dm1 = pyscf.scf.rhf.init_guess_by_1e(diatomic)
  h1e = rhf.get_hcore(diatomic)                                           
  s1e = rhf.get_ovlp(diatomic)   
  mo_energy, core_guess_coeff = rhf.eig(h1e, s1e)                         
  core_guess_occ = rhf.get_occ(mo_energy, core_guess_coeff)

  def _rotate_mo(mo_coeff, mo_occ, dx):
    decompositio=[]
    dr = pyscf.scf.hf.unpack_uniq_var(dx, mo_occ)
    u=scipy.linalg.expm(dr)
    decomposition = givens_decomposition_square(u)
    return np.dot(mo_coeff, u)
 
  def E_hf(x):
    #rotate by scaled eigenvector
    x_mo_coeff = _rotate_mo(core_guess_coeff, core_guess_occ, x)
    dm=rhf.make_rdm1(x_mo_coeff,core_guess_occ)
 
    vhf = rhf.get_veff(diatomic, dm)  
    s1e = rhf.get_ovlp(diatomic)  
 
    return rhf.energy_tot(dm,vhf=vhf)                                           
 
  # one zone
  #NG = 250
  #X1=np.linspace(-4*np.pi,4*np.pi,NG)                                                  
  #X2=np.linspace(-4*np.pi,4*np.pi,NG)                                               

  NG = 1000
  X1=np.linspace(-16*np.pi,16*np.pi,NG)                                                  
  X2=np.linspace(-16*np.pi,16*np.pi,NG)                                               
  for x1 in X1:                                                                 
    print("",file=f)        
    for x2 in X2:              
            x = np.array([x1,x2])                                           
            energy = E_hf(x)
            print(x1,"\t",x2,"\t",energy, file = f)
