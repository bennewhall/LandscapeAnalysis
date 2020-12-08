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
from mpl_toolkits.mplot3d import Axes3D     
from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features    

fig = plt.figure()                                                              
ax = fig.add_subplot(111, projection='3d') 


Z=1
Z1=1
Z2=1

sample = create_initial_sample(100, 2, type = 'lhs', lower_bound=[.2 , 0], upper_bound=[3.2, np.pi])

obj_val = []

for s in sample:                                                           
  diatomic = pyscf.gto.Mole()
  diatomic.basis = 'sto-3g'
  x=s[0]
  diatomic.atom = [[Z1, (0, 0, 0)], [Z2, (x, 0, 0)]]
  diatomic.verbose = 0
  diatomic.build()

  rhf = pyscf.scf.RHF(diatomic)                                                
  dm1 = pyscf.scf.rhf.init_guess_by_1e(diatomic)                               
  h1e = rhf.get_hcore(diatomic)                                                
  s1e = rhf.get_ovlp(diatomic)                                                 
  mo_energy, core_guess_coeff = rhf.eig(h1e, s1e)                              
  core_guess_occ = rhf.get_occ(mo_energy, core_guess_coeff)

  def _rotate_mo(mo_coeff, mo_occ, dx):
    dr = pyscf.scf.hf.unpack_uniq_var(dx, mo_occ)
    
    u=scipy.linalg.expm(dr)
    #print("u=",u2)
    #print( np.cos(x) * np.eye(2) - np.sin(x)*np.matrix([[0,1],[-1,0]]))
    return np.dot(mo_coeff, u)



  def E_hf(x):
    #rotate by scaled eigenvector
    x_mo_coeff = _rotate_mo(core_guess_coeff, core_guess_occ, x)  
    dm=rhf.make_rdm1(x_mo_coeff,core_guess_occ)  

    #print("dmr=",dm)

    #energy with new broken density matrix                                
    #h1e = rhf.get_hcore(diatomic)                                                   
    vhf = rhf.get_veff(diatomic, dm)  
    s1e = rhf.get_ovlp(diatomic)  

    #print("Tr(PS)=", dm.dot(s1e).trace() )
    return rhf.energy_tot(dm,vhf=vhf)   

  obj_val.append(E_hf(s[1]))

feat_obj = create_feature_object(sample, obj_val,lower=0, upper=3.3, blocks = [5,5])

ela_features = calculate_features(feat_obj)

print(ela_features)
                                                
  
                      
