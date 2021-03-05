#File for reading in qm7 dataset and doing landscape analysis on rhf landscapes


import pyscf 
from pyscf import gto, scf
import numpy as np
import scipy
import matplotlib.pyplot as plt                                                 
from matplotlib import cm 
from scipy import fftpack                                                      
from mpl_toolkits.mplot3d import Axes3D                                         
import matplotlib.pyplot as plt
from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features, list_available_feature_sets, plot_information_content
import pandas as pd
import math
import multiprocessing as mp

#Plotting control parameters
sample_points = 20
sample_max    = 2*np.pi
sample_min    = -2*np.pi

#read data from csv into pandas 

r_df = pd.read_csv("R.csv", header=None)

z_df = pd.read_csv("Z.csv", header=None)

#for each mol in dataset construct mol and do landscape analysis

dataframes = []

for i, data in r_df.iterrows():

    charges = z_df.iloc[i]

    c_np = charges.to_numpy()

    r_np = data.to_numpy() 

    mol = gto.Mole()
    for e in range(0,c_np.size):

        if(c_np[e] != 0):
            mol.atom.extend( [ [ c_np[e] , [ r_np[e], r_np[c_np.size+e], r_np[ c_np.size*2+e ] ] ] ] )
            

    mol.build()

    print(i)


    rhf = pyscf.scf.RHF(mol)

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

    #find dimension of x

    nmo = len(core_guess_occ)
    idx = pyscf.scf.hf.uniq_var_indices(core_guess_occ)

    x1 = np.zeros((nmo,nmo))                                 
    m = len(x1[idx])

    lower_bound = []
    upper_bound = []
    blocks = []

    for r in range(0,m):
      lower_bound.append(float(sample_min))
      upper_bound.append(float(sample_max))
      blocks.append(8.0)
    
    #Landscape Analysis
    sample = create_initial_sample(100, m, type = 'random', lower_bound=lower_bound, upper_bound=upper_bound)
    obj_val = []

    
    for s in sample:
      obj_val.append(float(E_hf(s)))
    
   
    feat_obj = create_feature_object(sample, obj_val,minimize = True, lower=-10.0, upper=10.0)

    ela_features = calculate_feature_set(feat_obj,"ela_meta")
    


    #print(ela_features)
    def rhf_callback(envs):
      energy = envs['last_hf_e']
      cycle = envs['cycle']
      if(envs['scf_conv']):
          rhf_callback.nc=cycle+1

      if not hasattr(rhf_callback,'ens'):
          rhf_callback.ens=[]
          
      rhf_callback.ens.append(energy)



    print("Optimizing using SCF")

    rhf.callback = rhf_callback

    mf = rhf.run()

    #energies at step k
    #hf_ens= rhf_callback.ens

    #last energy should be converged value

    
    print(mf.converged)
    print( "\n\n")
    

    
    ela_features["Converged"] = mf.converged
    ela_features["Total Energy"] = mf.e_tot

    if mf.converged :
      #number of cycles to convergence via callback function
      ela_features["nsteps"] = rhf_callback.nc
    else :

      ela_features["nsteps"] = 0
      print(0)

    ela_features["Run Number"] = [i]
    dataframes.append(pd.DataFrame.from_dict(ela_features))

pd.concat(dataframes)
print(pd.concat(dataframes))



    


