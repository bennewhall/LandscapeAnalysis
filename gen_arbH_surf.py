#Test file to illustrate how to generate 2D surfaces using custom Hamiltonians
#
# Only works for 2D currently but should be clear how to generalize 
# (sees line: 77, 78)
#
# <James.D.Whitfield@Dartmouth.edu>

# Should be able to read-write Hamiltonians into rhf objects using this library

#Modified by Ben Newhall
import readwriteH

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

#Plotting control parameters
sample_points = 20
sample_max    = 2*np.pi
sample_min    = -2*np.pi

Z=1
################################################################################
print( """ 
This file reads input from [hcore, ovlp, eri] where the file formatting is 
"idx1 idx2 idx3 idx4 val" or "idx1 idx2 val". The first line should have
'M' followed by the number of basis functions.  Output is analysis.
""")
do_write=False
#change to do_write to True to generate example input files
num_tests= 9
#change num_tests to equal the number of test files
dataframes = []
for test in range(1,num_tests+1):
  path = 'Tests/GridTest'+str(test)+'/'
  with open(path+'analysis', 'w') as f: 
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
      rhf=readwriteH.readH(path)
  
    
    dm1 = rhf.init_guess_by_1e()
    h1e = rhf.get_hcore()                                           
    s1e = rhf.get_ovlp()  
     
    mo_energy, core_guess_coeff = rhf.eig(h1e, s1e)                         
    core_guess_occ = rhf.get_occ(mo_energy, core_guess_coeff)

    len(core_guess_occ[core_guess_occ==0])

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
    m = len(core_guess_occ[core_guess_occ==0])

    lower_bound = []
    upper_bound = []
    blocks = []
    for i in range(0,m):
      lower_bound.append(float(sample_min))
      upper_bound.append(float(sample_max))
      blocks.append(8.0)
    
    #Landscape Analysis
    sample = create_initial_sample(1000, m, type = 'random', lower_bound=lower_bound, upper_bound=upper_bound)
    obj_val = []

    for s in sample:
      obj_val.append(float(E_hf(s)))
    
   
    feat_obj = create_feature_object(sample, obj_val,minimize = True, lower=-10.0, upper=10.0)

    ela_features = calculate_feature_set(feat_obj,"ela_meta")
    


    print(ela_features, file=f)

    

    print("Optimizing using SCF")

    #do optimization using scf
    rhf.verbose = 4
    mf = rhf.run()
    
    print(mf.converged)
    print( "\n\n")

    
    ela_features["Converged"] = mf.converged
    ela_features["Total Energy"] = mf.e_tot
    ela_features["Run Number"] = [test]
    dataframes.append(pd.DataFrame.from_dict(ela_features))

pd.concat(dataframes)
print(pd.concat(dataframes))
    #make 2d plot
    # with open(path+'pes.dat', 'w') as datafile:
    #   NG = 100
    #   X1=np.linspace(-np.pi,np.pi,NG)                                                  
    #   X2=np.linspace(-np.pi,np.pi,NG)                                               
    #   for x1 in X1:                                                                 
    #     print("",file=datafile)        
    #     for x2 in X2:              
    #         x = np.zeros(m)   
    #         x[0]=x1
    #         x[1]=x2                                        
    #         energy = E_hf(x)
    #         print(x1,"\t",x2,"\t",energy, file = datafile)
  