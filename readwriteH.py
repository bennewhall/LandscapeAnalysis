#!/usr/bin/env python
#
# Author: James Daniel Whitfield <James.D.Whitfield@dartmouth.edu>
#

import numpy
from pyscf import gto, scf, ao2mo
import os
import pdb
'''
Customizing Hamiltonian for SCF module.
Three steps to define Hamiltonian for SCF:
1. Specify the number of electrons. (Note mole object must be "built" before doing this step)
2. Overwrite three attributes of scf object
    .get_hcore
    .get_ovlp
    ._h2
3. Specify initial guess (to overwrite the default atomic density initial guess)
Note you will see warning message on the screen:
        Overwritten attributes  get_ovlp get_hcore  of <class 'pyscf.scf.hf.RHF'>
'''
def readH(path):
    mol = gto.M()
    M = 2
    mol.nelectron = 2

    mf = scf.RHF(mol)
    h1 = None
    S = 0
    M=0

    with open(path + 'hcore', 'r') as h1f:
        for line in h1f:
            line=line.strip()
            if line.startswith("#"):
                continue
            
            if line.startswith('M'):
                M_in = line.split()
                M_in = M_in[-1]
                M = int(M_in)
                h1 = numpy.zeros((M,M))
                #import pdb; pdb.set_trace()

                continue
                

            lineitems = line.split()
            i   = int(lineitems[0])
            j   = int(lineitems[1])
            val = float(lineitems[2])

            h1[i,j]=val

        if ( not numpy.allclose(h1.transpose() , h1 ) ):
            h1 = h1.transpose() + h1 - numpy.diag(numpy.diag(h1))

    if os.path.exists(path + 'ovlp'):
      with open(path + 'ovlp', 'r') as s1f:
        for line in s1f:
            line=line.strip()
            if line.startswith("#"):
                continue
            
            
            if line.startswith('M'):
                M_in = line.split()
                M_in = M_in[-1]
                if( not int(M_in) == M ):
                    print("Overlap matix dimensions are off")
                    1/0
                S = numpy.zeros((M,M))
                continue
                
                
            lineitems = line.split()
            
            i   = int(lineitems[0])
            j   = int(lineitems[1])
            val = float(lineitems[2])

            S[i,j]=val

        if ( not numpy.allclose(S.transpose() , S ) ):
            S = S.transpose() + S - numpy.diag(numpy.diag(S)) 
    else:
        S=eye(M)


    h2 = numpy.zeros((M,M,M,M))
    with open(path + "eri") as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('M'):
                M_in = line.split()
                M_in = M_in[-1]
                M = int(M_in)
                if( not int(M_in) == M ):
                    print("eri tensor dimensions are off")
                    1/0
                continue
            lineitems = line.split()

            i   = int(float(lineitems[0]))
            j   = int(float(lineitems[1]))
            k   = int(float(lineitems[2]))
            l   = int(float(lineitems[3]))
            val = float(lineitems[4])

            h2[i,j,k,l]=val

    #Save data into the mf object
    
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: S
    # ao2mo.restore(8, h2, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, h2, M)

    mol.incore_anyway = True

    #mf.mol=mol
    
    return mf

def writematrix(A,fname):
    if(A.shape[0]==A.shape[1]):
        M = A.shape[0]

    with open(fname,'w') as f:
        f.write(f"M {M}\n")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                val=A[i,j]
                f.write(f"{i} {j} {val}\n")

def writeH(mf):
    h1e = mf.get_hcore()   
    s1e = mf.get_ovlp()  
    eri = mf.mol.intor('int2e')
    M=h1e.shape[0]

    if not numpy.allclose(s1e , numpy.eye(s1e.shape[0])):
        writematrix(s1e,"ovlp")
    
    writematrix(h1e,'hcore')

    with open("eri",'w') as f :
        for i in range(eri.shape[0]):
            for j in range(eri.shape[1]):
                for k in range(eri.shape[2]):
                    for l in range(eri.shape[3]):
                        val = eri[i,j,k,l]
                        f.write(f"{i} {j} {k} {l} {val}\n")

    

