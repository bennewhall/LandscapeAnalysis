from pyscf import tools
from pyscf import lib
from pyscf.soscf import newton_ah
#from pyscf.soscf.newton_ah import _gen_rhf_response, _gen_uhf_response
import numpy as np

#computing the full hessian matrix
def get_orbital_hessian(mo,mo_ints,mo_occ):
    "Compute orbital hessian.  Needs h1 and h2 and an occupancy vector"
    
    if not isinstance(mo_occ, list): #assume they've passed the number of electron direct
        o = int(mo_occ/2) 
    else:
        o = sum(mo_occ)
    v = len(mo)-o
    #print(o,v)
    H = np.zeros((o*v,o*v))
    #print(mo_ints.shape)
    mm=-1
    for l in range (0,v):
     for k in range (0,o):
       i = k
       a = l+o
       mm = mm+1
       nn = -1
       for n in range (0,v):
        for m in range (0,o):
          j = m 
          b = n+o
          nn=nn+1
          #print(i,a,j,b)
          #print(mm,nn)
          if (i==j and a==b):
           H[mm][nn] = mo[a]-mo[i]-mo_ints[i][i][a][a]-mo_ints[i][a][i][a]
          else: 
           H[mm][nn] = -mo_ints[i][j][a][b]-mo_ints[i][b][j][a] 
       
    E,V = np.linalg.eigh(H)

    return E,V

def symm_is_broken(x):
    """Get the rhf and uhf solutions as a function of R
       Returns (E_rhf, E_uhf) with nuclear energy"""
    debug=verbose
    skip_uhf=False

    diatomic.atom = [[Z, (0, 0, 0)], [Z, (x, 0, 0)]]
    diatomic.build()

    rhf = pyscf.scf.RHF(diatomic)
    rhf.callback = rhf_callback
    rhf.check_convergence=True
    rhf.max_cycle=150
    rhf.run()

    #get dm
    Er = rhf.e_tot  # electron + nuclear energy
   
    #HESSSIAN CHECK    
    mo_energies = rhf.mo_energy

    #two-body transform to MO basis                                             
    ao_ints = diatomic.intor('cint2e_sph') #chemist notation i.e. ints[i,j,k,l]= (ij|kl)                                                                 
    mo_ints = pyscf.ao2mo.incore.full(ao_ints, rhf.mo_coeff) #convert to MO basis
    rhf.mo_ints=mo_ints
    [E,V]=get_orbital_hessian(mo_energies, mo_ints, diatomic.nelectron)
    
    minE = min(E)
    
    if minE < -1e-5:
        return True
    else:
        return False


#Modified from file stability.py
def _rotate_mo(mo_coeff, mo_occ, dx):
    dr = pyscf.scf.hf.unpack_uniq_var(dx, mo_occ)
    u = newton_ah.expmat(dr)
    return np.dot(mo_coeff, u)

def rhf_external(mf, with_symmetry=True, show_hessian=True, verbose=None):
        #log = logger.new_logger(mf, verbose)
        hop1, hdiag1, hop2, hdiag2 = _gen_hop_rhf_external(mf, with_symmetry)

        def precond(dx, e, x0):
            hdiagd = hdiag1 - e
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            return dx/hdiagd
        
        x0 = np.zeros_like(hdiag1)
        x0[hdiag1>1e-5] = 1. / hdiag1[hdiag1>1e-5]
        if not with_symmetry:  # allow to break point group symmetry
            x0[np.argmin(hdiag1)] = 1
        e1, v1 = lib.davidson(hop1, x0, precond, tol=1e-4)
        if e1 < -1e-5:
            print('RHF/RKS wavefunction has a real -> complex instablity at ',x)
        #else:
            #print('RHF/RKS wavefunction is stable in the real -> complex stability analysis')

        def precond(dx, e, x0):
            hdiagd = hdiag2 - e
            hdiagd[abs(hdiagd)<1e-8] = 1e-8
            return dx/hdiagd
        x0 = v1

        #hop2 is a function (so as to work with the davidson algorithm)
        #we get the matrix by applying to each basis vector.
        if show_hessian==True:
            #capital Hop2 is the matrix
            hop2_matrix=np.zeros((len(x0),len(x0)))

            for k in range(len(x0)):
                xk=np.zeros_like(x0)
                xk[k]=1
                yk=hop2(xk)

                hop2_matrix[:,k]=yk

            #better printing to read the hessian
            np.set_printoptions(linewidth=300,suppress=True,precision=3)

            print("Printing Hessian:")
            print(hop2_matrix)
            print("Done")

            #reset parameters
            np.set_printoptions(linewidth=80, suppress=False,precision=5)

        e3, v3 = lib.davidson(hop2, x0, precond, tol=1e-10)
        
        if e3 < -1e-5:
            if verbose:
                print('RHF/RKS wavefunction has a RHF/RKS -> UHF/UKS instablity at ',x)
                print(e3)
                print(v3)
            Ca,Cb = (_rotate_mo(mf.mo_coeff, mf.mo_occ, v3), mf.mo_coeff)

            mo_occ = mf.get_occ(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff)

            dm_broken=pyscf.scf.UHF(diatomic).make_rdm1((Ca, Cb), (mo_occ, mo_occ))
            return dm_broken
        else:
            return False;           
    

def _gen_hop_rhf_external(mf, with_symmetry=True, verbose=None):
        mol = mf.mol
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        occidx = np.where(mo_occ==2)[0]
        viridx = np.where(mo_occ==0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        if with_symmetry and mol.symmetry:
            orbsym = hf_symm.get_orbsym(mol, mo_coeff)
            sym_forbid = orbsym[viridx].reshape(-1,1) != orbsym[occidx]

        h1e = mf.get_hcore()
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = h1e + mf.get_veff(mol, dm0)
        fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:,None],occidx]
        fvv = fock[viridx[:,None],viridx]

        hdiag = fvv.diagonal().reshape(-1,1) - foo.diagonal()
        if with_symmetry and mol.symmetry:
            hdiag[sym_forbid] = 0
        hdiag = hdiag.ravel()

        vrespz = mf.gen_response(mf, singlet=None, hermi=2)
        def hop_real2complex(x1):
            x1 = x1.reshape(nvir,nocc)
            if with_symmetry and mol.symmetry:
                x1 = x1.copy()
                x1[sym_forbid] = 0
            x2 = np.einsum('ps,sq->pq', fvv, x1)
            x2-= np.einsum('ps,rp->rs', foo, x1)

            d1 = reduce(np.dot, (orbv, x1*2, orbo.conj().T))
            dm1 = d1 - d1.conj().T
    
            v1 = vrespz(dm1)
            x2 += reduce(np.dot, (orbv.conj().T, v1, orbo))
            if with_symmetry and mol.symmetry:
                x2[sym_forbid] = 0
            return x2.ravel()

        vresp1 = mf.gen_response(mf, singlet=False, hermi=1)
        def hop_rhf2uhf(x1):
            from pyscf.dft import numint
            # See also rhf.TDA triplet excitation
            x1 = x1.reshape(nvir,nocc)
            if with_symmetry and mol.symmetry:
                x1 = x1.copy()
                x1[sym_forbid] = 0
            x2 = np.einsum('ps,sq->pq', fvv, x1)
            x2-= np.einsum('ps,rp->rs', foo, x1)

            d1 = reduce(np.dot, (orbv, x1*2, orbo.conj().T))
            dm1 = d1 + d1.conj().T
            v1ao = vresp1(dm1)
            x2 += reduce(np.dot, (orbv.conj().T, v1ao, orbo))
            if with_symmetry and mol.symmetry:
                x2[sym_forbid] = 0
            return x2.real.ravel()

        return hop_real2complex, hdiag, hop_rhf2uhf, hdiag



"""Naive symmetry breaking that works for simple problems by arbitrarily 
breaking the symmetry.  However, one should use the Hessian to correctly
follow the saddle point"""
def init_guess_mixed(mol, mixing_parameter=np.pi/4):
    ''' Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.

    psi_1a = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
    psi_1b = np.cos(q)*psi_homo - np.sin(q)*psi_lumo

    psi_2a = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
    psi_2b =  np.sin(q)*psi_homo + np.cos(q)*psi_lumo

    Returns:
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi

    # based on hf.init_guess_by_1e()
    rhf = pyscf.scf.RHF(mol)
    h1e = rhf.get_hcore(mol)
    s1e = rhf.get_ovlp(mol)
    mo_energy, mo_coeff = rhf.eig(h1e, s1e)
    mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx = 0
    lumo_idx = 1

    for i in range(len(mo_occ)-1):
        if mo_occ[i] > 0 and mo_occ[i+1] < 0:
            homo_idx = i
            lumo_idx = i+1

    psi_homo = mo_coeff[:, homo_idx]
    psi_lumo = mo_coeff[:, lumo_idx]

    Ca = np.zeros_like(mo_coeff)
    Cb = np.zeros_like(mo_coeff)

    # mix homo and lumo of alpha and beta coefficients
    q = mixing_parameter

    Ca_homo = np.cos(q)*psi_homo + np.sin(q)*psi_lumo
    Cb_homo = np.cos(q)*psi_homo - np.sin(q)*psi_lumo

    Ca_lumo = -np.sin(q)*psi_homo + np.cos(q)*psi_lumo
    Cb_lumo = np.sin(q)*psi_homo + np.cos(q)*psi_lumo

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:, k] = Ca_homo
            Cb[:, k] = Cb_homo
        elif k == lumo_idx:
            Ca[:, k] = Ca_lumo
            Cb[:, k] = Cb_lumo
        else:
            Ca[:, k] = mo_coeff[:, k]
            Cb[:, k] = mo_coeff[:, k]

    dm = pyscf.scf.UHF(mol).make_rdm1((Ca, Cb), (mo_occ, mo_occ))
    return dm


