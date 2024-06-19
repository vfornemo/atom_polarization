#!/usr/bin/env python
# -*- coding: utf-8 -*

from functools import reduce
import numpy
from pyscfad.lib import stop_grad
from pyscfad import gto, scf, lo, lib
from pyscfad.lo import pipek
from pyscfad import config
from typing import Union

config.update('pyscfad_scf_implicit_diff', True)

import jax
from jax import jacrev
from jax import numpy as jnp

jnp.set_printoptions(threshold=100000)
jnp.set_printoptions(linewidth=jnp.inf)


# static external electric field
E0 = jnp.array([0.] * 3)
# localization threshold
LOC_TOL = 1.e-10
# basis set
BASIS = 'aug-pcseg-1'
# orbitals and population method
MOS = ['pm']
POP_METHODS = ['mulliken','iao']

# test molecules and reference polarizabilities (from pyscf prop module)
H2O_GEOM = '''
O
H 1 0.96
H 1 0.96 2 104.52
'''
HF_GEOM = '''
H
F 1 0.91
'''
H2O_POL_REF = jnp.array([[ 8.85292796e+00,  3.33125639e-15, -3.56629337e-01],
                         [ 3.33125639e-15,  7.81868672e+00, -2.96946650e-15],
                         [-3.56629337e-01, -2.96946650e-15,  8.66820107e+00]])
HF_POL_REF = jnp.array([[ 5.63380397e+00,  3.10959550e-16, -5.55134039e-17],
                        [ 3.10959550e-16,  4.36007043e+00,  1.05471187e-15],
                        [-5.55134039e-17,  1.05471187e-15,  4.36007043e+00]])
MOLS = ['H2O', 'HF']
GEOMS = [H2O_GEOM, HF_GEOM]
POL_REFS = [H2O_POL_REF, HF_POL_REF]

def assign_rdm1s(mol, mo_coeff, pop_method):
    """
    list of population weights of each spin-orbital on the individual atoms
    """
    pops = pipek.atomic_pops(mol, mo_coeff, method = pop_method)
    return pops.diagonal(axis1=1, axis2=2).transpose()


def prop_tot(mol, mf, mo_coeff, mo_occ, weights):
    """
    atom-decomposed energy
    """
    # ao dipole integrals
    # ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    # atomic function
    
    # nuclear repulsion property
    # rdm1_tot = jnp.array(make_rdm1(mo_coeff, mo_occ))
    rdm1_tot = jnp.einsum('ip,jp->ij', mo_occ * mo_coeff, mo_coeff)
    rdm1_eff = jnp.array(rdm1_tot, rdm1_tot) * .5

    # print('rdm1_eff', rdm1_eff)
    

    # core hamiltonian
    kin, nuc, sub_nuc, ext = _h_core(mf, mol)
    # fock potential
    vj, vk = mf.get_jk(mol=mol, dm=rdm1_eff)
    
    
    def prop_atom(weights):
        """
        this function returns atomic contributions
        """
        # atom_idx = jnp.arange(mol.natm)

        
        # atom-specific rdm1
        def _loop(mo_coeff, mo_occ, weights):
            # orbital-specific rdm1
            rdm1_orb = jnp.outer(mo_occ * mo_coeff, mo_coeff)
            # weighted contribution to rdm1_atom
            return rdm1_orb * weights

        rdm1_atom = jax.vmap(_loop, (1,0,0))(mo_coeff, mo_occ, weights)
        rdm1_atom = jnp.sum(rdm1_atom, axis=0)
        
        exch = -jnp.einsum('ij,ij', vk, rdm1_atom) * 0.5
        coul = jnp.einsum('ij,ij', vj, rdm1_atom)
        kinetic = jnp.einsum('ij,ij', kin, rdm1_atom)
        nuc_att_loc = jnp.einsum('ij,ij', nuc, rdm1_atom) * 0.5
        external = jnp.einsum('ij,ij', ext, rdm1_atom)
        elec = kinetic + exch + coul + nuc_att_loc + external
        
        return elec
    
        
    # perform decomposition
    res = jax.vmap(prop_atom, (1,))(weights)
    
    nuc_att_glob = jnp.einsum('xij,ij->x', sub_nuc, rdm1_tot) * 0.5  # maybe wrong
    
    res = res + nuc_att_glob
         
    return res


def _h_core(mf: scf.hf.SCF, mol: gto.Mole):
        """
        this function returns the components of the core hamiltonian
        """
        # kinetic integrals
        kin = mol.intor_symmetric('int1e_kin')
        # coordinates and charges of nuclei
        coords = mol.atom_coords()
        charges = mol.atom_charges()
        # individual atomic potentials
        sub_nuc =jnp.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=jnp.float64)
        for k in range(mol.natm):
            with mol.with_rinv_at_nucleus(k):
                sub_nuc = sub_nuc.at[k].set(-mol.intor('int1e_rinv') * charges[k])
        # total nuclear potential
        nuc =jnp.sum(sub_nuc, axis=0)
        # Check additional terms in hcore
        ext = mf.get_hcore() - kin - nuc
        return kin, nuc, sub_nuc, ext

def main(mol, mf, mo_coeff, mo_occ, pop_method):
    """
    main program
    """
    # compute population weights
    weights = assign_rdm1s(mol, mo_coeff, pop_method)
    # atomic dipole moments
    return prop_tot(mol, mf, mo_coeff, mo_occ, weights)

def pm_jacobi_sweep(mol, mo_coeff, mo_occ, s1e, pop_method, conv_tol=LOC_TOL):
    orbocc = numpy.asarray(stop_grad(mo_coeff[:, mo_occ>0]))
    mlo = pipek.PM(stop_grad(mol), orbocc)
    mlo.pop_method = pop_method
    mlo.conv_tol = conv_tol
    _ = mlo.kernel()
    mlo = pipek.jacobi_sweep(mlo)
    orbloc = mlo.mo_coeff
    u0 = reduce(numpy.dot, (orbocc.T, stop_grad(s1e), orbloc))
    return jnp.dot(mo_coeff[:, mo_occ>0], u0)

def energy(E, mol, mo, pop_method):
    """
    compute atomic contributions to molecular dipole
    """
    mf = scf.RHF(mol)
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    h1 = mf.get_hcore()
    field = jnp.einsum('x,xij->ij', E, ao_dip)
    mf.get_hcore = lambda *args, **kwargs: h1 + field
    mf.kernel()
    assert mf.converged, 'mf not converged'
    # localized pm orbitals or ibos
    mo_occ = mf.mo_occ[mf.mo_occ > 0.]
    if mo == 'can':
        mo_coeff = mf.mo_coeff[:, mf.mo_occ > 0.]
    else:
        orbocc = pm_jacobi_sweep(mol, mf.mo_coeff, mf.mo_occ, mf.get_ovlp(), pop_method)
        mo_coeff = pipek.pm(mol, orbocc,
                            pop_method = pop_method, conv_tol = LOC_TOL)
    return main(mol, mf, mo_coeff, mo_occ, pop_method)

# loop over test molecules
for mol_name, geom, pol_ref in zip(MOLS, GEOMS, POL_REFS):
    for mo in MOS:
        for pop_method in POP_METHODS:
            mol = gto.Mole()
            mol.atom = geom
            mol.basis = BASIS
            mol.verbose = 3
            mol.build(trace_coords=False, trace_exp=False, trace_ctr_coeff=False)
            # atomic polarizability
            # atomic polarizability
            pol_full = -jacrev(jacrev(energy))(E0, mol, mo, pop_method)
            
            pol = jnp.sum(pol_full, axis=0)
            
            pol1 = jnp.trace(pol) / 3.
            # assert differences
            print(f'{mol_name:} / {mo:} / {pop_method:}:')
            # print('total polarizabilities:\n', pol1)
            print('total polarizabilities:\n', pol)
            print('isotropic polarizabilities:\n', pol1)
