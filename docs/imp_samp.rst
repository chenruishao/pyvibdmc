Guided DMC / Importance Sampling
========================================

PyVibDMC supports the use of guiding functions to reduced the required number of walkers to obtain a reasonable result
from DMC simulations. These guiding functions must be supplied by the user.

Theory of Importance Sampling
-------------------------------------------------------
The theory of importance sampling in the context of vibrational Diffusion Monte Carlo is outlined in
the supplementary information of a paper by `Lee, Madison, and McCoy <https://pubs.acs.org/doi/abs/10.1021/acs.jpca.8b11213>`_.

Further applications can be found in papers by `Finney, DiRisio, and McCoy <https://pubs.acs.org/doi/10.1021/acs.jpca.0c07181>`_ as well as
`Lee, Vetterli, Boyer, and McCoy <https://pubs.acs.org/doi/full/10.1021/acs.jpca.0c05686?ref=recommended>`_.

Using Guiding functions in PyVibDMC
----------------------------------------------
The guiding function interface is quite similar to the potential energy interface for PyVibDMC. The user *must* provide
a Python function that takes in the ``num_walkers x num_atoms x 3`` walker array and evaluate the value of the trial
wave function for each walker. The code is restricted to use direct product wave functions.

The value of the trial wave function should be returned as a NumPy array of size ``num_walkers``.
The user may also pass along ``trial_kwargs`` in the form of a dictionary,
much like what can be done for
`the potential energy interface <https://pyvibdmc.readthedocs.io/en/latest/potentials.html#passing-more-than-just-the-coordinates-to-the-potential-manager>`_.

Optionally, the user may also provide one other Python function that calculate the first and second
derivatives of the wave function with respect to the ``3N`` Cartesian coordinates. The function should return
two ``num_walkers x num_atoms x 3`` NumPy arrays that correspond to the value of the derivatives of each walker
in the x,y, and z components of the various atoms in the molecular system **divided by the trial wave function**.

If no first and second derivatives are provided, PyVibDMC defaults to computing the derivatives numerically using finite
difference.

IMPORTANT REQUIREMENTS:

* The potential manager is responsible for setting up the parallelization in the importance sampling. If one is using multiprocessing in the potential manager, the importance sampling will be parallelized with a separate multiprocessing pool, MPI for MPI, and single-core for single-core.

  * With ``Potential`` and ``ImpSampManager``, ``Potential(num_cores=...)`` controls the potential-only pool. ``ImpSampManager(imp_num_cores=...)`` controls the importance-sampling-only pool and defaults to ``Potential.num_cores`` if not provided. DMC propagation uses these pools sequentially, so active CPU use is bounded by the currently running operation, while resident memory includes both pools.

  * The one exception to this rule is when using a NN_Potential. One can pass the ``new_pool_num_cores`` argument in order to use ImpSampManager, which uses multiprocessing, if using Potential_NoMP or NN_Potential.

* The Python file that holds the function that calculates the trial wave function (and optionally the derivatives) MUST be in the same directory as the Python file that calls the potential energy surface. This is a restrictive measure that was put in to make calls cleaner inside PyVibDMC, as sometimes the current working directory matters when one loads in data files on-the-fly and things of the like.

* External trial-wavefunction derivative code should be written to avoid unnecessary large temporaries inside inner loops. This is best handled in the trial-wavefunction implementation itself, rather than relying only on runtime settings.

  NumPy operations, or PyTorch if used in custom trial code, may automatically use threaded BLAS/OpenMP libraries. When trial-wavefunction or derivative calls are already parallelized with multiprocessing, those internal thread pools can oversubscribe the CPU. For these workloads, it is often best to set the relevant thread counts to one before importing NumPy, PyTorch, or running the simulation::

      export OMP_NUM_THREADS=1
      export OPENBLAS_NUM_THREADS=1
      export MKL_NUM_THREADS=1

There are three importance sampling managers: ``ImpSampManager``, ``ImpSampManager_NoMP``, and ``MPI_ImpSampManager``. The first two can
easily be used with PyVibDMC::

    import pyvibdmv as pv
    pot_func = ...
    py_file = ...
    pot_dir = ...

    """
    # No multiprocessing at all.
    water_pot = pv.Potential_NoMP(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          chdir=False)
    water_imp = pv.ImpSampManager_NoMP(trial_function,
                         trial_directory, #same as potential_directory if using MP or MPI
                         python_file,
                         pot_manager=water_pot,
                         chdir=..., #optional
                         deriv_function=..., #optional
                         trial_kwargs=..., #May pass a dict with important things to trial function call
                         deriv_kwargs=..., #May pass a dict with important things to trial function call - only use if deriv_function is set to something
                         )
    """

    # Using multiprocessing for potential and imp samp
    water_pot = pv.Potential(potential_function=pot_func,
                          python_file=py_file,
                          potential_directory=pot_dir,
                          num_cores=2) #potential uses its own pool of workers

    water_imp = pv.ImpSampManager(trial_function,
                         trial_directory, #same as potential_directory
                         python_file,
                         pot_manager=water_pot,
                         imp_num_cores=..., #optional, separate pool size; defaults to Potential.num_cores
                         deriv_function=..., #optional string like trial_function
                         trial_kwargs=..., #May pass a dict with important things to trial function call
                         deriv_kwargs=..., #May pass a dict with important things to deriv function call - only use if deriv_function is set to something
                         )
    # pass to DMC_Sim
    my_sim = pv.DMC_sim(...,
                        imp_samp=water_imp,
                        # imp_samp_oned=False, # only true if DMC sim is on 1D problem.
                                               # Defaults to False.
                        ...)

    # Close both multiprocessing pools after the simulation finishes.
    water_imp.mp_close()
    water_pot.mp_close()

Examples of a trial wavefunction (and derivatives) can be found in the Partridge Schwenke sample potential ``h2o_trial.py`` or the
harmonic oscillator sample potential ``harm_trial_wfn.py``.

The MPI version of the ImpSampManager can be used in the same way as above, except one must use an
``MPI_Potential`` object for the potential manager and and ``MPI_ImpSampManager`` object::

    import pyvibdmc as pv
    from pyvibdmc.simulation_utilities.mpi_potential_manager import MPI_Potential
    from pyvibdmc.simulation_utilities.mpi_imp_samp_manager import MPI_ImpSampManager
    ...
    # Must import this way in order to access the MPI modules.
    mpi_pot = MPI_Potential(...)
    mpi_imp = MPI_ImpSampManager(...)
    my_sim = pv.DMC_Sim(...,
                        imp_samp=mpi_imp,
                        ...)


Chain rule helper
----------------------------------------------

The McCoy group typically uses trial wave functions that are products of 1D wave functions. The wave functions
are typically functions of internal coordinates, bond lengths and bond angles in particular. Since the derivatives required
by importance sampling are with respect to Cartesian coordinates, it can be a non-trivial task to calculate the proper
derivatives. PyVibDMC has a ``ChainRuleHelper`` that can be used to help calculate Cartesian derivatives if one is
using internal coordinates for the trial wave function.  A concrete example is the water monomer. All of the
following code can be found in the tutorial ``Partridge_Schwenke_H2O`` directory::

    import pyvibdmc as pv
    import numpy as np
    # This is an example of user-side code that uses the ChainRuleHelper.

    def sec_deriv(cds):
        ... # Calculates the second derivative of psi with respect to r and theta at each of the coordinates.
            #  num_modes x num_walkers array

    def first_deriv(cds):
        ... # Calculates the first derivative of psi with respect to r and theta at each of the coordinates.
            # returns num_modes x num_walkers array

    def trial_wavefunction(cds, ret_pdt=True):
        """Calculates the trial wave function at each of the coordinates. ret_pdt will be true for
        pyvibdmc, but can be set to false so that we can construct derivatives down in dpsi_dx()."""
        ...
        if ret_pdt:
            return np.prod(psi...) # returns a num_walkers array. Used by default by PyVibDMC.
        else:
            return psi # returns a num_walkers x num_modes array.

    def dpsi_dx(cds):
        """Retruns the first and second derivative"""
        # First, calculate trial wave function (num_walkers x num_modes)
        trl = trial_wavefunction(cds, ret_pdt=False)
        dpsi_dr = first_deriv(cds) / trl
        d2psi_dr2 = sec_deriv(cds) / trl
        ohs = [[0, 2], [2, 1]]
        hoh = [0, 2, 1]

        # Chain rule derivatives
        # pass in numpy module, can also use cupy to get GPU accelerations
        crh = pv.ChainRuleHelper(cds, np)

        dr_dxs = np.stack([crh.dr_dx(oh) for oh in ohs])
        d2r_dx2s = np.stack([crh.d2r_dx2(oh, dr_dx=dr_dxs[num]) for num, oh in enumerate(ohs)])

        dcth_dx = crh.dcth_dx(hoh,
                              dr_da=dr_dxs[0],
                              dr_dc=dr_dxs[1])
        dth_dx = crh.dth_dx(hoh,
                            dcth_dx=dcth_dx,
                            dr_da=dr_dxs[0],
                            dr_dc=dr_dxs[1])

        d2th_dx2 = crh.d2th_dx2(hoh,
                                dcth_dx=dcth_dx,
                                dr_da=dr_dxs[0],
                                dr_dc=dr_dxs[1],
                                d2r_da2=d2r_dx2s[0],
                                d2r_dc2=d2r_dx2s[1])
        # Calculate dpsi/dx / psi
        dint_dx = np.concatenate((dr_dxs, xp.expand_dims(dth_dx, 0)))
        dp_dx = crh.dpsidx(dpsi_dr, dint_dx)

        # Calculate d2psi/dx2 / psi
        d2int_dx2 = np.concatenate((d2r_dx2s, xp.expand_dims(d2th_dx2, 0)))
        d2p_dx2 = crh.d2psidx2(d2psi_dr2, d2int_dx2, dpsi_dr, dint_dx)
        return dp_dx, d2p_dx2

Note that when the ``ChainRuleHelper`` calculates ``dpsidx`` and ``d2psidx2``, it assumes that the derivatives with respect
to Psi are divided through by the trial wave function. All else is done internally.
