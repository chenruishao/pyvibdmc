import pytest
import pyvibdmc as pv
import os
import numpy as np

sim_ex_dir = "imp_samp_results"


def test_imp_manager_defaults_to_shared_potential_pool():
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')
    harm_pot = pv.Potential(potential_function='oh_stretch_harm',
                            python_file='harmonicOscillator1D.py',
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             deriv_function='derivative')

    try:
        assert impo.pool is harm_pot.pool
        assert impo.num_cores == harm_pot.num_cores
        trial = impo.call_trial(np.zeros((4, 1, 1)))
        assert trial.shape == (4,)
    finally:
        impo.mp_close()
        harm_pot.mp_close()


def test_imp_manager_uses_explicit_imp_pool_size_for_shared_potential_pool():
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')
    harm_pot = pv.Potential(potential_function='oh_stretch_harm',
                            python_file='harmonicOscillator1D.py',
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             imp_num_cores=1,
                             deriv_function='derivative')

    try:
        assert impo.pool is harm_pot.pool
        assert impo.num_cores == 1
        assert harm_pot.num_cores == 2
        assert harm_pot.pool_num_cores == 2
        assert harm_pot.pool._processes == 2
        trial = impo.call_trial(np.zeros((4, 1, 1)))
        derivz, sderivz = impo.call_derivs(np.zeros((4, 1, 1)))
        impo.mp_close()
        pot_vals = harm_pot.getpot(np.zeros((4, 1, 1)))
        assert trial.shape == (4,)
        assert derivz.shape == (4, 1, 1)
        assert sderivz.shape == (4, 1, 1)
        assert pot_vals.shape == (4,)
    finally:
        impo.mp_close()
        harm_pot.mp_close()


def test_explicit_imp_num_cores_reserves_single_core_potential_worker(tmp_path):
    potential_file = tmp_path / "chunk_probe_potential.py"
    trial_file = tmp_path / "chunk_probe_trial.py"
    potential_file.write_text(
        "import numpy as np\n\n"
        "def chunk_size_potential(cds):\n"
        "    return np.repeat(len(cds), len(cds))\n"
    )
    trial_file.write_text(
        "import numpy as np\n\n"
        "def chunk_size_trial(cds):\n"
        "    return np.repeat(len(cds), len(cds))\n"
    )

    pot = pv.Potential(potential_function='chunk_size_potential',
                       python_file=potential_file.name,
                       potential_directory=str(tmp_path),
                       num_cores=1)
    initial_pool = pot.pool
    impo = pv.ImpSampManager(trial_function='chunk_size_trial',
                             trial_directory=str(tmp_path),
                             python_file=trial_file.name,
                             pot_manager=pot,
                             imp_num_cores=4)

    try:
        potential_pool = pot.pool
        assert potential_pool is not initial_pool
        assert impo.pool is pot.pool
        assert pot.num_cores == 1
        assert pot.pool_num_cores == 5
        assert pot.pool._processes == 5
        assert impo.num_cores == 4
        np.testing.assert_array_equal(pot.getpot(np.zeros((8, 1, 1))), np.repeat(8, 8))
        np.testing.assert_array_equal(impo.call_trial(np.zeros((8, 1, 1))), np.repeat(2, 8))
        assert pot.pool is potential_pool
        impo.mp_close()
        np.testing.assert_array_equal(pot.getpot(np.zeros((8, 1, 1))), np.repeat(8, 8))
        assert pot.pool is potential_pool
        pot.mp_close()
        assert pot.pool is None
    finally:
        impo.mp_close()
        pot.mp_close()


def test_reserved_single_core_potential_worker_keeps_getpot_on_one_pid(tmp_path):
    potential_file = tmp_path / "pid_probe_potential.py"
    trial_file = tmp_path / "pid_probe_trial.py"
    log_file = tmp_path / "pid_probe.log"
    potential_file.write_text(
        "import os\n"
        "import numpy as np\n\n"
        "def pid_probe_potential(cds):\n"
        f"    with open({str(log_file)!r}, 'a', encoding='ascii') as handle:\n"
        "        handle.write(f'{os.getpid()}\\n')\n"
        "    return np.zeros(len(cds))\n",
        encoding="utf-8")
    trial_file.write_text(
        "import numpy as np\n\n"
        "def pid_probe_trial(cds):\n"
        "    return np.ones(len(cds))\n",
        encoding="utf-8")

    pot = pv.Potential(potential_function='pid_probe_potential',
                       python_file=potential_file.name,
                       potential_directory=str(tmp_path),
                       num_cores=1)
    impo = pv.ImpSampManager(trial_function='pid_probe_trial',
                             trial_directory=str(tmp_path),
                             python_file=trial_file.name,
                             pot_manager=pot,
                             imp_num_cores=4)

    try:
        assert pot.pool_num_cores == 5
        assert impo.num_cores == 4
        for _ in range(12):
            np.testing.assert_array_equal(pot.getpot(np.zeros((8, 1, 1))), np.zeros(8))

        pids = {line.strip() for line in log_file.read_text(encoding="ascii").splitlines()}
        assert len(pids) == 1
    finally:
        impo.mp_close()
        pot.mp_close()


def test_imp_manager_nomp_pool_override():
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')
    harm_pot = pv.Potential_NoMP(potential_function='oh_stretch_harm',
                                 python_file='harmonicOscillator1D.py',
                                 potential_directory=potDir)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             new_pool_num_cores=2,
                             deriv_function='derivative')

    try:
        trial = impo.call_trial(np.zeros((4, 1, 1)))
        derivz, sderivz = impo.call_derivs(np.zeros((4, 1, 1)))
        assert impo.num_cores == 2
        assert trial.shape == (4,)
        assert derivz.shape == (4, 1, 1)
        assert sderivz.shape == (4, 1, 1)
    finally:
        impo.mp_close()


def test_drift_serial_matches_multiprocessing_drift():
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')
    harm_pot = pv.Potential(potential_function='oh_stretch_harm',
                            python_file='harmonicOscillator1D.py',
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             deriv_function='derivative')

    try:
        imp = pv.ImpSamp(impo)
        cds = np.linspace(-0.2, 0.3, 6).reshape(6, 1, 1)
        drift_vals = imp.drift(cds)
        serial_vals = imp.drift_serial(cds)
        for drift_val, serial_val in zip(drift_vals, serial_vals):
            np.testing.assert_allclose(serial_val, drift_val)
    finally:
        harm_pot.mp_close()


def test_drift_serial_matches_nomp_drift_with_finite_difference():
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')
    impo = pv.ImpSampManager_NoMP(trial_function='trial_harm',
                                  trial_directory=potDir,
                                  python_file='harm_trial_wfn.py',
                                  chdir=True)

    imp = pv.ImpSamp(impo)
    cds = np.linspace(-0.2, 0.3, 6).reshape(6, 1, 1)
    drift_vals = imp.drift(cds)
    serial_vals = imp.drift_serial(cds)
    for drift_val, serial_val in zip(drift_vals, serial_vals):
        np.testing.assert_allclose(serial_val, drift_val)


def test_local_energy_logging_respects_log_every(tmp_path):
    pot_dir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')
    potential = pv.Potential_NoMP(potential_function='oh_stretch_harm',
                                  python_file='harmonicOscillator1D.py',
                                  potential_directory=pot_dir)
    imp_samp = pv.ImpSampManager_NoMP(trial_function='trial_harm',
                                      trial_directory=pot_dir,
                                      python_file='harm_trial_wfn.py',
                                      deriv_function='derivative',
                                      chdir=True)
    output_folder = tmp_path / "local_energy_log_cadence"
    sim = pv.DMC_Sim(sim_name="local_energy_log_cadence",
                     output_folder=str(output_folder),
                     weighting='discrete',
                     num_walkers=100,
                     num_timesteps=5,
                     equil_steps=10,
                     chkpt_every=100,
                     wfn_every=100,
                     desc_wt_steps=1,
                     atoms=["O-H"],
                     delta_t=1,
                     potential=potential,
                     second_impsamp_displacement=True,
                     imp_samp=imp_samp,
                     imp_samp_oned=True,
                     log_every=2,
                     start_structures=np.zeros((1, 1, 1)))

    try:
        sim.propagate()
        sim._logger.fl.flush()
    finally:
        sim._logger.fl.close()

    log_text = (output_folder / "local_energy_log_cadence_log.txt").read_text()
    log_blocks = log_text.split("Time step ")[1:]

    assert [int(block.splitlines()[0]) for block in log_blocks] == [0, 2, 4]
    assert all(block.count("Average local energy of ensemble:") == 1 for block in log_blocks)


def test_run_dmc_short():
    import shutil
    if os.path.isdir(sim_ex_dir):
        shutil.rmtree(sim_ex_dir)

    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'harmonicOscillator1D.py'
    potFunc = 'oh_stretch_harm'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager_NoMP(trial_function='trial_harm',
                                  trial_directory=potDir,
                                  python_file='harm_trial_wfn.py',
                                  deriv_function='derivative',
                                  chdir=True)

    myDMC = pv.DMC_Sim(sim_name="harm_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=1000,
                       num_timesteps=500,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       second_impsamp_displacement=True,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       )
    myDMC.run()
    assert True


def test_run_dmc_short_morse():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'morse_osc_1d.py'
    potFunc = 'oh_stretch_morse'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             deriv_function='derivative')

    myDMC = pv.DMC_Sim(sim_name="morse_osc_test",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=500,
                       num_timesteps=100,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=harm_pot,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       )
    myDMC.run()
    assert True

def test_run_dmc_large_ts_morse():
    # initialize potential
    potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/PythonPots/')  # only necesary for testing
    # purposes
    pyFile = 'morse_osc_1d.py'
    potFunc = 'oh_stretch_morse'
    harm_pot = pv.Potential(potential_function=potFunc,
                            python_file=pyFile,
                            potential_directory=potDir,
                            num_cores=2)

    impo = pv.ImpSampManager(trial_function='trial_harm',
                             trial_directory=potDir,
                             python_file='harm_trial_wfn.py',
                             pot_manager=harm_pot,
                             deriv_function='derivative')

    myDMC = pv.DMC_Sim(sim_name="large_ts_impsamp",
                       output_folder=sim_ex_dir,
                       weighting='discrete',
                       num_walkers=500,
                       num_timesteps=1000,
                       equil_steps=5,
                       chkpt_every=10,
                       wfn_every=10,
                       desc_wt_steps=5,
                       atoms=["O-H"],
                       delta_t=10,
                       potential=harm_pot,
                       imp_samp=impo,
                       imp_samp_oned=True,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       )
    myDMC.run()
    assert True


# def test_water():
#     # initialize potential
#     potDir = os.path.join(os.path.dirname(__file__), '../sample_potentials/FortPots/Partridge_Schwenke_H2O/')
#     # purposes
#     pyFile = 'h2o_potential.py'
#     potFunc = 'water_pot'
#     harm_pot = pv.Potential(potential_function=potFunc,
#                             python_file=pyFile,
#                             potential_directory=potDir,
#                             num_cores=8)
#
#     water_coord = np.array([[1.81005599, 0., 0.],
#                             [-0.45344658, 1.75233806, 0.],
#                             [0., 0., 0.]]) * 1.01
#     start_coord = np.expand_dims(water_coord, axis=0)  # Make it (1 x num_atoms x 3)
#
#     ex_args = {'dists':[[0,2],[2,1]],
#                'angs':[[0,2,1]]}
#     impo = pv.ImpSampManager(trial_function='trial_wavefunction',
#                              trial_directory=potDir,
#                              python_file='call_trl_h2o.py',
#                              pot_manager=harm_pot,
#                              deriv_kwargs=ex_args,
#                              trial_kwargs=ex_args)
#
#     myDMC = pv.DMC_Sim(sim_name="water_impsamp_test",
#                        output_folder=sim_ex_dir,
#                        weighting='discrete',
#                        num_walkers=2000,
#                        num_timesteps=5000,
#                        equil_steps=5,
#                        chkpt_every=10,
#                        wfn_every=10,
#                        desc_wt_steps=5,
#                        atoms=["H", "H", "O"],
#                        delta_t=1,
#                        potential=harm_pot,
#                        imp_samp=impo,
#                        log_every=1,
#                        start_structures=start_coord,
#                        )
#     myDMC.run()
#     assert True
