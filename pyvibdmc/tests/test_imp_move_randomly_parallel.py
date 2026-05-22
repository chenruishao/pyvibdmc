import os

import numpy as np
import pytest

import pyvibdmc as pv


MOVE_SEED = 24681357


def _harmonic_potential(num_cores):
    pot_dir = os.path.join(os.path.dirname(__file__), "../sample_potentials/PythonPots/")
    return pv.Potential(potential_function="oh_stretch_harm",
                        python_file="harmonicOscillator1D.py",
                        potential_directory=pot_dir,
                        num_cores=num_cores)


def _importance_sampler(potential, imp_num_cores=None, trial_directory=None, python_file="harm_trial_wfn.py",
                        pass_timestep=False):
    if trial_directory is None:
        trial_directory = os.path.join(os.path.dirname(__file__), "../sample_potentials/PythonPots/")
    trial_kwargs = {} if pass_timestep else None
    deriv_kwargs = {} if pass_timestep else None
    return pv.ImpSampManager(trial_function="trial_harm",
                             trial_directory=trial_directory,
                             python_file=python_file,
                             pot_manager=potential,
                             imp_num_cores=imp_num_cores,
                             pass_timestep=pass_timestep,
                             deriv_function="derivative",
                             trial_kwargs=trial_kwargs,
                             deriv_kwargs=deriv_kwargs)


def _dmc_sim(tmp_path,
             label,
             potential_num_cores,
             imp_num_cores,
             num_walkers,
             trial_directory=None,
             python_file="harm_trial_wfn.py",
             pass_timestep=False,
             use_data_parallel_imp_move=False):
    potential = _harmonic_potential(potential_num_cores)
    imp_samp = _importance_sampler(potential,
                                   imp_num_cores=imp_num_cores,
                                   trial_directory=trial_directory,
                                   python_file=python_file,
                                   pass_timestep=pass_timestep)
    start_structures = np.linspace(-0.2, 0.3, num_walkers).reshape(num_walkers, 1, 1)
    sim = pv.DMC_Sim(sim_name=f"imp_move_parallel_{label}",
                     output_folder=str(tmp_path / label),
                     weighting="discrete",
                     num_walkers=num_walkers,
                     num_timesteps=1,
                     equil_steps=1,
                     chkpt_every=100,
                     wfn_every=100,
                     desc_wt_steps=1,
                     atoms=["O-H"],
                     delta_t=1,
                     potential=potential,
                     imp_samp=imp_samp,
                     imp_samp_oned=True,
                     use_data_parallel_imp_move=use_data_parallel_imp_move,
                     log_every=1,
                     start_structures=start_structures,
                     cur_timestep=0)
    return sim, imp_samp, potential


def _close_sim(sim, imp_samp, potential):
    sim._logger.fl.close()
    imp_samp.mp_close()
    potential.mp_close()


def _assert_move_results_match(old_sim, new_sim, old_rejections, new_rejections):
    assert new_rejections == old_rejections
    assert new_sim.dt_factor == old_sim.dt_factor
    np.testing.assert_allclose(new_sim.walkers, old_sim.walkers, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_sim.f_x, old_sim.f_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_sim.psi_1, old_sim.psi_1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_sim.psi_sec_der, old_sim.psi_sec_der, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(("potential_num_cores", "imp_num_cores"), [(2, None), (4, 1), (1, 4)])
def test_imp_move_randomly_data_parallel_matches_existing_method(tmp_path, potential_num_cores, imp_num_cores):
    num_walkers = 259
    old_sim, old_imp, old_potential = _dmc_sim(tmp_path,
                                               "old",
                                               potential_num_cores,
                                               imp_num_cores,
                                               num_walkers)
    new_sim, new_imp, new_potential = _dmc_sim(tmp_path,
                                               "new",
                                               potential_num_cores,
                                               imp_num_cores,
                                               num_walkers)
    try:
        np.random.seed(MOVE_SEED)
        old_rejections = old_sim.imp_move_randomly()
        np.random.seed(MOVE_SEED)
        new_rejections = new_sim.imp_move_randomly_data_parallel()

        if imp_num_cores is None:
            assert new_imp.num_cores == potential_num_cores
        else:
            assert new_imp.num_cores == imp_num_cores
        assert new_potential.num_cores == potential_num_cores
        expected_pool_num_cores = max(potential_num_cores, new_imp.num_cores)
        if potential_num_cores == 1 and new_imp.num_cores > 1:
            expected_pool_num_cores = new_imp.num_cores + 1
        assert new_potential.pool_num_cores == expected_pool_num_cores
        _assert_move_results_match(old_sim, new_sim, old_rejections, new_rejections)
    finally:
        _close_sim(old_sim, old_imp, old_potential)
        _close_sim(new_sim, new_imp, new_potential)


def test_imp_move_randomly_data_parallel_handles_more_cores_than_walkers(tmp_path, monkeypatch):
    trial_module = tmp_path / "trial_no_squeeze.py"
    trial_module.write_text(
        "\n".join([
            "import numpy as np",
            "",
            "ALPHA = 0.35",
            "",
            "def trial_harm(x):",
            "    return np.exp(-ALPHA * x[:, 0, 0] ** 2)",
            "",
            "def derivative(x):",
            "    deriv = -2.0 * ALPHA * x",
            "    sderiv = 4.0 * ALPHA ** 2 * x ** 2 - 2.0 * ALPHA",
            "    return deriv, sderiv",
            "",
        ]),
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    sim, imp_samp, potential = _dmc_sim(tmp_path,
                                        "small_walker_count",
                                        1,
                                        4,
                                        3,
                                        trial_directory=str(tmp_path),
                                        python_file=trial_module.name)
    try:
        np.random.seed(MOVE_SEED)
        rejections = sim.imp_move_randomly_data_parallel()

        assert 0 <= rejections <= 3
        assert sim.walkers.shape == (3, 1, 1)
        assert sim.f_x.shape == (3, 1, 1)
        assert sim.psi_1.shape == (3,)
        assert sim.psi_sec_der.shape == (3, 1, 1)
    finally:
        _close_sim(sim, imp_samp, potential)


def test_imp_move_randomly_data_parallel_pass_timestep_matches_existing_method(tmp_path, monkeypatch):
    trial_module = tmp_path / "trial_with_timestep.py"
    trial_module.write_text(
        "\n".join([
            "import numpy as np",
            "",
            "def _alpha(kwargs):",
            "    return 0.35 + 0.01 * kwargs['timestep']",
            "",
            "def trial_harm(x, kwargs):",
            "    alpha = _alpha(kwargs)",
            "    return np.exp(-alpha * np.squeeze(x) ** 2)",
            "",
            "def derivative(x, kwargs):",
            "    alpha = _alpha(kwargs)",
            "    deriv = -2.0 * alpha * x",
            "    sderiv = 4.0 * alpha ** 2 * x ** 2 - 2.0 * alpha",
            "    return deriv, sderiv",
            "",
        ]),
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    num_walkers = 67
    old_sim, old_imp, old_potential = _dmc_sim(tmp_path,
                                               "old_pass_timestep",
                                               1,
                                               3,
                                               num_walkers,
                                               trial_directory=str(tmp_path),
                                               python_file=trial_module.name,
                                               pass_timestep=True)
    new_sim, new_imp, new_potential = _dmc_sim(tmp_path,
                                               "new_pass_timestep",
                                               1,
                                               3,
                                               num_walkers,
                                               trial_directory=str(tmp_path),
                                               python_file=trial_module.name,
                                               pass_timestep=True)
    try:
        np.random.seed(MOVE_SEED)
        old_rejections = old_sim.imp_move_randomly()
        np.random.seed(MOVE_SEED)
        new_rejections = new_sim.imp_move_randomly_data_parallel()

        assert old_imp.ct == 2
        assert new_imp.ct == 2
        assert old_imp.trial_kwargs["timestep"] == new_imp.trial_kwargs["timestep"] == 2
        assert old_imp.deriv_kwargs["timestep"] == new_imp.deriv_kwargs["timestep"] == 2
        _assert_move_results_match(old_sim, new_sim, old_rejections, new_rejections)
    finally:
        _close_sim(old_sim, old_imp, old_potential)
        _close_sim(new_sim, new_imp, new_potential)


def test_use_data_parallel_imp_move_flag_routes_standard_importance_move(tmp_path, monkeypatch):
    sim, imp_samp, potential = _dmc_sim(tmp_path,
                                        "flag_route",
                                        1,
                                        1,
                                        8,
                                        use_data_parallel_imp_move=True)
    calls = []

    def fake_parallel_move():
        calls.append("parallel")
        return 3

    def fail_old_move():
        raise AssertionError("standard imp_move_randomly should not be called")

    monkeypatch.setattr(sim, "imp_move_randomly_data_parallel", fake_parallel_move)
    monkeypatch.setattr(sim, "imp_move_randomly", fail_old_move)
    try:
        assert sim.use_data_parallel_imp_move is True
        rejected = sim._imp_move_randomly_selected()
        assert rejected == 3
        assert calls == ["parallel"]
    finally:
        _close_sim(sim, imp_samp, potential)


def test_data_parallel_imp_move_flag_rejects_second_displacement(tmp_path):
    potential = _harmonic_potential(1)
    imp_samp = _importance_sampler(potential, imp_num_cores=1)
    start_structures = np.zeros((4, 1, 1))
    with pytest.raises(ValueError, match="use_data_parallel_imp_move"):
        pv.DMC_Sim(sim_name="invalid_parallel_second_type",
                   output_folder=str(tmp_path / "invalid_parallel_second_type"),
                   weighting="discrete",
                   num_walkers=4,
                   num_timesteps=1,
                   equil_steps=1,
                   chkpt_every=100,
                   wfn_every=100,
                   desc_wt_steps=1,
                   atoms=["O-H"],
                   delta_t=1,
                   potential=potential,
                   imp_samp=imp_samp,
                   imp_samp_oned=True,
                   second_impsamp_displacement=True,
                   use_data_parallel_imp_move=True,
                   start_structures=start_structures)
    imp_samp.mp_close()
    potential.mp_close()
