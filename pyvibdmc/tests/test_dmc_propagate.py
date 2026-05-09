import os
import random
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest


PROPAGATE_SEED = 8675309
PROPAGATE_NUM_WALKERS = 100_000
PROPAGATE_RESULT_ENV = "PYVIBDMC_PROPAGATE_RESULT"
PROPAGATE_IMP_RESULT_ENV = "PYVIBDMC_PROPAGATE_IMP_RESULT"
PROPAGATE_REFERENCE_ENV = "PYVIBDMC_PROPAGATE_REFERENCE"
PROPAGATE_TEST_ENV = "PYVIBDMC_PROPAGATE_TEST"
METADATA_KEYS = {
    "python_version",
    "pyvibdmc_version",
    "potential_function",
    "potential_python_file",
    "potential_directory",
    "potential_manager",
    "potential_num_cores",
    "importance_sampling",
    "imp_samp_num_cores",
    "trial_function",
    "trial_python_file",
}


def load_propagate_result(result_path):
    """Load a saved single-step DMC_Sim.propagate() result."""
    with np.load(result_path, allow_pickle=False) as result:
        return {key: result[key] for key in result.files}


def _pyvibdmc_version(pyvibdmc_module):
    try:
        return metadata.version("pyvibdmc")
    except metadata.PackageNotFoundError:
        return getattr(pyvibdmc_module, "__version__", "unknown")


def assert_propagate_result_matches_reference(reference_path, test_path, rtol=0.0, atol=0.0):
    """Compare saved DMC result arrays; metadata is saved for inspection only."""
    reference = load_propagate_result(reference_path)
    test = load_propagate_result(test_path)
    result_keys = sorted(set(reference) - METADATA_KEYS)

    assert result_keys == sorted(set(test) - METADATA_KEYS)

    for key in result_keys:
        assert reference[key].shape == test[key].shape
        assert reference[key].dtype == test[key].dtype
        if np.issubdtype(reference[key].dtype, np.floating):
            np.testing.assert_allclose(test[key], reference[key], rtol=rtol, atol=atol)
        else:
            np.testing.assert_array_equal(test[key], reference[key])


def _harmonic_oscillator_potential(num_cores=2):
    import pyvibdmc as pv

    potential_directory = os.path.join(os.path.dirname(__file__), "../sample_potentials/PythonPots/")
    python_file = "harmonicOscillator1D.py"
    potential_function = "oh_stretch_harm"
    potential = pv.Potential(potential_function=potential_function,
                             python_file=python_file,
                             potential_directory=potential_directory,
                             num_cores=num_cores)
    potential_metadata = {
        "potential_function": potential_function,
        "potential_python_file": python_file,
        "potential_directory": potential_directory,
        "potential_manager": "Potential",
        "potential_num_cores": num_cores,
    }
    return potential, potential_metadata


def _importance_sampler(potential, imp_num_cores=None):
    import pyvibdmc as pv

    trial_directory = os.path.join(os.path.dirname(__file__), "../sample_potentials/PythonPots/")
    python_file = "harm_trial_wfn.py"
    trial_function = "trial_harm"
    imp_samp = pv.ImpSampManager(trial_function=trial_function,
                                 trial_directory=trial_directory,
                                 python_file=python_file,
                                 pot_manager=potential,
                                 imp_num_cores=imp_num_cores,
                                 deriv_function="derivative")
    return imp_samp, {
        "imp_samp_num_cores": imp_samp.num_cores,
        "trial_function": trial_function,
        "trial_python_file": python_file,
    }


def _save_propagate_result(result_path, dmc_sim, metadata_dict):
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": np.array(PROPAGATE_SEED, dtype=np.int64),
        "python_version": np.array(sys.version),
        "num_walkers": np.array(PROPAGATE_NUM_WALKERS, dtype=np.int64),
        "num_timesteps": np.array(dmc_sim.num_timesteps, dtype=np.int64),
        "cur_timestep": np.array(dmc_sim.cur_timestep, dtype=np.int64),
        "delta_t": np.array(dmc_sim.delta_t, dtype=np.float64),
        "masses": dmc_sim.masses,
        "sigmas": dmc_sim._sigmas,
        "walker_coords": dmc_sim.walkers,
        "walker_pots": dmc_sim._walker_pots,
        "vref_vs_tau": dmc_sim._vref_vs_tau,
        "pop_vs_tau": dmc_sim._pop_vs_tau,
        "vref": np.array(dmc_sim._vref, dtype=np.float64),
    }
    result.update({key: np.array(val) for key, val in metadata_dict.items()})
    if dmc_sim.impsamp_manager is not None:
        result["eff_ts"] = dmc_sim.eff_ts
    np.savez(result_path, **result)
    return result_path


def _single_step_dmc(result_path,
                     output_folder,
                     importance_sampling=False,
                     potential_num_cores=2,
                     imp_num_cores=None):
    import pyvibdmc as pv

    random.seed(PROPAGATE_SEED)
    np.random.seed(PROPAGATE_SEED)
    potential, potential_metadata = _harmonic_oscillator_potential(num_cores=potential_num_cores)
    imp_samp = None
    imp_metadata = {}
    if importance_sampling:
        imp_samp, imp_metadata = _importance_sampler(potential, imp_num_cores=imp_num_cores)

    myDMC = pv.DMC_Sim(sim_name="propagate_once_test",
                       output_folder=str(output_folder),
                       weighting="discrete",
                       num_walkers=PROPAGATE_NUM_WALKERS,
                       num_timesteps=1,
                       equil_steps=1,
                       chkpt_every=100,
                       wfn_every=100,
                       desc_wt_steps=1,
                       atoms=["O-H"],
                       delta_t=1,
                       potential=potential,
                       second_impsamp_displacement=importance_sampling,
                       imp_samp=imp_samp,
                       imp_samp_oned=importance_sampling,
                       log_every=1,
                       start_structures=np.zeros((1, 1, 1)),
                       cur_timestep=0)

    try:
        myDMC.propagate()
        metadata_dict = {
            "pyvibdmc_version": _pyvibdmc_version(pv),
            "importance_sampling": importance_sampling,
            **potential_metadata,
            **imp_metadata,
        }
        _save_propagate_result(result_path, myDMC, metadata_dict)
    finally:
        myDMC._logger.fl.close()
        if imp_samp is not None:
            imp_samp.mp_close()
        potential.mp_close()

    return result_path


def test_dmc_sim_propagate_once_saves_deterministic_result(tmp_path):
    result_path = Path(os.environ.get(PROPAGATE_RESULT_ENV, tmp_path / "dmc_propagate_once.npz"))
    output_folder = tmp_path / "dmc_propagate_output"

    saved_result = _single_step_dmc(result_path, output_folder)
    result = load_propagate_result(saved_result)

    assert saved_result.is_file()
    assert int(result["seed"]) == PROPAGATE_SEED
    assert int(result["num_walkers"]) == PROPAGATE_NUM_WALKERS
    assert int(result["num_timesteps"]) == 1
    assert result["walker_coords"].shape[1:] == (1, 1)
    assert result["walker_pots"].ndim == 1
    assert result["vref_vs_tau"].shape == (1,)
    assert result["pop_vs_tau"].shape == (1,)

    reference_path = os.environ.get(PROPAGATE_REFERENCE_ENV)
    if reference_path is not None:
        assert_propagate_result_matches_reference(reference_path, saved_result)


def test_dmc_sim_propagate_once_with_importance_sampling_saves_deterministic_result(tmp_path):
    result_path = Path(os.environ.get(PROPAGATE_IMP_RESULT_ENV, tmp_path / "dmc_propagate_once_impsamp.npz"))
    output_folder = tmp_path / "dmc_propagate_impsamp_output"

    saved_result = _single_step_dmc(result_path, output_folder, importance_sampling=True)
    result = load_propagate_result(saved_result)

    assert saved_result.is_file()
    assert int(result["seed"]) == PROPAGATE_SEED
    assert int(result["num_walkers"]) == PROPAGATE_NUM_WALKERS
    assert int(result["num_timesteps"]) == 1
    assert bool(result["importance_sampling"])
    assert result["walker_coords"].shape[1:] == (1, 1)
    assert result["walker_pots"].ndim == 1
    assert result["vref_vs_tau"].shape == (1,)
    assert result["pop_vs_tau"].shape == (1,)
    assert result["eff_ts"].shape == (1,)


@pytest.mark.parametrize(("potential_num_cores", "imp_num_cores"), [(4, 1), (1, 4)])
def test_dmc_sim_propagate_once_with_importance_sampling_cpu_allocations(tmp_path,
                                                                         potential_num_cores,
                                                                         imp_num_cores):
    result_path = tmp_path / f"dmc_propagate_impsamp_pot{potential_num_cores}_imp{imp_num_cores}.npz"
    output_folder = tmp_path / f"dmc_propagate_impsamp_pot{potential_num_cores}_imp{imp_num_cores}_output"

    saved_result = _single_step_dmc(result_path,
                                    output_folder,
                                    importance_sampling=True,
                                    potential_num_cores=potential_num_cores,
                                    imp_num_cores=imp_num_cores)
    result = load_propagate_result(saved_result)

    assert int(result["potential_num_cores"]) == potential_num_cores
    assert int(result["imp_samp_num_cores"]) == imp_num_cores
    assert bool(result["importance_sampling"])
    assert result["walker_coords"].shape[1:] == (1, 1)
    assert result["walker_pots"].ndim == 1
    assert result["eff_ts"].shape == (1,)


def test_compare_saved_dmc_sim_propagate_result_to_reference_from_env():
    reference_path = os.environ.get(PROPAGATE_REFERENCE_ENV)
    test_path = os.environ.get(PROPAGATE_TEST_ENV)
    if reference_path is None or test_path is None:
        pytest.skip("Set reference and test result environment variables to compare saved files.")

    assert_propagate_result_matches_reference(reference_path, test_path)
