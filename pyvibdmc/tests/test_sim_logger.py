from pyvibdmc.simulation_utilities.Constants import Constants
from pyvibdmc.simulation_utilities.sim_logger import SimLogger


def test_write_vref_labels_and_converts_reference_energy(tmp_path):
    log_path = tmp_path / "simulation.log"
    logger = SimLogger(log_path, overwrite=True)

    logger.write_vref(Constants.atomic_units["wavenumbers"])
    logger.fl.close()

    assert log_path.read_text() == "\tReference energy (Vref): 1.0 cm-1\n"
