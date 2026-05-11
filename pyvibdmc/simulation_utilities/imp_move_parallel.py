import numpy as np


def _advance_imp_samp_timestep_once(imp_manager):
    if getattr(imp_manager, "pass_timestep", False):
        imp_manager.ct += 1
        if imp_manager.trial_kwargs is not None:
            imp_manager.trial_kwargs['timestep'] = imp_manager.ct
        if imp_manager.deriv_kwargs is not None:
            imp_manager.deriv_kwargs['timestep'] = imp_manager.ct


def _imp_drift_serial_chunk(impsamp, cds):
    if len(cds) == 0:
        return np.empty_like(cds), np.empty((0,), dtype=cds.dtype), np.empty_like(cds)
    deriv, psi_t, sderiv = impsamp.drift_serial(cds)
    return deriv, np.atleast_1d(psi_t), sderiv


def _imp_metropolis_chunk(sigma_trip, trial_x, trial_y, disp_x, disp_y, D_x, D_y, dt):
    psi_ratio = (trial_y / trial_x) ** 2
    term_1 = np.exp(-1 * (disp_x - disp_y - D_y * dt) ** 2 / (2 * sigma_trip ** 2))
    term_2 = np.exp(-1 * (disp_y - disp_x - D_x * dt) ** 2 / (2 * sigma_trip ** 2))
    accep = term_1 / term_2
    if accep.shape[-1] == 1:
        accep = np.atleast_1d(accep.squeeze() * psi_ratio.squeeze())
    else:
        accep = np.prod(np.prod(accep, axis=1), axis=1) * psi_ratio
    flipped = np.where(trial_x * trial_y <= 0)[0]
    accep[flipped] = 0.0
    return accep


def _imp_move_randomly_chunk(impsamp,
                             walker_coords,
                             f_x,
                             psi_1,
                             psi_sec_der,
                             disps,
                             randos,
                             vector_score,
                             inv_masses_trip,
                             sigma_trip,
                             delta_t,
                             excited_state_imp_samp,
                             masses):
    if len(walker_coords) == 0:
        if excited_state_imp_samp and vector_score is None:
            vector_score = np.empty((0,), dtype=masses.dtype)
        return walker_coords, f_x, psi_1, psi_sec_der, vector_score, 0

    d_x = inv_masses_trip * f_x

    if excited_state_imp_samp:
        d_x2 = d_x.copy()
        d_x2 += 1e-50
        ms = 1 / inv_masses_trip[:, :, 0]
        v2 = np.linalg.norm(d_x2, axis=2) ** 2
        factor = np.divide(-1 + np.sqrt(1 + 2 * ms * v2), ms * v2)
        sh = factor.shape
        d_x2 = np.broadcast_to(factor[:, :, None], (sh[0], sh[1], 3)) * d_x2
        if vector_score is None:
            numer = np.sum(np.linalg.norm(d_x2, axis=2) ** 2 * masses[None, :], axis=1)
            denom = np.sum(np.linalg.norm(d_x, axis=2) ** 2 * masses[None, :], axis=1)
            vector_score = np.sqrt(numer / denom)
        d_x = d_x2

    displaced_cds = walker_coords + disps + d_x * delta_t

    f_y, psi_2, psi_sec_der_disp = impsamp.drift_serial(displaced_cds)
    psi_2 = np.atleast_1d(psi_2)
    d_y = inv_masses_trip * f_y

    if excited_state_imp_samp:
        d_y2 = d_y.copy()
        d_y2 += 1e-50
        ms = 1 / inv_masses_trip[:, :, 0]
        v2 = np.linalg.norm(d_y2, axis=2) ** 2
        factor = np.divide(-1 + np.sqrt(1 + 2 * ms * v2), ms * v2)
        sh = factor.shape
        d_y2 = np.broadcast_to(factor[:, :, None], (sh[0], sh[1], 3)) * d_y
        numer = np.sum(np.linalg.norm(d_y2, axis=2) ** 2 * masses[None, :], axis=1)
        denom = np.sum(np.linalg.norm(d_y, axis=2) ** 2 * masses[None, :], axis=1)
        vector_score_new = np.sqrt(numer / denom)
        d_y = d_y2

    met_nums = _imp_metropolis_chunk(sigma_trip=sigma_trip,
                                     trial_x=psi_1,
                                     trial_y=psi_2,
                                     disp_x=walker_coords,
                                     disp_y=displaced_cds,
                                     D_x=d_x,
                                     D_y=d_y,
                                     dt=delta_t)
    accept = np.argwhere(met_nums > randos)

    walker_coords = walker_coords.copy()
    f_x = f_x.copy()
    psi_1 = psi_1.copy()
    psi_sec_der = psi_sec_der.copy()

    walker_coords[accept] = displaced_cds[accept]
    f_x[accept] = f_y[accept]
    psi_1[accept] = psi_2[accept]
    psi_sec_der[accept] = psi_sec_der_disp[accept]
    if excited_state_imp_samp:
        vector_score = vector_score.copy()
        vector_score[accept] = vector_score_new[accept]

    return walker_coords, f_x, psi_1, psi_sec_der, vector_score, len(walker_coords) - len(accept)


def imp_move_randomly_data_parallel(sim):
    """
    Importance-sampling random displacement split over the importance-sampling worker pool.
    """
    pool = getattr(sim.impsamp_manager, "pool", None)
    num_cores = getattr(sim.impsamp_manager, "num_cores", None)
    if pool is None or num_cores is None:
        return sim.imp_move_randomly()

    if (sim.f_x is None or sim.psi_1 is None):
        cds_chunks = np.array_split(sim._walker_coords, num_cores)
        drift_results = pool.starmap(_imp_drift_serial_chunk,
                                     [(sim.impsamp, cds) for cds in cds_chunks],
                                     chunksize=1)
        sim.f_x = np.concatenate([res[0] for res in drift_results])
        sim.psi_1 = np.concatenate([res[1] for res in drift_results])
        sim.psi_sec_der = np.concatenate([res[2] for res in drift_results])
        _advance_imp_samp_timestep_once(sim.impsamp_manager)

    disps = np.random.normal(0.0,
                             sim._sigmas,
                             size=np.shape(sim._walker_coords.transpose(0, 2, 1))).transpose(0, 2, 1)
    randos = np.random.random(size=len(sim._walker_coords))

    vector_score_chunks = [None for _ in range(num_cores)]
    if sim.excited_state_imp_samp and sim.vector_score is not None:
        vector_score_chunks = np.array_split(sim.vector_score, num_cores)

    chunk_args = zip(np.array_split(sim._walker_coords, num_cores),
                     np.array_split(sim.f_x, num_cores),
                     np.array_split(sim.psi_1, num_cores),
                     np.array_split(sim.psi_sec_der, num_cores),
                     np.array_split(disps, num_cores),
                     np.array_split(randos, num_cores),
                     vector_score_chunks)
    move_results = pool.starmap(
        _imp_move_randomly_chunk,
        [(sim.impsamp,
          coords,
          f_x,
          psi_1,
          psi_sec_der,
          disp,
          rando,
          vector_score,
          sim.inv_masses_trip,
          sim.sigma_trip,
          sim.delta_t,
          sim.excited_state_imp_samp,
          sim.masses) for coords, f_x, psi_1, psi_sec_der, disp, rando, vector_score in chunk_args],
        chunksize=1)
    _advance_imp_samp_timestep_once(sim.impsamp_manager)

    sim._walker_coords = np.concatenate([res[0] for res in move_results])
    sim.f_x = np.concatenate([res[1] for res in move_results])
    sim.psi_1 = np.concatenate([res[2] for res in move_results])
    sim.psi_sec_der = np.concatenate([res[3] for res in move_results])
    if sim.excited_state_imp_samp:
        sim.vector_score = np.concatenate([res[4] for res in move_results])

    num_rejections = sum(res[5] for res in move_results)
    sim.dt_factor = (len(sim._walker_coords) - num_rejections) / len(sim._walker_coords)
    return num_rejections
