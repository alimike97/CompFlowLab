import os
import copy
import numpy as np
import cantera as ct

from compflowlab.utils import reshape_func
from compflowlab.boundary_condition import bc_func

#####################################################################
# Ensemble Kalman filter in tau space (twin experiment)             #
#                                                                   #
# Every ensemble member is a full FOM trajectory. At assimilation   #
# iterations the members are updated with a stochastic EnKF, either #
#   - enkf_space = physical : state = primitive profiles on the grid#
#   - enkf_space = tau      : each profile is transformed, region by#
#     region (one feature per region), to positions and values at   #
#     matched cumulative-variation levels (tau), updated there and  #
#     transformed back.                                             #
# The observations come from one extra member (the "truth", by      #
# default the median member) that is advanced without assimilation. #
#                                                                   #
# Supported cases                                                   #
#   - shock tube  : gas_model = Air, two-region IC; members differ  #
#                   in diaphragm location and left pressure         #
#   - periodic    : any IC profile (ic_path) on a periodic domain,  #
#     (e.g. RDE)    gas_model = Air or one-step chemistry (.inp);   #
#                   members are the IC shifted around the domain    #
#                   and with a scaled wave amplitude. The tau       #
#                   transform works on the ring (see tau section).  #
#####################################################################

################################ gas / domain helpers ################################

def gas_constant(solver_param, state):

    # specific gas constant R = p / (rho T) of the current gas model
    if solver_param['gas_model'] == 'Air':

        return ct.gas_constant / state.get('MW_gas', 28.97)

    return state['gas_lookup_table']['R_univ'] / state['gas_lookup_table']['MW']

def domain_period(solver_param):

    # length of the ring for a periodic domain (cell centers x[0] .. x[-1] plus one cell), else None
    if not np.all(solver_param['bc_data'] == 'periodic'):

        return None

    x = solver_param['x']

    return x[-1] - x[0] + (x[1] - x[0])

################################ tau transformation ################################

def compute_tau(xs, fs, regularization=1e-8):

    # normalized cumulative variation (strictly increasing thanks to a tiny dx term)
    increments = np.abs(np.diff(fs)) + regularization * np.diff(xs) / (xs[-1] - xs[0])
    tau        = np.concatenate([[0.0], np.cumsum(increments)])

    return tau / tau[-1]

def get_segment(x, f, left, right):

    # profile between two physical boundaries (exact boundary values interpolated)
    interior = (x > left) & (x < right)
    xs       = np.concatenate([[left], x[interior], [right]])

    return xs, np.interp(xs, x, f)

def find_features(g, g_smooth, tol, feature_frac):

    # features: connected stretches where the smoothed gradient exceeds tol (start/end
    # inclusive, on cell faces) that carry at least feature_frac of the total variation
    change = np.diff(np.concatenate([[0], (g_smooth > tol).astype(int), [0]]))
    starts = np.where(change ==  1)[0]
    ends   = np.where(change == -1)[0] - 1

    variation = np.array([g[s:e + 1].sum() for s, e in zip(starts, ends)])
    keep      = variation >= feature_frac * g.sum()

    return starts[keep], ends[keep]

def find_cuts(x, f, n_jumps, flat_tol=1e-3, feature_frac=0.01, smooth_cells=5):

    #####################################################################
    # Cut positions so that each region holds ONE feature:              #
    #   - one cut in the middle of the flat stretch before the first    #
    #     jump (separates the rarefaction from the first jump)          #
    #   - one cut in the middle of the flat stretch between each pair   #
    #     of neighbouring jumps                                         #
    # On a finite-volume solution the jumps are smeared and, early on,  #
    # a weak shock can be less steep than the rarefaction, so the       #
    # largest gradient does not reliably mark a jump. Instead the       #
    # profile is split into features: connected stretches where the    #
    # smoothed gradient exceeds flat_tol*range (stretches carrying less #
    # than feature_frac of the total variation are numerical wiggles    #
    # and count as flat). The cuts are the middles of the flat gaps     #
    # between consecutive features (the widest n_jumps gaps if there    #
    # are more). If features touch (e.g. a smeared contact right next   #
    # to the rarefaction tail) the flatness threshold is raised step by #
    # step until they separate. Without enough gaps (e.g. the initial   #
    # step) the middle of the flat stretch left of the first feature is #
    # used, then the one right of the last.                             #
    #####################################################################

    g        = np.abs(np.diff(f))
    f_range  = max(f.max() - f.min(), 1e-300)

    # smoothed gradient on the cell faces, so wiggles do not split a feature
    g_smooth = np.convolve(g, np.ones(smooth_cells) / smooth_cells, mode='same')
    x_face   = 0.5 * (x[:-1] + x[1:])

    starts, ends = find_features(g, g_smooth, flat_tol * f_range, feature_frac)

    if len(starts) == 0:

        # a flat profile: equally spaced cuts
        return list(np.linspace(x[0], x[-1], n_jumps + 2)[1:-1])

    # raise the flatness threshold until there is a gap for every cut
    for factor in [3, 10, 30]:

        if len(starts) - 1 >= n_jumps:
            break

        starts_try, ends_try = find_features(g, g_smooth, factor * flat_tol * f_range, feature_frac)

        if len(starts_try) - 1 > len(starts) - 1:
            starts, ends = starts_try, ends_try

    # flat gaps between consecutive features, widest first
    gaps = [(ends[j] + 1, starts[j + 1] - 1) for j in range(len(starts) - 1)]
    gaps = sorted(gaps, key=lambda gap: x_face[gap[1]] - x_face[gap[0]], reverse=True)

    cuts = [0.5 * (x_face[a] + x_face[b]) for a, b in gaps[:n_jumps]]

    # not enough gaps: the flat stretch left of the first feature, then right of the last
    if len(cuts) < n_jumps:
        cuts.append(0.5 * (x[0] + x_face[starts[0]]))
    if len(cuts) < n_jumps:
        cuts.append(0.5 * (x_face[ends[-1]] + x[-1]))

    return sorted(cuts)

def find_cuts_periodic(x, f, n_jumps, period, flat_tol=1e-3, feature_frac=0.01, smooth_cells=5):

    #####################################################################
    # Periodic version of find_cuts. On a ring there is no left/right   #
    # end, so the profile is opened at an ANCHOR: the middle of the     #
    # widest flat gap between features (in an RDE: the fresh-reactant  #
    # fill region ahead of the detonation). The n_jumps further cuts    #
    # are the middles of the next widest gaps, so a ring profile gets   #
    # n_jumps + 1 regions, like a line profile with n_jumps cuts.       #
    # Face i sits between cell i and i+1 (face n-1 wraps to cell 0);    #
    # positions are returned unwrapped (anchor in [x0, x0 + period)).   #
    #####################################################################

    n       = len(x)
    h       = period / n
    g       = np.abs(np.diff(np.append(f, f[0])))
    f_range = max(f.max() - f.min(), 1e-300)

    # circular smoothing of the gradient
    pad      = smooth_cells
    g_smooth = np.convolve(np.concatenate([g[-pad:], g, g[:pad]]), np.ones(smooth_cells) / smooth_cells, mode='same')[pad:-pad]

    def face_pos(i):

        return x[0] + (i + 0.5) * h

    def features(tol):

        # features on the ring (start/end face indices, end may run past n-1)
        mask = g_smooth > tol

        if mask.all() or not mask.any():

            return np.array([], dtype=int), np.array([], dtype=int)

        r            = int(np.argmin(mask))     # open the ring at a flat face
        starts, ends = find_features(np.roll(g, -r), np.roll(g_smooth, -r), tol, feature_frac)

        return starts + r, ends + r

    starts, ends = features(flat_tol * f_range)

    # raise the flatness threshold until there is a gap for the anchor and every cut
    for factor in [3, 10, 30]:

        if len(starts) >= n_jumps + 1:
            break

        starts_try, ends_try = features(factor * flat_tol * f_range)

        if len(starts_try) > len(starts):
            starts, ends = starts_try, ends_try

    if len(starts) == 0:

        # flat (or nowhere flat) profile: anchor at the smallest gradient, equally spaced cuts
        anchor = face_pos(int(np.argmin(g_smooth)))
        cuts   = [anchor + period * (j + 1) / (n_jumps + 1) for j in range(n_jumps)]

    else:

        # one flat gap after every feature (the last one wraps to the first feature), widest first
        m    = len(starts)
        gaps = [(ends[j] + 1, starts[(j + 1) % m] - 1 + (n if j == m - 1 else 0)) for j in range(m)]
        gaps = sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)
        mids = [0.5 * (face_pos(a) + face_pos(b)) for a, b in gaps]

        anchor = mids[0]
        cuts   = mids[1:n_jumps + 1]

        # not enough gaps: equally spaced cuts after the anchor
        while len(cuts) < n_jumps:
            cuts.append(anchor + period * (len(cuts) + 1) / (n_jumps + 1))

    anchor = x[0] + (anchor - x[0]) % period

    return anchor, cuts

def tau_sample(x, f, edges, s_levels, regularization):

    # region by region: positions and values at the tau levels -> [all positions, all values]
    xs_all, fs_all = [], []

    for r in range(len(edges) - 1):

        xs, fs = get_segment(x, f, edges[r], edges[r + 1])
        tau    = compute_tau(xs, fs, regularization)

        xs_all.append(np.interp(s_levels, tau, xs))     # position at each local tau
        fs_all.append(np.interp(s_levels, tau, fs))     # value at each local tau

    return np.concatenate([np.concatenate(xs_all), np.concatenate(fs_all)])

def to_tau_state(x, f, n_jumps, s_levels, regularization=1e-8, flat_tol=1e-3, feature_frac=0.01, smooth_cells=5):

    # line profile -> state vector [positions of all regions, values of all regions]
    edges = [x[0]] + find_cuts(x, f, n_jumps, flat_tol, feature_frac, smooth_cells) + [x[-1]]

    return tau_sample(x, f, edges, s_levels, regularization)

def to_tau_state_periodic(x, f, anchor, cuts, period, s_levels, regularization=1e-8):

    #####################################################################
    # ring profile -> state vector, opened at the (unwrapped) anchor:   #
    # the regions run from anchor to anchor + period, so the positions  #
    # are continuous and may leave [x0, x0 + period)                    #
    #####################################################################

    cuts  = sorted(anchor + (c - anchor) % period for c in cuts)
    edges = [anchor] + cuts + [anchor + period]

    # the profile over three periods starting at or before the anchor
    base = x + np.floor((anchor - x[0]) / period) * period
    xp   = np.concatenate([base, base + period, [base[0] + 2 * period]])
    fp   = np.concatenate([f, f, [f[0]]])

    return tau_sample(xp, fp, edges, s_levels, regularization)

def from_tau_state(z, x, period=None):

    # state vector -> profile on the grid x
    half    = len(z) // 2
    x_curve = np.maximum.accumulate(z[:half])           # positions must not go backwards
    f_curve = z[half:]

    if period is None:

        return np.interp(x, x_curve, f_curve)

    # ring: one period starting at the anchor (the closing point anchor + period is the anchor again)
    keep = x_curve < x_curve[0] + period

    return np.interp(x, x_curve[keep], f_curve[keep], period=period)

def tau_state_size(solver_param, n_jumps):

    # n_jumps cuts (+ the anchor on a ring) -> n_jumps + 1 regions, each with n_levels positions and n_levels values
    return 2 * (n_jumps + 1) * solver_param['tau_n_levels']

def circular_mean(positions, x0, period):

    angle = 2 * np.pi * (np.asarray(positions) - x0) / period

    return x0 + period * (np.angle(np.mean(np.exp(1j * angle))) / (2 * np.pi)) % period

################################ DA state vector ################################

def members_to_da_state(solver_param, prim_members):

    #####################################################################
    # prim_members : (N, num_prim_var, cell_number) interior primitives #
    # returns Z    : (N, n) states; each DA variable is one block       #
    #####################################################################

    x = solver_param['x']

    if solver_param['enkf_space'] == 'physical':

        return np.array([np.concatenate([prim[v] for v in solver_param['da_state_var_index']]) for prim in prim_members])

    period  = solver_param['tau_period']
    options = (solver_param['tau_flat_tol'], solver_param['tau_feature_frac'], solver_param['tau_smooth_cells'])
    blocks  = []                                         # blocks[variable][member]

    for v, n_jumps in zip(solver_param['da_state_var_index'], solver_param['tau_n_jumps']):

        if period is None:

            blocks.append([to_tau_state(x, prim[v], n_jumps, solver_param['tau_s_levels'], solver_param['tau_regularization'], *options)
                           for prim in prim_members])

            continue

        # ring: the anchors are unwrapped to the copy closest to the ensemble's circular mean anchor,
        # so members whose waves sit on either side of x0 still line up
        anchor_cuts = [find_cuts_periodic(x, prim[v], n_jumps, period, *options) for prim in prim_members]
        reference   = circular_mean([a for a, _ in anchor_cuts], x[0], period)

        block_v = []

        for prim, (anchor, cuts) in zip(prim_members, anchor_cuts):

            anchor = anchor + period * np.round((reference - anchor) / period)

            block_v.append(to_tau_state_periodic(x, prim[v], anchor, cuts, period, solver_param['tau_s_levels'], solver_param['tau_regularization']))

        blocks.append(block_v)

    return np.array([np.concatenate([block_v[k] for block_v in blocks]) for k in range(len(prim_members))])

def da_state_to_profiles(solver_param, z):

    # one DA state vector -> {prim var index: profile on the grid}
    x        = solver_param['x']
    profiles = {}
    start    = 0

    for v, n_jumps in zip(solver_param['da_state_var_index'], solver_param['tau_n_jumps']):

        if solver_param['enkf_space'] == 'physical':

            size        = solver_param['cell_number']
            profiles[v] = z[start:start + size].copy()

        else:

            size        = tau_state_size(solver_param, n_jumps)
            profiles[v] = from_tau_state(z[start:start + size], x, solver_param['tau_period'])

        start = start + size

    return profiles

def observation_operator(solver_param, profiles):

    # every observed variable at every sensor location (linear interpolation on the grid, wrapping on a ring)
    x = solver_param['x']

    return np.concatenate([np.interp(solver_param['obs_x_locations'], x, profiles[v], period=solver_param['tau_period'])
                           for v in solver_param['obs_state_var_index']])

################################ EnKF ################################

def enkf(Z, HZ, y, obs_std, rng):

    #####################################################################
    # stochastic EnKF (works for any state and any observation operator)#
    # Z: (N, n) states, HZ: (N, m) predicted observations, y: (m,)      #
    # obs_std: (m,) observation standard deviation                     #
    #####################################################################

    N  = Z.shape[0]
    A  = Z  - Z.mean(axis=0)
    HA = HZ - HZ.mean(axis=0)

    PHt = A.T  @ HA / (N - 1)                                  # cov(state, obs)
    S   = HA.T @ HA / (N - 1) + np.diag(obs_std**2)            # innovation covariance

    # perturbed observations, one set per member
    Y_pert = y + obs_std * rng.standard_normal((N, len(y)))

    # Z_a = Z + (Y_pert - HZ) S^-1 PHt^T   (S is symmetric)
    return Z + np.linalg.solve(S, (Y_pert - HZ).T).T @ PHt.T

################################ realizability ################################

def make_realizable(solver_param, state, prim_int):

    #####################################################################
    # neither the physical nor the tau update knows about realizability #
    # or about the equation of state: positive density and pressure,   #
    # mass fractions in [0, 1], and (if pressure is a DA variable) the #
    # temperature that matches the updated density and pressure        #
    #####################################################################

    floor       = 1e-6
    prim_int[0] = np.maximum(prim_int[0], floor)   # density
    prim_int[2] = np.maximum(prim_int[2], floor)   # pressure

    if prim_int.shape[0] > 4:

        prim_int[4:] = np.clip(prim_int[4:], 0.0, 1.0)

    if 2 in solver_param['da_state_var_index']:

        prim_int[3] = prim_int[2] / prim_int[0] / gas_constant(solver_param, state)

    return prim_int

################################ ensemble statistics ################################

def ensemble_mean_prim(solver_param, state, prim_members):

    #####################################################################
    # mean primitive profile of the ensemble; in tau mode the DA vars   #
    # use the tau mean (mean of the tau states, mapped back) so shocks  #
    # stay sharp, the others use the plain arithmetic mean              #
    #####################################################################

    mean_prim = np.mean(prim_members, axis=0)

    if solver_param['enkf_space'] == 'tau':

        profiles = da_state_to_profiles(solver_param, members_to_da_state(solver_param, prim_members).mean(axis=0))

        for v in profiles:

            mean_prim[v] = profiles[v]

    if mean_prim.shape[0] > 4:

        mean_prim[4:] = np.clip(mean_prim[4:], 0.0, 1.0)

    # temperature consistent with the (mean) density and pressure
    mean_prim[3] = mean_prim[2] / mean_prim[0] / gas_constant(solver_param, state)

    return mean_prim

def interior_prim(solver_param, state_k):

    prim_full = reshape_func.results_solver2user_converter(solver_param['num_prim_var'], solver_param['cell_number'], state_k['Q_prim'])

    return prim_full[:, 2:-2].copy()

def heat_release(solver_param, state, physics, prim_int):

    # heat release of a primitive profile (reacting gas models only; same formula as the FOM)
    tmp = {'gas_lookup_table': state['gas_lookup_table']}

    tmp['Q_prim']    = reshape_func.solver_add_ghost(solver_param['cell_number'], solver_param['num_prim_var'], prim_int.ravel())
    tmp              = physics.prim2cons_converter(solver_param, tmp)
    tmp              = bc_func.update_ghost_cell(solver_param, tmp)
    tmp              = physics.cache_cantera(solver_param, tmp)
    tmp['d_flux_dx'] = np.zeros_like(tmp['Q_cons'])
    tmp              = physics.source_calculator(solver_param, {}, tmp)

    return tmp['heat_release'][2:-2]

def with_heat_release(solver_param, state, physics, prim_int):

    # saved results follow the FOM layout: reacting cases get the heat release as an extra last row
    if solver_param['gas_model'] == 'Air':

        return prim_int

    return np.vstack((prim_int, heat_release(solver_param, state, physics, prim_int)))

################################ initialization ################################

def build_member(solver_param, state, physics, prim_int):

    # an independent copy of the state dict holding this primitive profile (consistent cons/prim)
    state_k = copy.deepcopy(state)

    state_k['Q_prim'] = reshape_func.solver_add_ghost(solver_param['cell_number'], solver_param['num_prim_var'], prim_int.ravel())
    state_k = bc_func.update_ghost_cell(solver_param, state_k)
    state_k = physics.prim2cons_converter(solver_param, state_k)
    state_k = physics.cons2prim_converter(solver_param, state_k)
    state_k = bc_func.update_ghost_cell(solver_param, state_k)

    return state_k

def shock_tube_members(solver_param, state, rng):

    # uncertain diaphragm location and left pressure (uniform)
    num_ensmbl = solver_param['num_ensmbl']
    x0_ens     = rng.uniform(*solver_param['ens_x_diaphragm_range'], num_ensmbl)
    pL_ens     = rng.uniform(*solver_param['ens_press_left_range'],  num_ensmbl)

    rho_L, rho_R = float(eval(solver_param['ic_data'][0][2])), float(eval(solver_param['ic_data'][1][2]))
    u_L,   u_R   = float(eval(solver_param['ic_data'][0][3])), float(eval(solver_param['ic_data'][1][3]))
    p_R          = float(eval(solver_param['ic_data'][1][4]))

    x = solver_param['x']

    prims = []

    for k in range(num_ensmbl):

        left = x < x0_ens[k]

        rho_profile   = np.where(left, rho_L,     rho_R)
        vel_profile   = np.where(left, u_L,       u_R)
        press_profile = np.where(left, pL_ens[k], p_R)
        temp_profile  = press_profile / rho_profile / gas_constant(solver_param, state)

        prims.append(np.vstack((rho_profile, vel_profile, press_profile, temp_profile)))

    return prims, {'x_diaphragm': x0_ens, 'press_left': pL_ens}

def periodic_members(solver_param, state, rng):

    #####################################################################
    # members from the IC profile on a ring (e.g. an RDE detonation):   #
    #   - wave strength: the deviation of rho, u, p, T from their       #
    #     minimum (the unburnt fill state) is scaled by an amplitude    #
    #   - wave position: the profile is shifted around the ring (whole #
    #     cells, so the front stays as sharp as in the IC)              #
    # p is then made consistent with the equation of state (build_member)
    #####################################################################

    if solver_param['tau_period'] is None:

        raise ValueError('TAUENKF with an IC profile (ic_path) needs a periodic domain (all BCs periodic)')

    num_ensmbl = solver_param['num_ensmbl']
    shift_ens  = rng.uniform(*solver_param['ens_shift_range'],     num_ensmbl)
    amp_ens    = rng.uniform(*solver_param['ens_amplitude_range'], num_ensmbl)

    base = interior_prim(solver_param, state)
    dx   = solver_param['x'][1] - solver_param['x'][0]

    prims = []

    for k in range(num_ensmbl):

        prim = base.copy()

        for v in range(4):

            prim[v] = base[v].min() + amp_ens[k] * (base[v] - base[v].min())

        prims.append(np.roll(prim, int(np.round(shift_ens[k] / dx)), axis=1))

    return prims, {'shift': shift_ens, 'amplitude': amp_ens}

def init_ensemble(solver_param, state, physics):

    #####################################################################
    # Draws one IC per member and gives every member its own            #
    # independent copy of the state dict. The truth member is taken out #
    # of the forecast ensemble and provides the observations.           #
    #####################################################################

    rng = np.random.default_rng(solver_param['ens_seed'])

    if isinstance(solver_param['ic_data'], np.ndarray):

        prims, params = periodic_members(solver_param, state, rng)

    elif len(solver_param['ic_data']) == 2 and solver_param['gas_model'] == 'Air':

        prims, params = shock_tube_members(solver_param, state, rng)

    else:

        raise ValueError('TAUENKF expects a two-region shock tube IC (gas_model = Air) or an IC profile (ic_path) on a periodic domain')

    members = [build_member(solver_param, state, physics, prim) for prim in prims]

    # truth member: unless given, the median member, i.e. the one whose parameters are
    # closest to the ensemble median (each parameter scaled by its sampling range)
    k_truth = solver_param['obs_member_index']

    if k_truth is None:

        values  = np.column_stack(list(params.values()))
        scale   = np.maximum(np.ptp(values, axis=0), 1e-300)
        k_truth = int(np.argmin(np.sum(((values - np.median(values, axis=0)) / scale)**2, axis=1)))
        solver_param['obs_member_index'] = k_truth

    # take the truth member out of the forecast ensemble
    state['truth']      = members.pop(k_truth)
    state['ensemble']   = members
    state['ens_params'] = {name: np.delete(value, k_truth) for name, value in params.items()}
    state['ens_params'].update({'truth_' + name: value[k_truth] for name, value in params.items()})

    truth_text = ', '.join(f"{name} = {value[k_truth]:.4g}" for name, value in params.items())

    print(f"TAUENKF: {len(members)} forecast members, truth = member {k_truth} ({truth_text}), "
          f"enkf_space = {solver_param['enkf_space']}, periodic = {solver_param['tau_period'] is not None}")

    # observation noise generator (kept separate so physical/tau runs see the same observations)
    state['obs_rng']  = np.random.default_rng(solver_param['enkf_seed'])
    state['enkf_rng'] = np.random.default_rng(solver_param['enkf_seed'] + 1)

    np.save(os.path.join(solver_param['dir_results'], 'ens_params.npy'), state['ens_params'], allow_pickle=True)

    return state

################################ time stepping ################################

def advance_member(solver_param, state_k, physics, time_integration, rom_param):

    # one FOM step of a single trajectory
    state_k = physics.residual_calculator(solver_param, rom_param, state_k)
    state_k = time_integration.advance_time(solver_param, rom_param, state_k, physics)

    if solver_param['injection']:

        state_k = physics.injection_correction(solver_param, state_k)

    state_k = physics.cons2prim_converter(solver_param, state_k)
    state_k = bc_func.update_ghost_cell(solver_param, state_k)
    state_k = physics.prim2cons_converter(solver_param, state_k)

    return state_k

def set_member_prim(solver_param, state_k, prim_int, physics):

    prim_int = make_realizable(solver_param, state_k, prim_int)

    state_k['Q_prim'] = reshape_func.solver_add_ghost(solver_param['cell_number'], solver_param['num_prim_var'], prim_int.ravel())
    state_k = bc_func.update_ghost_cell(solver_param, state_k)
    state_k = physics.prim2cons_converter(solver_param, state_k)
    state_k = physics.cons2prim_converter(solver_param, state_k)   # consistent temperature / pressure
    state_k = bc_func.update_ghost_cell(solver_param, state_k)

    return state_k

def assimilate(solver_param, state, physics):

    prim_members = np.array([interior_prim(solver_param, state_k) for state_k in state['ensemble']])
    prim_truth   = interior_prim(solver_param, state['truth'])

    # observations: the truth member at the sensors plus noise
    y_true = observation_operator(solver_param, {v: prim_truth[v] for v in solver_param['obs_state_var_index']})
    y_obs  = y_true + solver_param['obs_std_vec'] * state['obs_rng'].standard_normal(len(y_true))

    # forecast (prior) state and predicted observations (always observed through the profile)
    Z_prior  = members_to_da_state(solver_param, prim_members)
    HZ_prior = np.array([observation_operator(solver_param, da_state_to_profiles(solver_param, z)) for z in Z_prior])

    # analysis
    Z_analysis = enkf(Z_prior, HZ_prior, y_obs, solver_param['obs_std_vec'], state['enkf_rng'])

    # back to the members
    for k, z in enumerate(Z_analysis):

        prim_int = prim_members[k].copy()

        for v, profile in da_state_to_profiles(solver_param, z).items():

            prim_int[v] = profile

        state['ensemble'][k] = set_member_prim(solver_param, state['ensemble'][k], prim_int, physics)

    prim_posterior = np.array([interior_prim(solver_param, state_k) for state_k in state['ensemble']])

    # record this assimilation cycle
    dir_da     = os.path.join(solver_param['dir_results'], 'da')
    save_title = str(solver_param['iter']) + 'iteration'

    np.save(os.path.join(dir_da, f"{save_title}_prior_prim.npy")         , prim_members)
    np.save(os.path.join(dir_da, f"{save_title}_posterior_prim.npy")     , prim_posterior)
    np.save(os.path.join(dir_da, f"{save_title}_prior_mean_prim.npy")    , ensemble_mean_prim(solver_param, state, prim_members))
    np.save(os.path.join(dir_da, f"{save_title}_posterior_mean_prim.npy"), ensemble_mean_prim(solver_param, state, prim_posterior))
    np.save(os.path.join(dir_da, f"{save_title}_truth_prim.npy")         , prim_truth)
    np.save(os.path.join(dir_da, f"{save_title}_obs.npy")                , y_obs)

    print(f"TAUENKF: assimilated {len(y_obs)} observations at iteration {solver_param['iter']}")

    return state

################################ storing ################################

def prepare_to_store(solver_param, state, physics):

    # the ensemble mean is reported as the main state (used for visualization);
    # reacting cases get a heat release row, like the FOM results
    prim_members = np.array([interior_prim(solver_param, state_k) for state_k in state['ensemble']])
    mean_prim    = ensemble_mean_prim(solver_param, state, prim_members)

    state['mean_prim']         = mean_prim
    state['prim_results_save'] = with_heat_release(solver_param, state, physics, mean_prim)
    state['prim_members_save'] = np.array([with_heat_release(solver_param, state, physics, prim) for prim in prim_members])
    state['truth_prim_save']   = with_heat_release(solver_param, state, physics, interior_prim(solver_param, state['truth']))

    return state

def results_recorder(solver_param, state):

    dir_results = solver_param['dir_results']
    save_title  = str(solver_param['iter']) + 'iteration'

    np.save(os.path.join(dir_results, 'mean'    , f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results, 'ensemble', f"{save_title}_prim.npy"), state['prim_members_save'])
    np.save(os.path.join(dir_results, 'truth'   , f"{save_title}_prim.npy"), state['truth_prim_save'])

def advance_one_time_step(solver_param, state, physics, time_integration, rom_param=None):

    # build the ensemble (and the truth trajectory) exactly once
    if 'ensemble' not in state:

        solver_param['tau_period'] = domain_period(solver_param)

        state = init_ensemble(solver_param, state, physics)

    iter = solver_param['iter']

    # forecast: advance the truth and every member one FOM step
    state['truth'] = advance_member(solver_param, state['truth'], physics, time_integration, rom_param)

    for k in range(len(state['ensemble'])):

        solver_param['ensmbl'] = k
        state['ensemble'][k]   = advance_member(solver_param, state['ensemble'][k], physics, time_integration, rom_param)

    # analysis
    if iter >= solver_param['da_start_iter'] and (iter - solver_param['da_start_iter']) % solver_param['da_interval'] == 0:

        state = assimilate(solver_param, state, physics)

    # ensemble mean (tau mean in tau mode) is only formed when it is needed
    save_now   = iter % solver_param['save_interval'] == 0
    visual_now = solver_param['visual'] and iter % solver_param['vis_update_interval'] == 0

    if save_now or visual_now:

        state = prepare_to_store(solver_param, state, physics)

        state['Q_prim'] = reshape_func.solver_add_ghost(solver_param['cell_number'], solver_param['num_prim_var'], state['mean_prim'].ravel())
        state           = physics.prim2cons_converter(solver_param, state)

    if save_now:

        results_recorder(solver_param, state)

    return solver_param, state, rom_param
