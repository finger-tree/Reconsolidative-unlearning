"""
Run all simulations.

Python translation of imm_run.m (Gershman, memory-modification repo).

USAGE: results = imm_run(design, [opts], *varargin)

Notes on translation
---------------------
- MATLAB is 1-indexed; all cluster/psi/N index literals below have been
  converted to 0-indexed equivalents (e.g. MATLAB `opts.psi(sum(N)-1)` on a
  vector of length `sum(N)` refers to the second-to-last element, which in
  0-indexed Python is `psi[sum(N) - 2]`).
- `varargin` is passed through as *varargin (varargin[0] == MATLAB
  varargin{1}, etc).
- The MATLAB "run all designs" branch (called with no `design` argument)
  relies on hitting a MATLAB error (undefined variable for unhandled design
  numbers, or an out-of-bounds varargin access for designs that require
  extra arguments) to know when to stop appending to the results array.
  That behavior is reproduced faithfully here using exceptions, so
  `imm_run()` with no varargin will, like the original, stop after design
  17 (design 18 needs two extra arguments) unless those are supplied.
"""

import numpy as np

from imm_localmap import imm_localmap


class DesignError(Exception):
    """Raised for unrecognized design numbers (mirrors MATLAB's undefined-
    variable error when a switch/case falls through with no matching case)."""


def _need(varargin, n):
    """Mirror MATLAB's error when varargin{n} is accessed but not provided."""
    if len(varargin) < n:
        raise IndexError(f"varargin[{n - 1}] requested but only {len(varargin)} "
                          f"extra argument(s) were supplied")
    return varargin[n - 1]


def imm_run(design=None, opts=None, *varargin):

    # if no input, run all designs
    if design is None:
        results = []
        for i in range(1, 101):
            try:
                if opts is not None:
                    res = imm_run(i, opts)
                else:
                    res = imm_run(i)
            except Exception:
                return results
            results.append(res)
        return results

    # design parameters (default opts, used when caller doesn't supply any)
    if opts is None:
        opts = dict(alpha=0.1, g=1, psi=0, eta=0.3, maxIter=3,
                    w0=0, sr=0.4, sx=1, theta=0.02, lam=0.01, K=15)
    else:
        opts = dict(opts)  # shallow copy; we may mutate (e.g. opts['psi'])

    # imm_localmap expects the key 'lambda' (a python-safe string key, not
    # the keyword) -- normalize here so callers may use either 'lam' or
    # 'lambda'.
    if 'lam' in opts and 'lambda' not in opts:
        opts['lambda'] = opts.pop('lam')

    # experimental parameters
    nAcq = 3
    train_ext_interval = 20
    ext_test_interval = 200

    N = feats = rewards = I = None

    if design == 1:
        # extinction w/o retrieval
        N = [nAcq, 1, 18, 1]
        feats = [[1], [1], [1], [1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 0, ext_test_interval]

    elif design == 2:
        # extinction w/ retrieval, short ITI
        N = [nAcq, 1, 18, 1]
        feats = [[1], [1], [1], [1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 3, ext_test_interval]

    elif design == 3:
        # extinction w/ retrieval, long ITI
        N = [nAcq, 1, 18, 1]
        feats = [[1], [1], [1], [1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 100, ext_test_interval]

    elif design == 4:
        # ext-ret-ext, short ITI (reinstatement)
        N = [nAcq, 5, 1, 15, 1, 1]
        feats = [[1], [1], [1], [1], [0], [1]]
        rewards = [1, 0, 0, 0, 1, 0]
        I = [train_ext_interval, 20, 3, 20, 20]

    elif design == 5:
        # retrieval - extinction, short ITI (reinstatement)
        N = [nAcq, 1, 18, 1, 1]
        feats = [[1], [1], [1], [0], [1]]
        rewards = [1, 0, 0, 1, 0]
        I = [train_ext_interval, 3, 20, 20]

    elif design == 6:
        # PSI injection, short duration
        N = [1, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 7:
        # PSI injection, long duration
        N = [1, 3, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 8:
        # no PSI injection, long duration (control)
        N = [1, 3, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]

    elif design == 9:
        # PSI injection, long interval
        N = [1, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval * 2, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 10:
        # PSI injection, strong training
        N = [2, 1, 1]
        feats = [[1], [1], [1]]
        # feats = [1 0; 1 0; 1 0]   (commented out in original)
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 11:
        # PSI injection, short duration, with US
        N = [1, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 1, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 12:
        # PSI injection, strong training, novelty
        N = [2, 1, 1]
        # feats = [1 0 1; 1 1 1; 1 0 1]  (commented out in original)
        feats = [[1, 0], [1, 1], [1, 0]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 13:
        # extinction w/ retrieval, variable ITI
        N = [nAcq, 1, 18, 1]
        feats = [[1], [1], [1], [1]]
        rewards = [1, 0, 0, 0]
        iti = varargin[0] if len(varargin) >= 1 else 3
        I = [train_ext_interval, iti, ext_test_interval]

    elif design == 14:
        # Schiller et al., (2010), Experiment 2
        # CSa+ (reminded), CSb+ (not reminded), CS-
        opts['theta'] = 0.0016
        opts['lambda'] = 0.00008
        nExt = 18
        N = [1] * (nAcq * 3) + [1, 1, 1] + [1] * (nExt * 3) + [1, 1, 1]
        eye3 = np.eye(3)
        feats = np.vstack([
            np.tile(eye3, (nAcq, 1)),
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            np.tile(eye3, (nExt, 1)),
            eye3,
        ])
        rewards = np.concatenate([
            np.tile([1, 1, 0], nAcq),
            [0, 0, 0, 0, 0],
            np.zeros(nExt * 3),
            [0, 0, 0],
        ])
        I = ([0] * (nAcq * 3 - 1) + [train_ext_interval, 0, 3, 0]
             + [0] * (nExt * 3 - 1) + [ext_test_interval, 0, 0])

    elif design == 15:
        # PSI injection, short duration (Doyere et al., 2007)
        N = [1, 1, 1, 1, 1]
        feats = [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]
        rewards = [1, 1, 0, 0, 0]
        I = [0, train_ext_interval, 20, 0]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][2] = 1

    elif design == 16:
        # no PSI injection, short duration (Doyere et al., 2007)
        N = [1, 1, 1, 1, 1]
        feats = [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]
        rewards = [1, 1, 0, 0, 0]
        I = [0, train_ext_interval, 20, 0]

    elif design == 17:
        # non-contingent footshock
        # training | amnestic | shock | test
        nAcq_local = 1
        opts['lambda'] = 0.2
        opts['theta'] = 0.1
        opts['psi'] = np.array([0.0] * nAcq_local + [0.01, 0, 0])
        N = [nAcq_local, 1, 1, 1]
        feats_col = np.array([1, 1, 0, 1]).reshape(-1, 1)
        feats = np.hstack([feats_col, np.ones((4, 1))])  # context feature
        rewards = [1, 0, 1, 0]
        I = [train_ext_interval, 20, 0]

    elif design == 18:
        # multiple retrievals (Jarome et al., 2012)
        # PSI injection, short duration
        ITI = _need(varargin, 1)
        PSI = _need(varargin, 2)
        N = [2, 1, 1, 1]
        feats = [[1], [1], [1], [1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, ITI, 20]
        opts['psi'] = np.zeros(sum(N))
        if PSI:
            opts['psi'][sum(N) - 2] = 1

    elif design == 19:
        # PSI injection, variable retention interval
        retention_interval = _need(varargin, 1)
        N = [1, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [100, retention_interval]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 20:
        # Sevenster et al. (2013) PE experiment
        N = [1, 1, 1, 1, 12, 1, 1]
        feats_col = np.array([1, 1, 1, 1, 1, 0, 1]).reshape(-1, 1)
        feats = np.hstack([np.ones_like(feats_col), feats_col])
        rewards = [1, 1, 1, 1, 0, 1, 0]
        I = [0, 0, train_ext_interval, train_ext_interval, 0, 0]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][3] = 1

    elif design == 21:
        # Constanzi et al. (2011) variation on Monfils et al. (2009):
        # longer training-extinction interval
        N = [nAcq, 1, 18, 1]
        feats = [[1], [1], [1], [1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval * 29, 3, ext_test_interval]

    elif design == 22:
        N = [nAcq, 1, 18, 1]
        feats = [[1, 1], [1, 0], [1, 0], [1, 1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 3, train_ext_interval]

    elif design == 23:
        N = [nAcq, 1, 18, 1]
        feats = [[1, 1], [1, 0.8], [1, 0.8], [1, 1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 3, train_ext_interval]

    elif design == 24:
        # PSI injection, Debiec et al. (2013)
        N = [1, 1, 1, 1]
        feats = [[1, 1], [1, 0], [1, 0], [0, 1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 20, 0]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][1] = 1

    elif design == 25:
        # no PSI injection, Debiec et al. (2013)
        N = [1, 1, 1, 1]
        feats = [[1, 1], [1, 0], [1, 0], [0, 1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 20, 0]

    elif design == 26:
        # US retrieval (Liu et al 2014)
        N = [nAcq, 1, 18, 1]
        feats = [[1, 1], [0, 1], [1, 1], [1, 1]]
        rewards = [1, 0, 0, 0]
        I = [train_ext_interval, 3, ext_test_interval]

    elif design == 27:
        # no PSI injection, short duration, context change
        N = [1, 1, 1]
        feats = [[1, 0], [1, 0], [0, 1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]

    elif design == 28:
        # PSI injection, short duration, context change
        N = [1, 1, 1]
        feats = [[1, 0], [1, 0], [0, 1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 29:
        # extinction, variable N
        nAcq_local = _need(varargin, 1)
        N = [nAcq_local, 1]
        feats = [[1], [1]]
        rewards = [1, 0]
        I = train_ext_interval  # scalar, as in original

    elif design == 30:
        # post-training PSI injection + pre-test reminder
        N = [2, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][N[0] - 1] = 1

    elif design == 31:
        # post-training PSI injection + no reminder
        N = [2, 1, 1]
        feats = [[1], [0], [1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][N[0] - 1] = 1

    elif design == 32:
        # paradoxical enhancement of fear (Rorbaugh & Riccio): short
        # retention interval + reminder
        N = [2, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [0.1, 0, 0]
        I = [train_ext_interval, 2]
        opts['psi'] = np.zeros(sum(N))

    elif design == 33:
        # paradoxical enhancement of fear (Rorbaugh & Riccio): short
        # retention interval + no reminder
        N = [2, 1, 1]
        feats = [[1], [0], [1]]
        rewards = [0.1, 0, 0]
        I = [train_ext_interval, 2]

    elif design == 34:
        # paradoxical enhancement of fear (Rorbaugh & Riccio): long
        # retention interval + reminder
        N = [2, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [0.1, 0, 0]
        I = [train_ext_interval, 50]
        opts['psi'] = np.zeros(sum(N))

    elif design == 35:
        # paradoxical enhancement of fear (Rorbaugh & Riccio): long
        # retention interval + no reminder
        N = [2, 1, 1]
        feats = [[1], [0], [1]]
        rewards = [0.1, 0, 0]
        I = [train_ext_interval, 50]

    elif design == 36:
        # forgetting of stimulus attributes
        N = [1, 1, 1]
        feats = [[1, 0, 1], [0, 0, 0], [1, 1, 0]]
        rewards = [0.1, 0, 0]
        I = [0, 1]

    elif design == 37:
        # variable interval
        N = [1, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [1, 0, 0]
        I = [100, 200]

    elif design == 38:
        # PSI after conditioning (Gisquet-Verrier 2015)
        N = [1, 1, 1]
        feats = [[1, 1], [0, 0], [1, 0]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 39:
        # no PSI at conditioning or test (Gisquet-Verrier 2015)
        N = [1, 1, 1]
        feats = [[1, 0], [0, 0], [1, 0]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]

    elif design == 40:
        # PSI at conditioning + test (Gisquet-Verrier 2015)
        N = [1, 1, 1]
        feats = [[1, 1], [0, 0], [1, 1]]
        rewards = [1, 0, 0]
        I = [train_ext_interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1
        opts['psi'][sum(N) - 1] = 1

    elif design == 41:
        # post-acquisition retrograde gradient of amnesia
        interval = _need(varargin, 1)
        feat_mid = _need(varargin, 2)
        N = [1, 1, 1]
        feats = [[1], [feat_mid], [1]]
        rewards = [1, 0, 0]
        I = [interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 42:
        # post-retrieval ANI, delayed
        interval = _need(varargin, 1)
        feat_mid = _need(varargin, 2)
        N = [1, 1, 1]
        feats = [[1], [feat_mid], [1]]
        rewards = [1, 0, 0]
        I = [interval, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    elif design == 43:
        # post-acquisition no PSI
        interval = _need(varargin, 1)
        feat_mid = _need(varargin, 2)
        N = [1, 1, 1]
        feats = [[1], [feat_mid], [1]]
        rewards = [1, 0, 0]
        I = [interval, 20]

    elif design == 44:
        # PSI injection, familiarization
        n_familiar = _need(varargin, 1)
        N = [n_familiar, 1, 1]
        feats = [[1], [1], [1]]
        rewards = [0, 1, 0]
        I = [20, 20]
        opts['psi'] = np.zeros(sum(N))
        opts['psi'][sum(N) - 2] = 1

    else:
        raise DesignError(f"Unrecognized design number: {design}")

    N = np.asarray(N, dtype=int).reshape(-1)
    feats = np.atleast_2d(np.asarray(feats, dtype=float))
    rewards = np.asarray(rewards, dtype=float).reshape(-1)
    if np.isscalar(I) or (isinstance(I, np.ndarray) and I.ndim == 0):
        I = [float(I)]
    I = list(np.asarray(I, dtype=float).reshape(-1))

    # ----- construct features and rewards ---%
    X_parts = []
    r_parts = []
    for i in range(feats.shape[0]):
        X_parts.append(np.tile(feats[i, :], (int(N[i]), 1)))
        r_parts.append(np.tile(rewards[i], int(N[i])))
    X = np.vstack(X_parts)
    r = np.concatenate(r_parts)

    # ------- construct time indices ---------%
    I_full = [0] + I
    t = 0
    Time = []
    for i in range(len(N)):
        t = t + I_full[i]
        for j in range(int(N[i])):
            Time.append(t)
            t = t + 1
    Time = np.asarray(Time, dtype=float)

    # ------- construct distance matrix ---------%
    T = int(N.sum())
    Dist = np.abs(Time.reshape(-1, 1) - Time.reshape(1, -1))

    # -------- run particle filter -----------%
    results = imm_localmap(X, r, Dist, opts)
    results.design = design
    results.N = N
    results.I = np.asarray(I_full[1:])
    results.Time = Time
    results.X = X
    results.r = r
    results.opts = opts
    return results
