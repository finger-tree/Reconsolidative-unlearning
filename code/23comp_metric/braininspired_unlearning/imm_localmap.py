"""
Latent cause-modulated Rescorla-Wagner model.

Python translation of imm_localmap.m (Gershman, memory-modification repo).
"""

from types import SimpleNamespace
import numpy as np
from scipy.stats import norm


DEFAULT_OPTS = {
    'alpha': 0.1,     # concentration parameter
    'g': 1,            # temporal scaling parameter
    'psi': 0,          # binary vector: protein synthesis inhibitor injected
    'eta': 0.2,        # learning rate
    'maxIter': 3,       # max EM iterations between trials
    'w0': 0,            # initial weight value
    'sr': 0.4,          # US variance
    'sx': 1,            # stimulus variance
    'theta': 0.03,      # response threshold
    'lambda': 0.005,    # response gain
    'K': 15,            # max number of latent sources
}


def _broadcast_to_T(value, T):
    """Mirror MATLAB's `value .* ones(T,1)` broadcasting behaviour."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(T, arr.item())
    if arr.size == T:
        return arr.astype(float).copy()
    raise ValueError(f"Cannot broadcast array of size {arr.size} to length {T}")


def imm_localmap(X, r, Dist, opts=None):
    """
    Latent cause-modulated Rescorla-Wagner model.

    Parameters
    ----------
    X : (N, D) array
        X[n, d] is the stimulus intensity of stimulus d on trial n.
    r : (N,) array
        US (reward) on each trial.
    Dist : (N, N) array
        Temporal distance between each timepoint.
    opts : dict, optional
        Options overriding DEFAULT_OPTS. Any missing / None-valued field is
        filled in from DEFAULT_OPTS (mirroring the MATLAB fieldnames loop).

    Returns
    -------
    results : SimpleNamespace with fields
        V   - (N,) conditioned response on each trial
        Zp  - (N, K) latent cause posterior including reward (final EM iter)
        Z   - (N, K) latent cause hard assignment (MAP)
        S   - (N, N) temporal similarity kernel Dist ** (-g)
        w   - (D, K) weight matrix snapshot at the last processed trial
             (before the reward-driven EM updates on that trial)
        p   - (K,) prior-only posterior snapshot at the last processed trial
        W, P - dict keyed by (t, iter) -> weight matrix / posterior at each
               EM iteration (mirrors MATLAB cell arrays results.W{t,iter})
    """
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)
    Dist = np.asarray(Dist, dtype=float)

    # ###### DEFAULT PARAMETERS ##########
    merged = dict(DEFAULT_OPTS)
    if opts:
        for key, default_val in DEFAULT_OPTS.items():
            val = opts.get(key, None)
            if val is None or (np.isscalar(val) and False):
                merged[key] = default_val
            else:
                # treat empty arrays as "not provided", same as MATLAB isempty
                arr = np.atleast_1d(val)
                if arr.size == 0:
                    merged[key] = default_val
                else:
                    merged[key] = val
    opts = merged

    # ######## INITIALIZATION ############
    T, D = X.shape
    alpha = _broadcast_to_T(opts['alpha'], T)
    eta = _broadcast_to_T(opts['eta'], T)
    psi = _broadcast_to_T(opts['psi'], T)
    K = int(opts['K'])
    sx = float(opts['sx'])
    sr = float(opts['sr'])
    w0 = float(opts['w0'])
    theta = opts['theta']
    lam = float(opts['lambda'])
    g = float(opts['g'])

    Z = np.zeros((T, K))
    V = np.zeros(T)
    Zp = np.zeros((T, K))
    W = np.zeros((D, K)) + w0

    with np.errstate(divide='ignore'):
        S = Dist ** (-g)  # temporal similarity kernel (diag entries -> inf, unused)

    cause_index = []  # grows by one entry every time a genuinely new cause is created

    results = SimpleNamespace(V=V, Zp=Zp, Z=Z, S=S, W={}, P={}, w=None, p=None,
                               cause_index=cause_index)

    # ########## RUN INFERENCE ############
    for t in range(T):  # 0-indexed trial, matches MATLAB t=1:T with t-1 offset

        # determine how many EM iterations to perform based on ITI
        if t == T - 1:
            nIter = 1
        else:
            nIter = int(min(opts['maxIter'], round(Dist[t, t + 1])))

        # calculate (unnormalized) posterior, not including reward
        Zhist = Z[:t, :]           # history of cluster assignments (t x K)
        Xhist = X[:t, :]
        N = Zhist.sum(axis=0)      # cluster counts, shape (K,)

        # ddCRP prior: S(1:t-1,t)' * Z(1:t-1,:)
        prior = S[:t, t] @ Zhist   # shape (K,); empty-array matmul -> zeros(K)

        # probability of a new cluster goes on the *first* empty cluster
        zero_idx = np.flatnonzero(N == 0)
        if zero_idx.size > 0:
            prior[zero_idx[0]] = alpha[t]

        L = prior / prior.sum()   # normalize prior

        xsum = Xhist.T @ Zhist    # [D x K] matrix of feature sums
        nu = sx / (N + sx) + sx

        for d in range(D):
            xhat = xsum[d, :] / (N + sx)
            L = L * norm.pdf(X[t, d], loc=xhat, scale=np.sqrt(nu))  # likelihood

        # reward prediction, before feedback
        post = L / L.sum()
        V[t] = (X[t, :] @ W) @ post
        results.w = W.copy()
        results.p = post.copy()
        if not (isinstance(theta, float) and np.isnan(theta)):
            V[t] = 1 - norm.cdf(theta, loc=V[t], scale=lam)

        # loop over EM iterations
        for it in range(nIter):
            Vpred = X[t, :] @ W                                   # reward prediction, shape (K,)
            post = L * norm.pdf(r[t], loc=Vpred, scale=np.sqrt(sr))  # unnormalized posterior w/ reward
            post = post / post.sum() if post.sum() > 0 else np.zeros_like(post)
            Zp[t, :] = post

            rpe = np.tile((r[t] - Vpred) * post, (D, 1))          # reward prediction error, (D,K)
            xrep = np.tile(X[t, :].reshape(-1, 1), (1, K))        # (D,K)
            W = W + eta[t] * xrep * rpe                            # weight update

            if psi[t] > 0:
                W = W * (1 - np.tile(post, (D, 1))) * psi[t]

            results.W[(t, it)] = W.copy()
            results.P[(t, it)] = post.copy()

        # cluster assignment (MAP)
        k = int(np.argmax(post))
        if N[k] == 0:
            # this cause had never been used before this trial -> newly created.
            # Populate the growing cause-index array with a record of the event.
            cause_index.append({'trial': t, 'cause_id': k})
        Z[t, k] = 1

    # store results
    results.Z = Z
    results.S = S
    results.V = V
    results.Zp = Zp
    return results
