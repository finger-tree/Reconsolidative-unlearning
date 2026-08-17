"""
LatentCauseLayer
=================

A thin reframing of the latent-cause Rescorla-Wagner model (imm_localmap)
around the vocabulary:

    conditioned stimulus (CS)   -> data
    unconditioned stimulus (US) -> unlearning_marker
    latent cause                -> cause
    conditioned response (CR)   -> output

The layer's output is computed *entirely* as a function of the inferred
hidden causes for the current input (`output = f(data, cause_posterior)`),
matching the "layer output determined entirely by hidden causes" framing.
Every time inference creates a genuinely new cause (i.e. assigns MAP credit
to a cluster that had never been used before), that event is appended to a
growing array, `self.causes`, rather than living implicitly inside a
pre-allocated fixed-size matrix.
"""

from types import SimpleNamespace
import numpy as np

from imm_localmap import imm_localmap, DEFAULT_OPTS


class LatentCauseLayer:
    """
    A single latent-cause layer.

    Attributes (populated after `forward`)
    ---------------------------------------
    data              : (T, D) array - the input passed in (formerly "CS")
    unlearning_marker : (T,) array   - the target/label signal (formerly "US")
    causes            : list[dict]   - grows by one entry *each time* a new
                                        cause is created, e.g.
                                        [{'trial': 0, 'cause_id': 0},
                                         {'trial': 7, 'cause_id': 1}, ...]
    cause_posterior   : (T, K) array - P(cause | data, unlearning_marker) per trial
    output            : (T,) array   - layer output (formerly "CR"),
                                        a function only of data and cause_posterior
    """

    def __init__(self, opts=None):
        self.opts = dict(DEFAULT_OPTS)
        if opts:
            self.opts.update(opts)

        self.data = None
        self.unlearning_marker = None
        self.causes = []          # the growing array requested
        self.cause_posterior = None
        self.output = None
        self._raw_results = None  # full underlying imm_localmap results, if needed

    def forward(self, data, unlearning_marker, Dist):
        """
        Run one forward pass of the layer.

        Parameters
        ----------
        data : (T, D) array-like
            Input features per trial (formerly the CS matrix X).
        unlearning_marker : (T,) array-like
            Target/label signal per trial (formerly the US vector r).
        Dist : (T, T) array-like
            Temporal distance matrix between trials (unchanged from the
            original model; governs how quickly causes decay/merge).

        Returns
        -------
        output : (T,) np.ndarray
            The layer's output, dependent entirely on the inferred causes.
        """
        results = imm_localmap(data, unlearning_marker, Dist, self.opts)

        self.data = np.asarray(data, dtype=float)
        self.unlearning_marker = np.asarray(unlearning_marker, dtype=float)
        self.cause_posterior = results.Zp
        self.output = results.V
        self.causes = results.cause_index   # the growing array of creation events
        self._raw_results = results

        return self.output

    @property
    def num_causes_created(self):
        """How many distinct causes have been created so far."""
        return len(self.causes)

    def cause_timeline(self):
        """
        Return (trial_indices, cause_ids) arrays describing, in order, when
        each new cause was created -- i.e. the growing array made explicit
        as two parallel numpy arrays for easy plotting/inspection.
        """
        if not self.causes:
            return np.array([], dtype=int), np.array([], dtype=int)
        trials = np.array([c['trial'] for c in self.causes])
        ids = np.array([c['cause_id'] for c in self.causes])
        return trials, ids


if __name__ == '__main__':
    # Minimal smoke test / usage example, using hand-built dummy data instead
    # of imm_run() -- useful for testing LatentCauseLayer in isolation.
    from types import SimpleNamespace

    # dummy design: a single stimulus, 3 rewarded "acquisition" trials
    # followed by 5 unrewarded "extinction" trials. A large time gap (20
    # units) is inserted between the two phases -- similar to the gap used
    # in the real experiment designs -- which is what triggers the model to
    # infer a second, distinct cause for the extinction phase.
    design = SimpleNamespace(
        X=np.array([[5.0], [1.0], [1.0], [0.0], [1.0], [1.0], [5.0], [0.0]]),
        r=np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        Time=np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 23.0, 24.0]),
        # same opts imm_run's designs actually use by default (a bit
        # faster learning rate / lower response threshold than
        # imm_localmap's own bare defaults) -- strong enough training in
        # only 3 acquisition trials to create a clear reward-prediction
        # mismatch (and hence a second cause) once reward stops.
        opts=dict(alpha=0.1, g=1, psi=0, eta=0.3, maxIter=3, w0=0, sr=0.4,
                   sx=1, theta=0.02, **{'lambda': 0.01}, K=15),
    )

    layer = LatentCauseLayer(opts=design.opts)
    output = layer.forward(design.X, design.r, np.abs(
        design.Time.reshape(-1, 1) - design.Time.reshape(1, -1)))

    print("output (CR):", np.round(output, 3))
    print("num causes created:", layer.num_causes_created)
    print("cause creation events:", layer.causes)
    trials, ids = layer.cause_timeline()
    print("cause timeline -> trials:", trials, "cause_ids:", ids)
