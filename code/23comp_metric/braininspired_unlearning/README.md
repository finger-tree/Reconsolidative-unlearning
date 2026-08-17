# memory-modification (Python translation)

Python translation of Sam Gershman's MATLAB `memory-modification` repository
(https://github.com/sjgershm/memory-modification), which implements a
latent-cause-modulated Rescorla-Wagner model used to simulate memory
reconsolidation / retrieval-extinction experiments.

## Files

| Original MATLAB | Python           | Contents                                              |
|------------------|------------------|--------------------------------------------------------|
| `imm_localmap.m` | `imm_localmap.py`| Core model: `imm_localmap(X, r, Dist, opts=None)`      |
| `imm_run.m`      | `imm_run.py`     | 44 experiment designs: `imm_run(design, opts=None, *varargin)` |
| `imm_plot.m`     | `imm_plot.py`    | Figure reproductions: `imm_plot(mode, results=None)`   |

## Requirements

```
numpy
scipy
matplotlib
```

## Usage

```python
from imm_run import imm_run
from imm_plot import imm_plot

# Run a single design (e.g. design 1: extinction without retrieval)
results = imm_run(1)
print(results.V)     # conditioned response on each trial
print(results.Zp)    # latent cause posterior (per trial)

# Reproduce a figure from the paper (e.g. Monfils et al. 2009)
res = [imm_run(1), imm_run(2), imm_run(3)]
fig = imm_plot('monfils09', res)
fig.savefig('monfils09.png')
```

`results` is a `SimpleNamespace` mirroring the MATLAB struct returned by
`imm_localmap`/`imm_run`, with fields `V`, `Zp`, `Z`, `S`, `W`, `P`, `w`, `p`,
plus (for `imm_run` output) `design`, `N`, `I`, `Time`, `X`, `r`, `opts`.

Design numbers, options fields (`alpha`, `g`, `psi`, `eta`, `maxIter`, `w0`,
`sr`, `sx`, `theta`, `lambda`/`lam`, `K`), and `imm_plot` mode strings all
match the original MATLAB names 1:1 — see the docstrings in each file for
details and translation notes (index conventions, the `mytitle`/colormap
substitutions, etc).

## Translation notes

- MATLAB is 1-indexed; Python is 0-indexed. All indexing (cluster columns,
  `psi` injection trial indices, `results(i)` struct-array access) has been
  carefully converted — see inline comments, especially in `imm_run.py`.
- `imm_run()` called with no `design` argument reproduces the original's
  "run all designs" behavior, including the fact that it silently stops
  early (after design 17) because design 18 needs extra `varargin` that
  aren't supplied in that code path — this is a quirk of the *original*
  MATLAB script, faithfully preserved here.
- `imm_plot.m` calls an external `mytitle` helper (not included in the
  original repo); it's reproduced with `ax.set_title(..., loc='left')`.
- The `'ERE'` plot mode originally loads `ERE_data.mat`, which is not part
  of this repository; that panel gracefully shows a placeholder message
  instead of fabricating data.
