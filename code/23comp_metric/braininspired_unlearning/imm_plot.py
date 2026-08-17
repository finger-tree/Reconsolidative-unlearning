"""
Plot results.

Python translation of imm_plot.m (Gershman, memory-modification repo).

USAGE: imm_plot(mode, [results])

Notes on translation
---------------------
- MATLAB's `mytitle(str,'Left',...)` (an external helper not included in the
  original repo) is reproduced with `ax.set_title(str, loc='left')`.
- `colormap hot` / `colormap bone` are reproduced by passing
  `cmap='hot'` / `cmap='bone'` (or manually sampling the colormap) to the
  relevant matplotlib calls.
- Several cases in the original load a `.mat` file (`ERE_data`) that is not
  included in this repository; that branch is reproduced with a clear
  runtime error pointing this out, rather than silently fabricating data.
- `results` is expected to be a list of SimpleNamespace objects as returned
  by `imm_run`, indexed the same way as the MATLAB struct array
  (results(i) -> results[i-1]).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm

from imm_run import imm_run

YLIM = (-0.1, 1.1)
YTICK = [0, 0.2, 0.4, 0.6, 0.8, 1]

DEFAULT_OPTS = dict(alpha=0.1, g=1, psi=0, eta=0.3, maxIter=3, w0=0, sr=0.4,
                     sx=1, theta=0.02, lam=0.01, K=15)


def _mytitle(ax, text, **kwargs):
    """Stand-in for the external `mytitle(str,'Left',...)` MATLAB helper."""
    fontsize = kwargs.get('FontSize', 14)
    fontweight = 'bold' if kwargs.get('FontWeight', '').lower() == 'bold' else 'normal'
    ax.set_title(text, loc='left', fontsize=fontsize, fontweight=fontweight)


def imm_plot(mode, results=None):
    opts = dict(DEFAULT_OPTS)

    if mode == 'monfils09':
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(2, 3, (1, 3))
        n = np.cumsum(results[0].N)
        clr = ['bo', 'gs', 'r^']
        lwidth, msize = 3, 10
        for i in range(3):
            V = results[i].V
            ax1.plot(np.arange(1, n[0] + 1), V[:n[0]], '-' + clr[i],
                     linewidth=lwidth, markersize=msize, markerfacecolor='w')
        for i in range(3):
            V = results[i].V
            ax1.plot(n[1], V[n[1] - 1], clr[i], linewidth=lwidth,
                      markersize=msize, markerfacecolor='w')
        for i in range(3):
            V = results[i].V
            ax1.plot(np.arange(n[1] + 1, n[2] + 1), V[n[1]:n[2]], '-' + clr[i],
                      linewidth=lwidth, markersize=msize, markerfacecolor='w')
        for i in range(3):
            V = results[i].V
            ax1.plot(n[3], V[n[3] - 1], clr[i], linewidth=lwidth,
                      markersize=msize, markerfacecolor='w')
        ax1.set_ylabel('CR', fontsize=20)
        ax1.set_xlim(0, n[-1] + 1)
        ax1.set_ylim(*YLIM)
        ax1.tick_params(labelsize=20)
        ax1.set_yticks(YTICK)
        ax1.set_xticks([1, 4, 5, 23])
        ax1.set_xticklabels(['Acq', 'Ret', 'Ext', 'Test'])
        ax1.legend(['No Ret (interval = 0)', 'Ret-short (interval = 3)',
                    'Ret-long (interval = 100)'], fontsize=12)
        _mytitle(ax1, 'A', FontWeight='Bold', FontSize=20)
        for i in range(len(n) - 1):
            ax1.plot([n[i] + 0.5, n[i] + 0.5], YLIM, '--k')

        letters = ['B         No Ret', 'C         Ret-short', 'D         Ret-long']
        for i in range(3):
            ax = fig.add_subplot(2, 3, i + 4)
            z = np.array([results[i].Zp[n[j] - 1, :3] for j in range(4)])
            x = np.arange(4)
            width = 0.25
            for k in range(3):
                ax.bar(x + k * width, z[:, k], width, color=cm.hot(k / 3.0))
            if i == 0:
                ax.legend(['C1', 'C2', 'C3'], fontsize=12)
            ax.set_ylabel('Posterior Prob.', fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels(['Acq', 'Ret', 'Ext', 'Test'])
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=14)
            _mytitle(ax, letters[i], FontWeight='Bold', FontSize=16)
        fig.tight_layout()
        return fig

    elif mode == 'ERE':
        lwidth, msize = 3, 13
        fig = plt.figure(figsize=(10, 8))

        ax1 = fig.add_subplot(2, 2, 1)
        n = np.cumsum(results[4].N)
        z = np.array([results[4].Zp[n[j] - 1, :3] for j in range(5)])
        x = np.arange(5)
        width = 0.25
        for k in range(3):
            ax1.bar(x + k * width, z[:, k], width, color=cm.hot(k / 3.0))
        ax1.set_ylabel('Posterior Prob.', fontsize=14)
        ax1.legend(['C1', 'C2', 'C3'], fontsize=10, loc='upper center')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(['Train', 'Ret', 'Ext', 'US', 'Test'])
        ax1.set_ylim(0, 1)
        _mytitle(ax1, 'B          R-E', FontWeight='Bold', FontSize=16)

        ax2 = fig.add_subplot(2, 2, 2)
        n = np.cumsum(results[3].N)
        z = np.array([results[3].Zp[n[j] - 1, :3] for j in range(5)])
        for k in range(3):
            ax2.bar(x + k * width, z[:, k], width, color=cm.hot(k / 3.0))
        ax2.set_ylabel('Posterior Prob.', fontsize=14)
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(['Train', 'Ret', 'Ext', 'US', 'Test'])
        ax2.set_ylim(0, 1)
        _mytitle(ax2, 'C          E-R-E', FontWeight='Bold', FontSize=16)

        ax3 = fig.add_subplot(2, 2, 3)
        v = [results[4].V[-1], results[3].V[-1]]
        ax3.plot(1, v[0], 'ok', markerfacecolor='w', markersize=msize, linewidth=lwidth)
        ax3.plot(2, v[1], 'ok', markerfacecolor='k', markersize=msize, linewidth=lwidth)
        ax3.set_xticks([1, 2])
        ax3.set_xticklabels(['R-E', 'E-R-E'])
        ax3.set_xlim(0.5, 2.5)
        ax3.set_ylim(0, 1)
        _mytitle(ax3, 'D          Simulation', FontWeight='Bold', FontSize=16)
        ax3.set_ylabel('CR', fontsize=14)

        ax4 = fig.add_subplot(2, 2, 4)
        try:
            from scipy.io import loadmat
            mat = loadmat('ERE_data.mat')
            RE, ERE_ = mat['RE'], mat['ERE']
            v = [RE.mean(), ERE_.mean()]
            se = [RE.mean(axis=1).std(), ERE_.mean(axis=1).std()]
            ax4.errorbar(1, v[0], yerr=se[0], fmt='ok', markerfacecolor='w',
                         markersize=msize, linewidth=lwidth)
            ax4.errorbar(2, v[1], yerr=se[1], fmt='ok', markerfacecolor='k',
                         markersize=msize, linewidth=lwidth)
        except FileNotFoundError:
            ax4.text(0.5, 0.5, "ERE_data not available\n(not included in repo)",
                      ha='center', va='center', transform=ax4.transAxes)
        ax4.set_xticks([1, 2])
        ax4.set_xticklabels(['R-E', 'E-R-E'])
        ax4.set_xlim(0.5, 2.5)
        ax4.set_ylim(0, 100)
        ax4.set_ylabel('% Freezing', fontsize=14)
        _mytitle(ax4, 'E          Data', FontWeight='Bold', FontSize=16)
        fig.tight_layout()
        return fig

    elif mode == 'suzuki04':
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].bar([0, 1], [results[5].V[-1], results[6].V[-1]])
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Short', 'Long'])
        axes[0].set_ylim(0, YLIM[1])
        axes[0].set_ylabel('CR', fontsize=14)
        axes[0].set_xlabel('Reexposure duration', fontsize=14)
        axes[0].set_box_aspect(1)

        axes[1].bar([0, 1], [results[5].V[-1], results[8].V[-1]])
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Young', 'Old'])
        axes[1].set_ylim(0, YLIM[1])
        axes[1].set_ylabel('CR', fontsize=14)
        axes[1].set_xlabel('Memory age', fontsize=14)
        axes[1].set_box_aspect(1)

        axes[2].bar([0, 1], [results[5].V[-1], results[9].V[-1]], color=cm.bone([0.3, 0.7]))
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(['Weak', 'Strong'])
        axes[2].set_ylim(0, YLIM[1])
        axes[2].set_ylabel('CR', fontsize=14)
        axes[2].set_xlabel('Memory strength', fontsize=14)
        axes[2].set_box_aspect(1)
        fig.tight_layout()
        return fig

    elif mode == 'prediction_error':
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        v = [imm_run(6).V[-1], imm_run(11).V[-1]]
        axes[0].bar([0, 1], v, color=cm.bone([0.3, 0.7]))
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['No US', 'US'])
        axes[0].set_ylim(0, YLIM[1])
        axes[0].set_ylabel('CR', fontsize=14)
        axes[0].set_xlabel('Reexposure condition', fontsize=14)
        axes[0].set_box_aspect(1)

        v = [imm_run(10).V[-1], imm_run(12).V[-1]]
        axes[1].bar([0, 1], v, color=cm.bone([0.3, 0.7]))
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Strong', 'Strong+N'])
        axes[1].set_ylim(0, YLIM[1])
        axes[1].set_ylabel('CR', fontsize=14)
        axes[1].set_xlabel('Memory strength', fontsize=14)
        axes[1].set_box_aspect(1)
        fig.tight_layout()
        return fig

    elif mode == 'timecourse':
        opts['maxIter'] = 3
        ITI = 3
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))

        res = imm_run(13, opts, ITI)
        # W{3,min(ITI,maxIter)} / P{3,min(ITI,maxIter)}  (MATLAB trial 3 -> idx 2)
        it0 = min(ITI, opts['maxIter']) - 1
        w = [res.W[(2, it0)][0, :]]
        p = [res.P[(2, it0)]]
        for j in range(opts['maxIter']):
            w.append(res.W[(3, j)][0, :])
            p.append(res.P[(3, j)])
        w = np.array([wi[0] for wi in w])
        p = np.array([pi[0] for pi in p])
        axes[0].plot(w, p, '-sk', linewidth=3, markersize=12, markerfacecolor='w')
        nums = ['0'] + [str(j + 1) for j in range(opts['maxIter'])]
        for wi, pi, num in zip(w, p, nums):
            axes[0].text(wi + 0.01, pi - 0.07, num, fontsize=14)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_xlabel('Associative strength  (Acq. cause)', fontsize=14)
        axes[0].set_ylabel('Posterior prob. (Acq. cause)', fontsize=14)
        _mytitle(axes[0], 'A', FontWeight='Bold', FontSize=16)

        x = np.linspace(0.01, 20, 60)
        y = np.zeros(len(x))
        for i, xi in enumerate(x):
            y[i] = imm_run(13, opts, xi).P[(4, 0)][0]
        axes[1].plot(1 + x, y, '-k', linewidth=4)
        axes[1].set_ylim(0, 1.1)
        axes[1].set_xlim(-1, x[-1] + 1)
        axes[1].set_xlabel('Retrieval-extinction interval', fontsize=14)
        axes[1].set_ylabel('Posterior prob. (Acq. cause)', fontsize=14)
        _mytitle(axes[1], 'B', FontWeight='Bold', FontSize=16)
        fig.tight_layout()
        return fig

    elif mode == 'compression':
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(1, 1, 1)

        def f(t):
            return t ** (-1.0)

        t = np.linspace(1, 52, 100)
        j = [2, 5]
        y = np.column_stack([f(t) / (f(t + jj) + f(t)) for jj in j])
        ax.plot(t, y[:, 0], '-k', linewidth=4)
        ax.set_xlim(0, t[-1] + 1)
        ax.set_ylim(0.5, 1)
        ax.set_xlabel(r'Memory age, $\tau(t_3)-\tau(t_2)$', fontsize=14)
        ax.set_ylabel(r'$P(z_3=z_2)$', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'schiller10':
        res = imm_run(14)
        fig, ax = plt.subplots(figsize=(7, 5))
        idx_acq = [6, 7, 8]      # MATLAB 7,8,9 -> 0-indexed
        idx_ext = list(range(len(res.V) - 6, len(res.V) - 3))
        idx_test = list(range(len(res.V) - 3, len(res.V)))
        vals = np.array([res.V[idx_acq], res.V[idx_ext], res.V[idx_test]]) + 1e-2
        x = np.arange(3)
        width = 0.25
        for k in range(3):
            ax.bar(x + k * width, vals[:, k], width, color=cm.bone(k / 3.0),
                   label=['CSa+ (Ret)', 'CSb+ (No Ret)', 'CS-'][k])
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Acquisition', 'Extinction', 'Test'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('CR', fontsize=14)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    elif mode == 'doyere07':
        res = imm_run(15)
        v = list(res.V[-2:])
        res = imm_run(16)
        v += list(res.V[-2:])
        fig, ax = plt.subplots(figsize=(6, 5))
        # v = [PSI-CSr, PSI-CSn, Control-CSr, Control-CSn]
        x = np.arange(2)
        width = 0.35
        ax.bar(x, [v[0], v[2]], width, color=cm.bone(0.3), label='CSr')
        ax.bar(x + width, [v[1], v[3]], width, color=cm.bone(0.7), label='CSn')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(['PSI', 'Control'])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left', fontsize=12)
        ax.set_ylabel('CR', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'jarome12':
        iti = [5, 50, 100, 200]
        v = np.zeros((len(iti), 2))
        for i, it in enumerate(iti):
            v[i, 0] = imm_run(18, None, it, 1).V[-1]
            v[i, 1] = imm_run(18, None, it, 0).V[-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(v[:, 0], '-ok', markersize=10, markerfacecolor='k', linewidth=2.5)
        ax.plot(v[:, 1], '-ok', markersize=10, markerfacecolor='w', linewidth=2.5)
        ax.set_xticks(range(len(iti)))
        ax.set_xticklabels(iti)
        ax.set_xlim(-0.5, len(iti) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(['PSI', 'Control'], fontsize=12, loc='center right')
        ax.set_xlabel('ITI', fontsize=14)
        ax.set_ylabel('CR', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'rbf':
        x = np.linspace(-2, 2, 100)
        sr = 0.5
        y = norm.pdf(x, 0, sr)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y, '-k', linewidth=2.5)
        ax.plot([0, 0], ax.get_ylim(), '--k', linewidth=2.5)
        m = np.array([-0.33, 0.33])
        p = norm.pdf(m, 0, sr)
        ax.plot(m, p, '-', linewidth=2.5, color=(0.5, 0.5, 0.5))
        ax.text(m[0] + 0.05, p[0] - 0.04, r'$\sigma_r$', fontsize=13)
        ax.set_xlabel(r'Prediction error ($\delta$)', fontsize=14)
        ax.set_ylabel('Activation', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'power06':
        r = [1] + list(range(10, 101, 10))
        v = [imm_run(19, None, ri).V[-1] for ri in r]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(r, v, '-ok', markersize=10, linewidth=2.5, markerfacecolor='k')
        ax.set_xlim(0, r[-1] + 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Retrieval-test interval', fontsize=14)
        ax.set_ylabel('CR', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'constanzi11':
        res1 = imm_run(2)
        res2 = imm_run(21)
        v = [res1.V[-1], res2.V[-1]]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar([0, 1], v, color=cm.bone([0.3, 0.7]))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['1 day', '29 days'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('CR', fontsize=14)
        ax.set_xlabel('Acquisition-retrieval interval', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'renewal':
        res1 = imm_run(23)
        res2 = imm_run(22)
        V = [res1.V[-1], res2.V[-1]]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar([0, 1], V, color=cm.bone([0.3, 0.7]))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['A*', 'B'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('CR', fontsize=14)
        ax.set_xlabel('Retrieval/extinction context', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'debiec13':
        res = imm_run(24)
        v = list(res.V[-2:])
        res = imm_run(25)
        v += list(res.V[-2:])
        # v = [PSI-CSr, PSI-CSn, Control-CSr, Control-CSn]
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(2)
        width = 0.35
        ax.bar(x, [v[0], v[2]], width, color=cm.bone(0.3), label='CSr')
        ax.bar(x + width, [v[1], v[3]], width, color=cm.bone(0.7), label='CSn')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(['PSI', 'Control'])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper left', fontsize=14)
        ax.set_ylabel('CR', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'new_cause_prob':
        a = [0.1, 0.13, 2]
        c = [0, 0.3, 0.5]
        N = 20
        X = [[0.4, 0.3], [0.1, 0.3], [0.4, 0.1], [0.1, 0.1]]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        for k in range(4):
            ax = axes[k]
            opts_k = dict(opts)
            opts_k['eta'] = X[k][1]
            opts_k['sr'] = X[k][0]
            labels = []
            for i, ai in enumerate(a):
                labels.append(rf'$\alpha$ = {ai}')
                Y = np.zeros(N)
                for n in range(1, N + 1):
                    opts_k2 = dict(opts_k)
                    opts_k2['alpha'] = ai
                    res = imm_run(29, opts_k2, n)
                    Y[n - 1] = res.P[(len(res.V) - 1, 0)][1]
                ax.plot(np.arange(1, N + 1), Y, '-', linewidth=5, color=(c[i], c[i], c[i]))
            if k == 0:
                ax.legend(labels, fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Number of acquisition trials (N)', fontsize=12)
            ax.set_ylabel('P(new cause)', fontsize=12)
            ax.set_title(rf'$\eta$ = {X[k][1]}, $\sigma_r^2$ = {opts_k["sr"]}', fontsize=14, fontweight='bold')
        fig.tight_layout()
        return fig

    elif mode == 'prediction_error_schematic':
        x = np.linspace(-1, 1, 1000)
        y = norm.pdf(x, 0, 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y, '-k', linewidth=4)
        ax.set_xlabel('Prediction error', fontsize=18)
        ax.set_ylabel('Weight change (acquisition cause)', fontsize=18)
        ylim = ax.get_ylim()
        ax.plot([0, 0], ylim, '--', linewidth=3, color=(0.5, 0.5, 0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(-0.9, y.max() + 0.012, 'Memory', fontsize=18)
        ax.text(-0.9, y.max(), 'modification', fontsize=18)
        ax.text(0.3, y.max() + 0.012, 'Memory', fontsize=18)
        ax.text(0.3, y.max(), 'formation', fontsize=18)
        fig.tight_layout()
        return fig

    elif mode == 'ryan15':
        def _ryan_point(design):
            res = imm_run(design)
            v0 = res.V[-1]
            p = res.p.copy()
            p[0] = 1
            p[1:] = 0
            v1 = res.w[0, :] @ p
            v1 = 1 - norm.cdf(opts['theta'], v1, opts['lambda'] if 'lambda' in opts else opts['lam'])
            return v0, v1

        v = np.zeros((2, 2))
        v[0, :] = _ryan_point(27)
        v[1, :] = _ryan_point(28)
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(2)
        width = 0.35
        ax.bar(x, v[:, 0], width, color=cm.bone(0.3), label='SAL')
        ax.bar(x + width, v[:, 1], width, color=cm.bone(0.7), label='ANI')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(['Off', 'On'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('CR', fontsize=14)
        ax.legend(fontsize=14, loc='upper center')
        fig.tight_layout()
        return fig

    elif mode == 'paradoxical_enhancement':
        v = np.zeros((2, 2))
        v[0, 0] = imm_run(32).V[-1]
        v[1, 0] = imm_run(33).V[-1]
        v[0, 1] = imm_run(34).V[-1]
        v[1, 1] = imm_run(35).V[-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        x = np.arange(2)
        width = 0.35
        ax.bar(x, v[0, :], width, color=cm.bone(0.3), label='Ret')
        ax.bar(x + width, v[1, :], width, color=cm.bone(0.7), label='No Ret')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(['Short', 'Long'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('CR', fontsize=18)
        ax.set_xlabel('Retrieval-test interval', fontsize=18)
        ax.legend(fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'state_dependent':
        v = [imm_run(i).V[-1] for i in range(38, 41)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(range(3), v, color=cm.bone([0.2, 0.5, 0.8]))
        ax.set_xticks(range(3))
        ax.set_xticklabels(['PSI-SAL', 'SAL-SAL', 'PSI-PSI'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('CR', fontsize=18)
        fig.tight_layout()
        return fig

    elif mode == 'nader00':
        data = np.array([[0.28, np.nan, np.nan, np.nan],
                          [0.65, 0.70, 0.30, 0.60]])
        n = [5, 15]
        v = np.zeros((2, 5))
        for i, ni in enumerate(n):
            v[i, 0] = imm_run(41, None, ni, 0).V[-1]
            v[i, 1] = imm_run(41, None, ni, 1).V[-1]
            v[i, 2] = imm_run(42, None, ni, 0).V[-1]
            v[i, 3] = imm_run(42, None, ni, 1).V[-1]
            v[i, 4] = imm_run(43, None, ni, 1).V[-1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar([0, 1], [v[0, 1], v[1, 1]], width=0.4, color=cm.bone([0.3, 0.6]))
        axes[0].plot([0, 1], [data[0, 0], data[1, 0]], 'or', markerfacecolor='r', markersize=15)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Immediate', 'Delayed'])
        axes[0].set_ylim(0, 1.1)
        axes[0].set_xlim(-0.5, 1.5)
        axes[0].set_ylabel('CR', fontsize=18)
        axes[0].set_xlabel('Acquisition-PSI interval', fontsize=18)

        axes[1].bar(range(3), [v[0, 2], v[0, 3], v[1, 3]], color=cm.bone([0.2, 0.5, 0.8]))
        axes[1].plot(range(3), data[1, 1:4], 'or', markerfacecolor='r', markersize=15)
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(['No Ret', 'Ret', 'Ret delayed'])
        axes[1].set_ylim(0, 1.1)
        axes[1].set_xlim(-0.5, 2.5)
        axes[1].set_ylabel('CR', fontsize=18)
        fig.set_size_inches(12, 5)
        fig.tight_layout()
        return fig

    elif mode == 'nader00_param':
        alpha = np.linspace(0.01, 2, 10)
        sr = np.linspace(0.1, 1, 10)
        v = np.zeros((len(alpha), len(sr)))
        for i, ai in enumerate(alpha):
            for j, srj in enumerate(sr):
                opts2 = dict(opts)
                opts2['alpha'] = ai
                opts2['sr'] = srj
                res = imm_run(42, opts2, 5, 1)
                v[i, j] = res.V[-1]
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        SR, ALPHA = np.meshgrid(sr, alpha)
        ax.plot_surface(SR, ALPHA, v, cmap='viridis')
        ax.set_ylabel(r'Concentration parameter ($\alpha$)', fontsize=14)
        ax.set_xlabel(r'Reward variance ($\sigma^2$)', fontsize=14)
        ax.set_zlabel('CR', fontsize=14)
        fig.tight_layout()
        return fig

    elif mode == 'familiarization':
        d = [0, 1]
        v = [imm_run(44, None, di).V[-1] for di in d]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar([0, 1], v, color=cm.bone([0.3, 0.7]))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Novel', 'Familiar'])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('CR', fontsize=18)
        fig.tight_layout()
        return fig

    else:
        raise ValueError(f"Unrecognized plot mode: {mode!r}")
