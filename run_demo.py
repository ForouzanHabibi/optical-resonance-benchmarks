"""Generate the reproducible figures and numerical checks for the slab benchmark."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from slab_resonances import (scattering, resonance_poles, denominator,
    truncated_pole_scattering, quality_factor, first_order_pole_shift)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
n = 3.0
x = np.linspace(0.15, 4.0, 1400)
t, r = scattering(x, n)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(x, np.abs(t)**2, color='black', lw=2.2, label='Exact transmission')
for N, style in [(3, '--'), (8, '-.'), (32, ':')]:
    tN, _ = truncated_pole_scattering(x, N, n)
    ax.plot(x, np.abs(tN)**2, ls=style, lw=1.5, label=f'Pole approximation: m = -{N} ... {N}')
ax.set(xlabel=r'Dimensionless vacuum wave number $x=k_0d$', ylabel=r'Transmission $|t|^2$',
       title='Open dielectric slab: exact vs truncated pole reconstruction')
ax.set_ylim(bottom=0)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT/'01_transmission_comparison.png', dpi=180); plt.close(fig)

# More sensitive complex-amplitude RMSE, not just intensity comparison.
Ns = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192]
rmse_t, rmse_r = [], []
for N in Ns:
    tp, rp = truncated_pole_scattering(x, N, n)
    rmse_t.append(float(np.sqrt(np.mean(np.abs(t-tp)**2))))
    rmse_r.append(float(np.sqrt(np.mean(np.abs(r-rp)**2))))
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.loglog(2*np.array(Ns)+1, rmse_t, 'o-', label='Transmission amplitude RMSE')
ax.loglog(2*np.array(Ns)+1, rmse_r, 's-', label='Reflection amplitude RMSE')
ax.set(xlabel='Number of included poles', ylabel='RMS complex-amplitude error',
       title='Convergence with the number of retained resonances')
ax.grid(True, which='both', alpha=.25)
ax.legend(); fig.tight_layout(); fig.savefig(OUT/'02_convergence.png', dpi=180); plt.close(fig)

m = np.arange(-7, 8)
poles = resonance_poles(m, n)
fig, ax = plt.subplots(figsize=(8, 4.0))
ax.scatter(poles.real, poles.imag, s=35, label='Analytic outgoing poles')
ax.axhline(0, color='gray', lw=.9, ls=':')
ax.set(xlabel=r'${\rm Re}(k_0 d)$', ylabel=r'${\rm Im}(k_0 d)$',
       title='Scattering poles in the lower complex-frequency half-plane')
ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
fig.savefig(OUT/'03_complex_poles.png', dpi=180); plt.close(fig)

n_new = 3.06
m_positive = np.arange(1, 9)
p0 = resonance_poles(m_positive, n)
p1 = resonance_poles(m_positive, n_new)
pert = np.array([first_order_pole_shift(int(mi), n_new-n, n) for mi in m_positive])
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(m_positive, (p1-p0).real, 'o-', label='Exact pole shift: real part')
ax.plot(m_positive, pert.real, 'x--', label='First-order estimate: real part')
ax.plot(m_positive, (p1-p0).imag, 's-', label='Exact pole shift: imaginary part')
ax.plot(m_positive, pert.imag, '+--', label='First-order estimate: imaginary part')
ax.set(xlabel='Positive mode number m', ylabel=r'Pole shift $\Delta(k_0d)$',
       title='Refractive-index perturbation: n = 3.00 to 3.06')
ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT/'04_pole_sensitivity.png', dpi=180); plt.close(fig)

with open(OUT/'convergence_metrics.csv', 'w', newline='', encoding='utf8') as fh:
    w = csv.writer(fh)
    w.writerow(['modes_per_side', 'pole_count', 't_complex_rmse', 'r_complex_rmse'])
    for N, et, er in zip(Ns, rmse_t, rmse_r):
        w.writerow([N, 2*N+1, f'{et:.9e}', f'{er:.9e}'])

balance = np.max(np.abs(np.abs(t)**2 + np.abs(r)**2 - 1))
resid = np.max(np.abs(denominator(poles, n)))
print(f'Energy conservation: max||t|²+|r|²-1| = {balance:.3e}')
print(f'Complex-pole test: max|D(x_m)| = {resid:.3e}')
print(f'Q (m=3) = {quality_factor(3, n):.4f}')
print(f'Pole approximation at N={Ns[-1]} (total {2*Ns[-1]+1} poles):')
print(f'  transmission amplitude RMSE: {rmse_t[-1]:.6e}')
print(f'  reflection amplitude RMSE:   {rmse_r[-1]:.6e}')
print('Saved 4 figures and convergence_metrics.csv in outputs/')
