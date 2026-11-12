"""Validated local 2D pole-residue reconstruction of exact cylinder Mie scattering.

Run: python run_cylinder_pole_expansion.py
Generates 3 figures, a convergence CSV and a pole/residue CSV.
Uses a local Cauchy background and NOT a global Maxwell RSE calculation.
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cylinder_mie import cylinder_coefficient
from cylinder_pole_expansion import build_local_expansion, complex_rmse

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(parents=True, exist_ok=True)
N = 3.0
CENTER, RADIUS = 2.6, 1.65
POLE_SEEDS = {
    2: [1.21 - .04j, 2.28 - .08j, 3.35 - .10j],
    3: [1.65 - .018j, 2.73 - .059j, 3.81 - .085j],
}

def make_expansion(m):
    return build_local_expansion(m, POLE_SEEDS[m], n=N, center=CENTER,
                                 radius=RADIUS, max_background_order=24,
                                 quadrature_points=4096)


def main():
    xs = np.linspace(1.50, 3.50, 701)
    exps = {m: make_expansion(m) for m in (2,3)}
    metrics = []
    poles_csv = []

    for m, exp in exps.items():
        exact = cylinder_coefficient(m, xs, N)
        for pidx,(pole,res) in enumerate(zip(exp.poles,exp.residues),start=1):
            poles_csv.append({'m':m,'order_by_contour_center':pidx,
                              'pole_re':float(pole.real),'pole_im':float(pole.imag),
                              'Q':float(pole.real/(-2*pole.imag)),
                              'residue_re':float(res.real),'residue_im':float(res.imag),
                              'number_of_contour_zeros':exp.pole_count_winding})
        for k in (-1,0,2,4,8,12,16,20,24):
            for npoles in range(4):
                pred=exp.evaluate(xs,npoles,k)
                metrics.append({'m':m,'included_poles':npoles,'background_order':k,
                                'amplitude_rmse':complex_rmse(pred,exact),
                                'max_abs_amplitude_error':float(np.max(np.abs(pred-exact)))})

    with (OUTPUT/'cylinder_pole_expansion_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    with (OUTPUT/'cylinder_resonances_and_residues.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(poles_csv[0]));writer.writeheader();writer.writerows(poles_csv)

    # Figure 07: Show pole-only vs finite analytic background vs exact Mie response.
    exp = exps[3]
    exact = cylinder_coefficient(3,xs,N)
    pole_only=exp.evaluate(xs,3,-1)
    approx2=exp.evaluate(xs,3,2)
    approx12=exp.evaluate(xs,3,12)
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.6))
    ax=axes[0]
    ax.plot(xs,np.abs(exact)**2,label='Exact Mie',lw=2.3,color='black')
    ax.plot(xs,np.abs(pole_only)**2,label='3 poles only',lw=1.2,ls=':')
    ax.plot(xs,np.abs(approx2)**2,label='3 poles + B2',lw=1.4,ls='--')
    ax.plot(xs,np.abs(approx12)**2,label='3 poles + B12',lw=1.3,ls='-.')
    ax.set(xlabel=r'$x=k_0 R$',ylabel=r'$|a_3(x)|^2$',title='Resonant partial-wave intensity')
    ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend(fontsize=8)
    ax=axes[1]
    ax.plot(xs,exact.real,color='black',lw=2,label='Re exact')
    ax.plot(xs,exact.imag,color='dimgray',lw=2,label='Im exact')
    ax.plot(xs,approx12.real,ls='--',lw=1.5,label='Re reconstruction')
    ax.plot(xs,approx12.imag,ls='--',lw=1.5,label='Im reconstruction')
    ax.set(xlabel=r'$x=k_0 R$',ylabel=r'Complex $a_3(x)$',title='Complex-amplitude comparison')
    ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('2D cylindrical scattering: local pole + contour-background reconstruction')
    fig.tight_layout();fig.savefig(OUTPUT/'07_cylinder_pole_reconstruction.png',dpi=200);plt.close(fig)

    # Figure 08: Two independent truncation axes: pole count and analytic background order.
    fig, axes=plt.subplots(1,2,figsize=(12.2,4.5))
    ax=axes[0]
    for k in (2,4,8,12):
        vals=[complex_rmse(exp.evaluate(xs,p,k),exact) for p in range(4)]
        ax.semilogy(range(4),vals,marker='o',label=f'B{k}')
    ax.set(xlabel='Number of poles included (0–3)',ylabel='Complex-amplitude RMSE',
           title='Pole-count dependence at fixed background order',xticks=range(4))
    ax.grid(alpha=.25,which='both');ax.legend()
    ax=axes[1]
    orders=[0,1,2,3,4,6,8,10,12,14,16,18,20,22,24]
    vals_full=[complex_rmse(exp.evaluate(xs,3,k),exact) for k in orders]
    vals_two=[complex_rmse(exp.evaluate(xs,2,k),exact) for k in orders]
    ax.semilogy(orders,vals_full,marker='o',label='3 poles (complete set)')
    ax.semilogy(orders,vals_two,marker='s',label='2 poles (one omitted)')
    ax.set(xlabel='Analytic background polynomial order K',ylabel='Complex-amplitude RMSE',
           title='Background convergence requires all enclosed poles')
    ax.grid(alpha=.25,which='both');ax.legend()
    fig.suptitle('Independent truncation diagnostics for cylindrical channel m=3')
    fig.tight_layout();fig.savefig(OUTPUT/'08_cylinder_pole_convergence.png',dpi=200);plt.close(fig)

    # Figure 09: second angular order as a cross-check (not only an m=3 fit).
    fig, axes=plt.subplots(1,2,figsize=(12.2,4.5))
    for ax,m in zip(axes,(2,3)):
        e=exps[m]
        direct=cylinder_coefficient(m,xs,N)
        est=e.evaluate(xs,3,16)
        ax.plot(xs,np.abs(direct)**2,lw=2.1,color='black',label='Exact Mie')
        ax.plot(xs,np.abs(est)**2,lw=1.4,ls='--',label='Local pole expansion')
        ax.set(xlabel=r'$x=k_0 R$',ylabel=rf'$|a_{m}(x)|^2$',
               title=rf'Channel m={m} | amplitude RMSE={complex_rmse(est,direct):.2e}')
        ax.set_ylim(bottom=0);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Independent two-channel verification (3 poles + background B16)')
    fig.tight_layout();fig.savefig(OUTPUT/'09_cylinder_two_channel_validation.png',dpi=200);plt.close(fig)

    print('Local 2D cylinder resonant-pole expansion (exact Mie benchmark)')
    print(f'Contour center={CENTER:.2f}, radius={RADIUS:.2f}; real interval x=1.50..3.50')
    for m,e in exps.items():
        direct=cylinder_coefficient(m,xs,N)
        print(f'm={m}: certified winding count = {e.pole_count_winding}, seeded poles = {len(e.poles)}')
        for p,res in zip(e.poles,e.residues):
            print(f'   pole={p.real:.8f}{p.imag:+.8f}j | residue={res.real:.6g}{res.imag:+.6g}j')
        for background_order in (-1,2,8,12,16,20):
            err=complex_rmse(e.evaluate(xs,3,background_order),direct)
            print(f'   all poles, background order {background_order:>2d}: complex RMSE {err:.3e}')
    print('Saved three figures, convergence metrics CSV, and pole-residue CSV to outputs/')


if __name__ == '__main__':
    main()
