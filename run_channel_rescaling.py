"""Stage 3: scattering-channel rescaling and artificial-channel-pole diagnostics.

Run from project directory: python run_channel_rescaling.py
Saves 4 figures and 2 CSV files in outputs/. No FDTD solver is needed.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import hankel1,hankel2
from channel_rescaling import (
    channel_scattering, channel_multiplier, mie_s, build_channel_expansion,
    find_channel_pole, residue_channel_pole, residue_physical_pole,
)
from cylinder_pole_expansion import complex_rmse

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
M=3; N=3.0


def write_csv(filename,rows):
    with (OUT/filename).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    ex=build_channel_expansion(m=M,n=N)
    pp=ex.physical_poles; qq=ex.channel_poles
    print('Channel-rescaling benchmark | 2D dielectric cylinder (m=3, n=3)')
    print(f'Argument-principle zero counts: D_m={ex.physical_winding}, H2_m={ex.channel_winding}')
    for p in pp:print(f'physical resonance: {p.real:.10f}{p.imag:+.10f}j')
    for p in qq:print(f'channel pole (H2 zero): {p.real:.10f}{p.imag:+.10f}j')

    # Fig 10. Complex-frequency landscape; same colorbar for all bases.
    xr=np.linspace(.70,4.32,340)
    yi=np.linspace(-.35,2.22,245)
    Z=xr[np.newaxis,:]+1j*yi[:,np.newaxis]
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.3),sharex=True,sharey=True)
    for ax,basis,name in zip(axes,('mie','exponential','hankel'),
                              ('Standard cylindrical channels','Exponential channel rescaling',
                               'Hankel channel rescaling')):
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            amp=np.log10(np.abs(channel_scattering(M,Z,N,basis)))
        im=ax.pcolormesh(xr,yi,np.clip(amp,-2.0,2.0),vmin=-2.,vmax=2.,cmap='viridis',rasterized=True,shading='auto')
        ax.scatter(pp.real,pp.imag,facecolors='none',edgecolors='orangered',s=62,lw=1.6,
                   marker='o',label='Physical resonances')
        if basis=='hankel':
            ax.scatter(qq.real,qq.imag,color='white',edgecolor='black',s=65,marker='X',
                       label='Channel pole')
        ax.axhline(0,color='white',lw=.8,alpha=.65)
        ax.set(title=name,xlabel=r'$\mathrm{Re}(k_0R)$')
        ax.grid(alpha=.12)
    axes[0].set_ylabel(r'$\mathrm{Im}(k_0R)$')
    axes[-1].legend(loc='upper right',fontsize=8)
    fig.subplots_adjust(wspace=.08,top=.85,bottom=.17,left=.07,right=.88)
    cax=fig.add_axes([.895,.18,.014,.65])
    fig.colorbar(im,cax=cax,label=r'$\log_{10}|S_m(k_0R)|$')
    fig.suptitle('Scattering-channel normalization changes the complex-plane pole map',y=.965)
    fig.savefig(OUT/'10_channel_rescaling_complex_plane.png',dpi=190,bbox_inches='tight');plt.close(fig)

    # Fig 11. Real-frequency phases differ, but amplitudes map exactly back to S0.
    x=np.linspace(1.5,3.5,701)
    s0=channel_scattering(M,x,N,'mie')
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.4))
    for basis,label in [('mie','Standard'),('exponential','Exponential'),('hankel','Hankel')]:
        sk=channel_scattering(M,x,N,basis)
        axes[0].plot(x,np.unwrap(np.angle(sk)),lw=1.5,label=label)
        axes[1].semilogy(x,np.maximum(abs(sk/channel_multiplier(M,x,basis)-s0),1e-17),
                          label=label)
    axes[0].set(xlabel=r'$x=k_0 R$',ylabel='Unwrapped phase of S [rad]',
                title='Scattering phase depends on channel definition')
    axes[1].set(xlabel=r'$x=k_0 R$',ylabel='Absolute complex-amplitude difference',
                title='Undoing channel rescaling recovers the physical S')
    for a in axes:a.grid(alpha=.25);a.legend(fontsize=9)
    fig.suptitle('Identical physical outgoing/incoming solutions under three bases')
    fig.tight_layout();fig.savefig(OUT/'11_channel_rescaling_real_axis.png',dpi=190);plt.close(fig)

    # Fig 12. Completeness test: physical poles alone vs physical + channel pole.
    order=22
    exact=channel_scattering(M,x,N,'hankel')
    full=ex.evaluate(x,order,True)
    missing=ex.evaluate(x,order,False)
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.45))
    axes[0].plot(x,exact.real,color='black',lw=2,label='Exact Re(S)')
    axes[0].plot(x,full.real,ls='--',lw=1.5,color='tab:green',label='3 physical + 1 channel')
    axes[0].plot(x,missing.real,ls=':',lw=1.8,color='tab:red',label='3 physical only')
    axes[0].set(xlabel=r'$x=k_0R$',ylabel=r'$\mathrm{Re}(S_m^\mathrm{Hankel})$',
                title='Complex scattering reconstruction (real part)')
    orders=[0,1,2,3,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    ef=[complex_rmse(ex.evaluate(x,k,True),exact) for k in orders]
    em=[complex_rmse(ex.evaluate(x,k,False),exact) for k in orders]
    axes[1].semilogy(orders,ef,'-o',ms=3.8,lw=1.4,label='4 poles: physical + channel')
    axes[1].semilogy(orders,em,'-s',ms=3.8,lw=1.4,label='3 physical poles only')
    axes[1].set(xlabel='Analytic background order',ylabel='Complex-amplitude RMSE',
                title='Channel-pole completeness controls convergence')
    for a in axes:a.grid(alpha=.25,which='both');a.legend(fontsize=8)
    fig.suptitle('Rescaled scattering matrix needs both physical and channel poles')
    fig.tight_layout();fig.savefig(OUT/'12_channel_pole_completeness.png',dpi=190);plt.close(fig)

    # Fig 13. Classification of poles in upper/lower complex half planes.
    fig,ax=plt.subplots(figsize=(7.6,4.85))
    ax.scatter(pp.real,pp.imag,s=95,facecolor='none',edgecolor='tab:red',lw=2,
               label='Outgoing resonances: D_m(x)=0')
    ax.scatter(qq.real,qq.imag,s=125,marker='X',color='tab:purple',
               label=r'Channel pole: $H_m^{(2)}(x)=0$')
    for i,p in enumerate(pp,1):ax.annotate(f'P{i}',(p.real,p.imag),xytext=(6,-15),textcoords='offset points')
    for i,p in enumerate(qq,1):ax.annotate(f'C{i}',(p.real,p.imag),xytext=(7,8),textcoords='offset points')
    ax.axhline(0,color='black',lw=.8)
    ax.set(xlabel=r'$\mathrm{Re}(k_0R)$',ylabel=r'$\mathrm{Im}(k_0R)$',
           title='Physical resonances vs channel-induced pole (m=3)',xlim=(.85,4.25),ylim=(-.38,2.08))
    ax.grid(alpha=.22);ax.legend(loc='upper right',fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'13_physical_vs_channel_poles.png',dpi=190);plt.close(fig)

    rows=[]
    for k,e1,e2 in zip(orders,ef,em):
        rows.append({'m':M,'index_n':N,'background_order':k,
                     'rmse_all_4_poles':e1,'rmse_physical_3_only':e2})
    write_csv('channel_rescaling_convergence.csv',rows)
    p_rows=[]
    for basis in ('mie','exponential','hankel'):
        for p in pp:
            res=residue_physical_pole(M,p,N,basis)
            p_rows.append({'basis':basis,'kind':'physical','pole_re':p.real,'pole_im':p.imag,
                           'residue_re':res.real,'residue_im':res.imag})
        if basis=='hankel':
            for p in qq:
                res=residue_channel_pole(M,p,N)
                p_rows.append({'basis':basis,'kind':'channel','pole_re':p.real,
                               'pole_im':p.imag,'residue_re':res.real,'residue_im':res.imag})
    write_csv('channel_rescaling_poles.csv',p_rows)
    print(f'complex RMSE (all physical + channel, B{order}): {complex_rmse(full,exact):.3e}')
    print(f'complex RMSE (physical only, B{order}): {complex_rmse(missing,exact):.3e}')
    print('Saved figures 10-13 and channel_rescaling_*.csv in outputs/')


if __name__=='__main__':main()
