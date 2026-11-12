"""Independent 2D dielectric-cylinder Mie scattering and outgoing-mode examples."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cylinder_mie import cylinder_scattering_efficiency, cylinder_coefficient, find_cylinder_pole

out=Path(__file__).resolve().parent/'outputs';out.mkdir(exist_ok=True)
n=3.0
x=np.linspace(.15,7.0,1500)
Q=cylinder_scattering_efficiency(x,n,18)
fig,ax=plt.subplots(figsize=(9,4.7))
ax.plot(x,Q,color='black',lw=2,label='Total 2D TMz scattering efficiency')
for m in range(4):
    w=1 if m==0 else 2
    Qm=2*w*np.abs(cylinder_coefficient(m,x,n))**2/x
    ax.plot(x,Qm,lw=1.1,label=f'Angular channel m={m}')
ax.set(xlabel=r'Size parameter $x=k_0 R$',ylabel=r'Scattering efficiency $Q_{sca}$',
       title='Mie-type scattering of a dielectric cylinder (n=3)')
ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8)
fig.tight_layout();fig.savefig(out/'05_cylinder_scattering.png',dpi=180);plt.close(fig)

# Local root search: representative outgoing poles, not an exhaustive mode census.
initial_guesses={0:[1.3-.12j,2.36-.12j,3.4-.12j],
                 1:[.76-.086j,1.81-.11j,2.87-.11j],
                 2:[1.21-.04j,2.28-.08j,3.35-.10j],
                 3:[1.65-.018j,2.73-.059j,3.81-.085j]}
fig,ax=plt.subplots(figsize=(8,4.5))
for m,seeds in initial_guesses.items():
    poles=[find_cylinder_pole(m,s,n) for s in seeds]
    ax.scatter(np.real(poles),np.imag(poles),s=50,label=f'm={m}')
ax.axhline(0,color='gray',lw=.8,ls='--')
ax.set(xlabel=r'${\rm Re}(k_0 R)$',ylabel=r'${\rm Im}(k_0 R)$',
       title='Representative outgoing resonances of the 2D dielectric cylinder')
ax.grid(alpha=.2);ax.legend(ncol=2)
fig.tight_layout();fig.savefig(out/'06_cylinder_complex_poles.png',dpi=180);plt.close(fig)

conservation_error=max(np.max(np.abs(np.abs(1+2*cylinder_coefficient(m,x,n))**2-1)) for m in range(8))
print(f'Cylinder partial-wave unitarity: max error = {conservation_error:.3e}')
print('Saved outputs/05_cylinder_scattering.png and 06_cylinder_complex_poles.png')
