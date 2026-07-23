import os
import CLAY.objects as objs 
import CLAY.datahandling as dh
import CLAY.model as mod
import numpy as np
import matplotlib.pyplot as plt


directory = "CLAY/test_data/GRAVITY/"
path_list = [directory+filename for filename in os.listdir(directory) if filename.endswith(".fits")]


# =============================================================================
# Importing data from a list of OIFITS files
# =============================================================================
ds = dh.dataSet(path_list)

# =============================================================================
# A few manipulations can be perfromed (merging, binnning, etc...)
# =============================================================================
# ds.merge_obs(merge_span=8)
# ds.spectral_binning(bin_factor=200)
# =============================================================================
# Can take a look at the data 
# =============================================================================
# ax1 = ds.plot_Vcurve(logscale=False, legend=True, axis_unit="MLamb")
# ax2 = ds.plot_CPhi(axis_unit="MLamb", legend=True)
# ax2 = ds.plot_uv(axis_unit='MLamb')
# ax3 = ds.plot_spectrum()



# =============================================================================
# Extract an emission line ('tellcorr' says if tellurics should be corrected)
# =============================================================================
Brg = dh.emission_line( ds, line='BrG', vContMax=2000, vContMin=-2000, tellcorr=False )

# =============================================================================
# Can take a look at the data
# =============================================================================
# Brg.plot_spectrum(x_unit='vel',vmin=-500, vmax=500)
# Brg.plot_Vcurve()


# =============================================================================
# Extract pure-line quantities and define bounds for the line
# =============================================================================
# Brg.normalize_dphi( vmin=-500, vmax=500, polyorder=1)
# Brg.correct_rv( systematic_vr=-16.1, dist=140 )
# Brg.define_line( Flux_threshold=0.27 )
# Brg.compute_pureline()
# =============================================================================
# Alternative way (all done together, but less versatile)
# =============================================================================
Brg.prepare_pureline( vmin=-500, vmax=500, poly=1, rv=-16.1, dist=140, Flx_thr=0.27)

# =============================================================================
# Recap of all differential data
# =============================================================================
Brg.plot_data( vmin=-500, vmax=500)


# =============================================================================
# Compute and show the astrometry from pure-line phases
# =============================================================================
Brg.compute_astrometry()
axis_lim=((80,-80),(-80,80))
Brg.plot_astrometry(vmin=-300, vmax=300, axis_lim=axis_lim, axis_unit="muas")

# =============================================================================
# Fitting of the pure-line visibility
# =============================================================================

# =============================================================================
# Define a synthetic Gaussian disk
# =============================================================================
u_line, v_line = Brg.B_line[0]/Brg.wvl_line[:,None,:], Brg.B_line[1]/Brg.wvl_line[:,None,:]
mG = objs.model( u=u_line, v=v_line, wvl=Brg.wvl_line, model='Gauss', params={"a":0.2,"inc":10, "pa":0} )

# =============================================================================
# Create and prepare a fitter that uses our Gaussian disk and pure-line data
# =============================================================================
### Order for Gaussian: "x0","y0","a","inc", "pa", should be respected for now
### Do not include parameters you do not want to fit
sigma = [ 0.02]
lims = [[   0 ], 
        [   2 ]]

fit = mod.Line_fitter(Brg, mG)
fit.prepare_run( nwalkers=20, nsteps=1000, 
                pos_init={"a":0.05}, prior="gaussian", 
                doNotFit = ["x0","y0", "inc","pa"],
                lims=lims, sigma=sigma )

# =============================================================================
# Run the fit, and recover things to plot and display
# =============================================================================
samp = fit.run_fit()

mf = mod.get_bestModel( fit, samp )
ax = mod.plot_chain(fit, samp)
figc = mod.plot_corner(fit, samp)

fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].hlines(1, 0, 1.1*fit.freqs.max(), color='k', lw=1.1)
residus = (abs(mf.Vtot)**2-Brg.V_pl**2)/(2*Brg.V_pl*Brg.e_V_pl )
for i in range(Brg.Nobs):
    for j in range(Brg.Nbl) :
        ax[0].errorbar(fit.freqs[i,j], Brg.V_pl[i,j]**2, yerr=2*Brg.V_pl[i,j]*Brg.e_V_pl[i,j], markerfacecolor='k', markeredgecolor=Brg.clbr_Nbl[j], ecolor='k', ls='None', capsize=3, marker='o', markersize=5, zorder=0, alpha=0.5 )
        ax[0].scatter( fit.freqs[i,j], abs(mf.Vtot[i,j])**2, c=Brg.clbr_Nbl[j], marker="+" )
        
        ax[1].scatter( fit.freqs[i,j], residus[i,j], c=Brg.clbr_Nbl[j], marker="+" )
ax[1].hlines(0,0,1.1*fit.freqs.max(), ls='dashed', color='k')
ax[0].grid(), ax[0].set_xlim(0,1.1*fit.freqs.max())
ax[0].set_ylim(0,1.05)
ax[0].set_ylabel("Visibility squared")
ax[1].grid(), ax[1].set_xlim(0,1.1*fit.freqs.max())
ax[1].set_ylim(-1.1*np.nanmax(np.sqrt(residus**2)), 1.1*np.nanmax(np.sqrt(residus**2)))
ax[1].set_ylabel(r"Residuals ($\sigma$)")

plt.show()