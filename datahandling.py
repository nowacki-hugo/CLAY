import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from itertools import combinations


def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)

class dataSet:
    ### a group of data, that can be basically manipulated (stacking, binning).
    ### Need to be able to define a spectral region of interest (reject some intervals, cut to certain intervals)
    ### Need to have an easy access to the main data : (u-v), Flux, V2, CPhi, dPhi
    def __init__(self, path):
        ### Give a path to the data
        ### Give the instrument name : 'GRAVITY', 'SPICA', 'MIRCX', 'MYSTIC' to validate.
        ### All the data need to be a matrix of shape (Nwl, Nbl, Nobs)
        self.merged = False
        self.Nobs, self.Nwl, self.Nbl, self.Ntr = len(path), [], [], []
        self.Flux, self.e_Flux = [], []
        self.Lambda = []
        self.V2, self.e_V2 = [], []
        self.CPhi, self.e_CPhi = [], []
        self.dphi, self.e_dphi = [], []
        self.u, self.v = [], []
        self.freqsCPhi = []
        self.array = []
        for k, p in enumerate(path) :
            ########### INSIDE THE FITS FILE
            hdul = fits.open(p)
            self.array.append( hdul['OI_ARRAY'].header["ARRNAME"] )
            if self.array[-1] == "VLTI" :
                try : ### If TELLURICS have been computed with Pmoired for this data.
                    self.Nwl.append( hdul["TELLURICS"].header["NAXIS2"] )
                    self.Lambda.append( hdul["TELLURICS"].data.field("CORR_WAVE") )
                    self.Flux.append( hdul["TELLURICS"].data.field("CORR_SPEC") )
                except KeyError :
                    try :### Otherwise, try combined polar mode
                        self.Nwl.append( hdul["OI_WAVELENGTH", 10].header["NAXIS2"] )
                        self.Lambda.append( hdul["OI_WAVELENGTH", 10].data.field("EFF_WAVE") )
                        self.Flux.append( hdul["OI_FLUX", 10].data.field("FLUX") )
                        self.e_Flux.append( hdul["OI_FLUX", 11].data.field("FLUXERR") )
                    except KeyError :### otherwise, try split polar mode
                        self.Nwl.append( hdul["OI_WAVELENGTH", 11].header["NAXIS2"] )
                        self.Lambda.append( hdul["OI_WAVELENGTH", 11].data.field("EFF_WAVE") )
                        self.Flux.append( hdul["OI_FLUX", 11].data.field("FLUX") )
                        self.e_Flux.append( hdul["OI_FLUX", 11].data.field("FLUXERR") )
                try :
                    self.V2.append( hdul["OI_VIS2", 10].data.field("VIS2DATA") )
                    self.e_V2.append( hdul["OI_VIS2", 10].data.field("VIS2ERR") )
                    self.CPhi.append( hdul["OI_T3", 10].data.field("T3PHI") )
                    self.e_CPhi.append( hdul["OI_T3", 10].data.field("T3PHIERR") )
                    self.u.append( hdul["OI_VIS2", 10].data.field("UCOORD") )
                    self.v.append( hdul["OI_VIS2", 10].data.field("VCOORD") )
                        
                except KeyError:
                    self.V2.append( hdul["OI_VIS2", 11].data.field("VIS2DATA") )
                    self.e_V2.append( hdul["OI_VIS2", 11].data.field("VIS2ERR") )
                    self.CPhi.append( hdul["OI_T3", 11].data.field("T3PHI") )
                    self.e_CPhi.append( hdul["OI_T3", 11].data.field("T3PHIERR") )
                    self.u.append( hdul["OI_VIS2", 11].data.field("UCOORD") )
                    self.v.append( hdul["OI_VIS2", 11].data.field("VCOORD") )
                
     
            else :
                self.Nwl.append( hdul["OI_WAVELENGTH"].header["NAXIS2"] )
                self.Lambda.append( hdul["OI_WAVELENGTH"].data.field("EFF_WAVE") )
                self.Flux.append( hdul["OI_FLUX"].data.field("FLUXDATA") )
                self.e_Flux.append( hdul["OI_FLUX"].data.field("FLUXERR") )
                self.V2.append( hdul["OI_VIS2"].data.field("VIS2DATA") )
                self.e_V2.append( hdul["OI_VIS2"].data.field("VIS2ERR") )
                self.CPhi.append( hdul["OI_T3"].data.field("T3PHI") )
                self.e_CPhi.append( hdul["OI_T3"].data.field("T3PHIERR") )
                self.u.append( hdul["OI_VIS2"].data.field("UCOORD") )
                self.v.append( hdul["OI_VIS2"].data.field("VCOORD") )
                
            self.Nbl.append( hdul["OI_VIS2"].header["NAXIS2"] )
            self.Ntr.append( hdul["OI_T3"].header["NAXIS2"] )
            

            
            if k == 0 :
                self.dicname = {i:n for i,n in zip(hdul['OI_ARRAY'].data.field('STA_INDEX'),
                                             hdul['OI_ARRAY'].data.field('STA_NAME'))}
                self.BL_idx = [ (i,j) for i,j in hdul['OI_VIS'].data.field('STA_INDEX') ]
                self.TR_idx = [ (i,j,k) for i,j,k in hdul['OI_T3'].data.field('STA_INDEX')]
                self.BLs = [self.dicname[i]+'-'+self.dicname[j] for i,j in hdul['OI_VIS'].data.field('STA_INDEX')]
                self.TRs = [self.dicname[i]+'-'+self.dicname[j]+'-'+self.dicname[k] for i,j,k in hdul['OI_T3'].data.field('STA_INDEX')]
            
            
            hdul.close()
            ########### OUTSIDE THE FITS FILE
        
        if all_equal(self.array) :
            self.array = self.array[0]
        
        if all_equal(self.Nwl) and all_equal(self.Nbl) and all_equal(self.Ntr):
            self.Nwl, self.Nbl, self.Ntr = self.Nwl[0], self.Nbl[0], self.Ntr[0]
            self.Lambda = np.asarray(self.Lambda)*1e6
            self.u, self.v = np.asarray(self.u)[:,:,None]/self.Lambda[:,None,:], np.asarray(self.v)[:,:,None]/self.Lambda[:,None,:]
            self.Flux  = np.asarray(self.Flux)
            self.V2, self.e_V2 = np.asarray(self.V2), np.asarray(self.e_V2)
            self.CPhi, self.e_CPhi = np.asarray(self.CPhi), np.asarray(self.e_CPhi)
        # else :
            ##### Pad the data in order to transform in arrays
            
        self.wl0 = self.Lambda.mean()
        self.freqs = np.hypot( self.u, self.v )
        
        
        self.freqsCPhi = []
        for idx in self.TR_idx :
            a = list(combinations(idx, 2))
            f=[]
            for test in a :
                i = self.BL_idx.index(test)
                f.append(self.freqs[:,i,:])
            i = np.argmax( np.asarray(f).mean(axis=(1,2)) )
            self.freqsCPhi.append( f[i] )
        self.freqsCPhi = np.transpose(np.asarray(self.freqsCPhi),(1,0,2))
        
        self.e_V2[self.e_V2 < 1e-4] = 1e-4
        self.e_V2[self.e_V2 > 1] = 1
        self.e_CPhi[self.e_CPhi <= 1] = 1
        self.e_CPhi[self.e_CPhi > 180] = 180
        self.e_CPhi[np.isnan(self.e_CPhi)] = 1
        
        if self.Nbl == 6:
            self.Ntel = 4
        elif self.Nbl == 15:
            self.Ntel = 6
            
        self.clbr_Nbl = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Nbl))
        self.clbr_Nobs = plt.cm.nipy_spectral_r(np.linspace(0.1, 0.9, self.Nobs))
        self.clbr_Ntel = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Ntel))
        self.clbr_Ntr = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Ntr))
    
    # def spectral_crop(self, wl_range, inout=in):
        ### A function that extracts a spectral region inside or inside (inout) a given wavelength range wl_range
        ### Need to be able to pass several ranges (i.e. to define the continuum)




    def spectral_binning(self, bin_factor=5):
        """
        Binning spectral pondéré avec interpolation des NaN.
        Compatible multi-dimensions (n'importe quelle forme [..., channels]).
    
        Étapes :
        1. interpolation des NaN sur le dernier axe
        2. centrage pour être multiple de bin_factor
        3. binning pondéré par 1/σ²
        """
    
        # ----------- UTILITAIRES GÉNÉRIQUES ----------- #
    
        def __interpolate_nan_1d(arr):
            """
            Interpole un vecteur 1D contenant des NaN.
            - interpolation linéaire sur les valeurs finies
            - si tout est NaN → renvoie arr
            - si NaN en bordure → extrapolation linéaire
            """
            x = np.arange(arr.size)
            mask = np.isfinite(arr)
    
            if mask.sum() == 0:
                return arr  # rien à interpoler, tout NaN
    
            f = interp1d(
                x[mask], arr[mask],
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            return f(x)
    
        def __interpolate_nan_last_axis(arr):
            """
            Applique interpolate_nan_1d sur le dernier axe,
            en préservant la forme générale du tableau N-D.
            """
            arr = np.asarray(arr)
            reshaped = arr.reshape(-1, arr.shape[-1])  # collapse all dims except last
            out = np.empty_like(reshaped)
    
            for i in range(reshaped.shape[0]):
                out[i] = __interpolate_nan_1d(reshaped[i])
    
            return out.reshape(arr.shape)
    
        def __centered_slice_nd(arr, factor):
            """
            Tronque arr sur son dernier axe de manière centrée
            pour une longueur divisible par factor.
            """
            n = arr.shape[-1]
            n_trim = (n // factor) * factor
            remove = n - n_trim
            left = remove // 2
            right = n - (remove - left)
    
            slicer = [slice(None)] * arr.ndim
            slicer[-1] = slice(left, right)
            return arr[tuple(slicer)]
    
        def __reshape_for_binning(arr, factor):
            """
            Transforme arr [..., N] → [..., nbins, factor].
            """
            new_shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
            return arr.reshape(new_shape)
    
        # ----------- BINNAGE PONDÉRÉ ----------- #
    
        def __bin_weighted(values, errors, factor):
            # Centrage
            v = __centered_slice_nd(values, factor)
            e = __centered_slice_nd(errors, factor)
    
            # Regroupement par bin
            v_r = __reshape_for_binning(v, factor)
            e_r = __reshape_for_binning(e, factor)
    
            # weight = 1/sigma**2
            w = 1.0 / e_r**2
    
            # Weighted average
            W = w.sum(axis=-1)
            vw = (v_r * w).sum(axis=-1)
    
            mean = np.where(W > 0, vw / W, np.nan)
    
            # weighted uncertainty = sqrt(1 / sum(w))
            err = np.where(W > 0, np.sqrt(1.0 / W), np.nan)
    
            return mean, err
    
        # --------------- STEP 1 : NaN INTERPOLATION --------------- #
        self.Lambda = __interpolate_nan_last_axis(self.Lambda)
        
        self.u, self.v = __interpolate_nan_last_axis(self.u), __interpolate_nan_last_axis(self.v)
        
        self.freqs   = __interpolate_nan_last_axis(self.freqs)
        self.freqsCPhi   = __interpolate_nan_last_axis(self.freqsCPhi)
        
        self.V2   = __interpolate_nan_last_axis(self.V2)
        self.e_V2  = __interpolate_nan_last_axis(self.e_V2)
    
        self.Flux  = __interpolate_nan_last_axis(self.Flux)
        # self.e_Flux = __interpolate_nan_last_axis(self.e_Flux)
    
        self.CPhi   = __interpolate_nan_last_axis(self.CPhi)
        self.e_CPhi  = __interpolate_nan_last_axis(self.e_CPhi)
    
        # --------------- STEP 2 : WEIGHTED BINNING --------------- #
        self.Lambda, duh = __bin_weighted(self.Lambda, np.ones(self.Lambda.shape),   bin_factor)
        
        self.u, duh = __bin_weighted(self.u, np.ones(self.u.shape),   bin_factor)
        self.v, duh = __bin_weighted(self.v, np.ones(self.v.shape),   bin_factor)
        
        self.freqs, duh = __bin_weighted(self.freqs, np.ones(self.freqs.shape),   bin_factor)
        self.freqsCPhi, duh = __bin_weighted(self.freqsCPhi, np.ones(self.freqsCPhi.shape),   bin_factor)
        self.V2,   self.e_V2   = __bin_weighted(self.V2, self.e_V2,   bin_factor)
        
        self.V2   = __interpolate_nan_last_axis(self.V2)
        
        self.Flux, duh = __bin_weighted(self.Flux, np.ones(self.Flux.shape), bin_factor)
        self.CPhi, self.e_CPhi = __bin_weighted(self.CPhi, self.e_CPhi, bin_factor)


    def merge_obs(self, merge_span=2):
    ### A function that merges in the Nobs dimension over merge_span files to reduce uncertainty on interf quantities 
    ### Caveat : uncertainty decreased on frequencies if too long 
        def __merge_partial( arr, err_arr, period, axsize ):
            lim = axsize//period
            if period <= 1:
                return arr
            elif lim <= 0 :
                medval = np.average( arr, weights=1/err_arr**2, axis=0 )
                return( np.asarray([medval]) )
            else :
                out_arr = []
                for i in range(lim) :
                    if i+1 < lim :
                        medval = np.average( arr[period*i:period*(i+1)], 
                                            weights=1/err_arr[period*i:period*(i+1)]**2, axis=0 )
                        out_arr.append( medval )
                    else :
                        medval = np.average( arr[period*i:], weights=1/err_arr[period*i:]**2, axis=0 )
                        out_arr.append( medval )
                return( np.asarray(out_arr) )
            
        def __uncertainty_improv( err_arr, period, axsize ):            
            lim = axsize//period
            if period <= 1:
                return err_arr
            elif lim <= 0:
                val = np.sqrt( (err_arr**2).sum(axis=0) ) / axsize
                return np.asarray([val])
            else :
                out_err_arr = []
                for i in range(lim) :
                    if i+1 < lim :
                        val = np.sqrt( (err_arr[period*i:period*(i+1)]**2).sum(axis=0) ) / period
                        out_err_arr.append( val )
                    else :
                        val = np.sqrt( (err_arr[period*i:]**2).sum(axis=0) ) / period
                        out_err_arr.append( val )
                return np.asarray(out_err_arr)
        
        self.u, self.v = __merge_partial( self.u, np.ones(self.u.shape), merge_span, self.Nobs ), __merge_partial( self.v, np.ones(self.v.shape), merge_span, self.Nobs )
        self.Lambda = __merge_partial( self.Lambda, np.ones(self.Lambda.shape), merge_span, self.Nobs )
        self.freqs, self.freqsCPhi = __merge_partial( self.freqs, np.ones(self.freqs.shape), merge_span, self.Nobs ), __merge_partial( self.freqsCPhi, np.ones(self.freqsCPhi.shape), merge_span, self.Nobs )
        self.V2, self.CPhi = __merge_partial( self.V2, self.e_V2, merge_span, self.Nobs ), __merge_partial( self.CPhi, self.e_CPhi, merge_span, self.Nobs )
        self.e_V2, self.e_CPhi = __uncertainty_improv( self.e_V2, merge_span, self.Nobs ), __uncertainty_improv( self.e_CPhi, merge_span, self.Nobs )
        
        self.merged = True
        if self.Nobs//merge_span > 0 :
            self.Nobs = self.Nobs//merge_span
        else :
            self.Nobs = 1


    def plot_spectrum(self):
        fig, ax = plt.subplots()
        ax.set_prop_cycle( color=self.clbr_Nobs )
        if self.array == "CHARA":
            ax.plot(self.Lambda.T, np.median(self.Flux,1).T, marker='o')
            ax.set_ylabel('Median flux (arbitrary units)')
        else :
            ax.plot(self.Lambda.T, self.Flux.T)
            ax.set_ylabel('Normalized flux')
        ax.set_xlabel(r'Wavelength ($\mu$m)')
        return ax
    
    def plot_uv(self, axis_unit='MLamb', legend=False):
    ### A function that plots the full uv coverage with labels to identify the baselines
        fig, ax = plt.subplots()
        if axis_unit == '1/rad':
            u, v = self.u, self.v
            ax.set_xlabel(r'u (x10$^6$ rad$^{-1}$)'), ax.set_ylabel(r'v (x10$^6$ rad$^{-1}$)')
        elif axis_unit == 'meters':
            u, v = self.u*self.Lambda[:,None,:], self.v*self.Lambda[:,None,:]
            ax.set_xlabel(r'u (meters)'), ax.set_ylabel(r'v (meters)')
        else :
            u, v = self.u*self.wl0, self.v*self.wl0
            ax.set_xlabel(r'u (M$\lambda$)'), ax.set_ylabel(r'v (M$\lambda$)')
        lim = 1.1*max( abs(min(u.min(), v.min())), max(u.max(),v.max()) )
        rc = lim/5
        XC = np.cos( np.linspace(0,2*np.pi,100) )
        YC = np.sin( np.linspace(0,2*np.pi,100) )
        
        for i in range(self.Nbl):
            ax.scatter( u[:,i,:], v[:,i,:], color=self.clbr_Nbl[i], edgecolors='k' ), ax.scatter( -u[:,i,:], -v[:,i,:], color=self.clbr_Nbl[i], edgecolors='k', label=self.BLs[i] )
        ax.hlines(0, -lim, lim, ls='dashed', color='k', zorder=0), ax.vlines(0, -lim, lim, ls='dashed', color='k', zorder=0)
        ax.plot(rc*np.linspace(1,7,7)[None,:]*XC[:,None], rc*np.linspace(1,7,7)[None,:]*YC[:,None], color='grey', ls='dotted', zorder=0)
        ax.set_xlim((lim, -lim)), ax.set_ylim(-lim,lim)
        if legend :
            ax.legend(ncols=2, title= 'Baselines:')
        return ax
        
        
    def plot_Vcurve(self, axis_unit="1/rad", logscale=True, legend=False):
        ### A function that plots V2 as a function of frequency
        fig, ax = plt.subplots()
        if axis_unit == "MLamb":
            ax.set_xlabel(r'Spatial frequency (M$\lambda$)')
            freqs = self.freqs*self.wl0
        elif axis_unit == "meters":
            ax.set_xlabel(r'Spatial frequency (meters)')
            freqs = self.freqs*self.Lambda[:,None,:]
        else :
            ax.set_xlabel(r'Spatial frequency (x10$^6$ rad$^{-1}$)')
            freqs = self.freqs

        ax.hlines(1, 0, 1.1*freqs.max(), color='k', lw=1.1)
        for j in range(self.Nbl):
            for i in range(self.Nobs):
                if i == 0 :
                    ax.errorbar(freqs[i,j], self.V2[i,j], yerr=self.e_V2[i,j], markerfacecolor=self.clbr_Nbl[j], markeredgecolor='k', ecolor='k', ls='None', capsize=3, marker='o', markersize=5, label=self.BLs[j])
                else:
                    ax.errorbar(freqs[i,j], self.V2[i,j], yerr=self.e_V2[i,j], markerfacecolor=self.clbr_Nbl[j], markeredgecolor='k', ecolor='k', ls='None', capsize=3, marker='o', markersize=5)
            
        ax.grid(), ax.set_xlim(0,1.1*freqs.max())
        if logscale :
            ax.set_yscale('log'), ax.set_ylim(1e-4,1.5)
        else :
            ax.set_ylim(0,1.05)
        ax.set_ylabel("Visibility squared")
        if legend :
            ax.legend(ncols=2, title='Baselines')
        return ax
    
    def plot_CPhi(self, axis_unit="rad", legend=False):
        ### A function that plots V2 as a function of frequency
        fig, ax = plt.subplots()
        if axis_unit == "MLamb":
            ax.set_xlabel(r'Spatial frequency (M$\lambda$)')
            freqs = self.freqsCPhi*self.wl0
        elif axis_unit == "meters":
            ax.set_xlabel(r'Spatial frequency (meters)')
            freqs = self.freqsCPhi*self.Lambda[:,None,:]
        else :
            ax.set_xlabel(r'Spatial frequency (x10$^6$ rad$^{-1}$)')
            freqs = self.freqsCPhi
            
        ax.hlines(0,0,1.1*freqs.max(), ls='dashed', color='k')
        for j in range(self.Ntr):
            for i in range(self.Nobs):
                if i == 0 :
                    ax.errorbar(freqs[i,j], self.CPhi[i,j], yerr=self.e_CPhi[i,j], markerfacecolor=self.clbr_Ntr[j], markeredgecolor='k', ecolor='k', ls='None', capsize=3, marker='o', markersize=5, label=self.TRs[j])
                else :
                    ax.errorbar(freqs[i,j], self.CPhi[i,j], yerr=self.e_CPhi[i,j], markerfacecolor=self.clbr_Ntr[j], markeredgecolor='k', ecolor='k', ls='None', capsize=3, marker='o', markersize=5 )
        ax.grid(), ax.set_xlim(0,1.1*freqs.max()), ax.set_ylim(-180,180)
        ax.set_ylabel("Closure phase (degrees)")
        
        if legend :
            ax.legend(ncols=4, title='Triplets', loc='lower left')
        return ax
    
    
    # def plot_data(self):
        ### A summary plot showing a V curve (V2+Cphi), the uv-coverage and the flux  
        
        
        