#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTINUUM AND LINE ANALYSIS OF YSOS -- CLAY
datahandling.py -- Allows to recover dataset(s) and format them for later use
Author : H. Nowacki  (hugo.nowacki@oca.eu)
Version : 0.2.0 (07/2026)
Licence : Creative Commons (CC)
No reference for the code yet -- please contact for academic use 
"""
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from itertools import combinations
import scipy.constants as csts
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.coordinates import ICRS, LSR
import astropy.units as units

def rms(arr, axis=-1) :
    return np.sqrt( np.nanmean(arr**2, axis=axis) )
    
def rad2mas(rad):
    mas = rad*180/np.pi*3600*1000
    return mas

def weighted_mean( x, e_x, axis=-1 ):
    num = np.nansum(x/e_x**2, axis=axis)
    den = np.nansum(1/e_x**2, axis=axis)
    val = num/den
    err = 1/np.sqrt( np.nansum(1/e_x**2, axis=axis) )
    return val, err

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
        ### All the data need to be a matrix of shape (Nwvl, Nbl, Nobs)
        self.merged = False
        self.Nobs, self.Nwvl, self.Nbl, self.Ntr = len(path), [], [], []
        self.Flux, self.e_Flux = [], []
        self.wvl = []
        self.V2, self.e_V2 = [], []
        self.Phi, self.e_Phi = [], []
        self.CPhi, self.e_CPhi = [], []
        self.dphi, self.e_dphi = [], []
        self.u, self.v = [], []
        self.freqsCPhi = []
        self.array = []
        self.inst = []
        self.tellurics = []
        self.dates = []
        self.target_coords = { "RA":None, "DEC":None, "pmRA":None, "pmDEC":None }
        for k, p in enumerate(path) :
            ########### INSIDE THE FITS FILE
            hdul = fits.open(p)
            self.array.append( hdul['OI_ARRAY'].header["ARRNAME"] )
            self.inst.append( hdul['Primary'].header["INSTRUME"] )
            if self.array[-1] == "VLTI" :
                try : ### If TELLURICS have been computed with Pmoired for this data.
                    self.Nwvl.append( hdul["TELLURICS"].header["NAXIS2"] )
                    self.wvl.append( hdul["TELLURICS"].data.field("CORR_WAVE") )
                    self.Flux.append( hdul["TELLURICS"].data.field("CORR_SPEC") )
                    self.tellurics.append(True)
                except KeyError :
                    self.tellurics.append(False)
                    try :### Otherwise, try combined polar mode
                        self.Nwvl.append( hdul["OI_WAVELENGTH", 10].header["NAXIS2"] )
                        self.wvl.append( hdul["OI_WAVELENGTH", 10].data.field("EFF_WAVE") )
                        self.Flux.append( hdul["OI_FLUX", 10].data.field("FLUX") )
                        self.e_Flux.append( hdul["OI_FLUX", 10].data.field("FLUXERR") )
                    except KeyError :### otherwise, try split polar mode
                        self.Nwvl.append( hdul["OI_WAVELENGTH", 11].header["NAXIS2"] )
                        self.wvl.append( hdul["OI_WAVELENGTH", 11].data.field("EFF_WAVE") )
                        self.Flux.append( hdul["OI_FLUX", 11].data.field("FLUX") )
                        self.e_Flux.append( hdul["OI_FLUX", 11].data.field("FLUXERR") )
                try :
                    self.V2.append( hdul["OI_VIS2", 10].data.field("VIS2DATA") )
                    self.e_V2.append( hdul["OI_VIS2", 10].data.field("VIS2ERR") )
                    self.Phi.append( hdul["OI_VIS", 10].data.field("VISPHI") )
                    self.e_Phi.append( hdul["OI_VIS", 10].data.field("VISPHIERR") )
                    self.CPhi.append( hdul["OI_T3", 10].data.field("T3PHI") )
                    self.e_CPhi.append( hdul["OI_T3", 10].data.field("T3PHIERR") )
                    self.u.append( hdul["OI_VIS2", 10].data.field("UCOORD") )
                    self.v.append( hdul["OI_VIS2", 10].data.field("VCOORD") )
                        
                except KeyError:
                    self.V2.append( hdul["OI_VIS2", 11].data.field("VIS2DATA") )
                    self.e_V2.append( hdul["OI_VIS2", 11].data.field("VIS2ERR") )
                    self.Phi.append( hdul["OI_VIS", 11].data.field("VISPHI") )
                    self.e_Phi.append( hdul["OI_VIS", 11].data.field("VISPHIERR") )
                    self.CPhi.append( hdul["OI_T3", 11].data.field("T3PHI") )
                    self.e_CPhi.append( hdul["OI_T3", 11].data.field("T3PHIERR") )
                    self.u.append( hdul["OI_VIS2", 11].data.field("UCOORD") )
                    self.v.append( hdul["OI_VIS2", 11].data.field("VCOORD") )
                
     
            else :
                self.Nwvl.append( hdul["OI_WAVELENGTH"].header["NAXIS2"] )
                self.wvl.append( hdul["OI_WAVELENGTH"].data.field("EFF_WAVE") )
                self.Flux.append( hdul["OI_FLUX"].data.field("FLUXDATA") )
                self.e_Flux.append( hdul["OI_FLUX"].data.field("FLUXERR") )
                self.V2.append( hdul["OI_VIS2"].data.field("VIS2DATA") )
                self.e_V2.append( hdul["OI_VIS2"].data.field("VIS2ERR") )
                self.Phi.append( hdul["OI_VIS"].data.field("VISPHI") )
                self.e_Phi.append( hdul["OI_VIS"].data.field("VISPHIERR") )
                self.CPhi.append( hdul["OI_T3"].data.field("T3PHI") )
                self.e_CPhi.append( hdul["OI_T3"].data.field("T3PHIERR") )
                self.u.append( hdul["OI_VIS2"].data.field("UCOORD") )
                self.v.append( hdul["OI_VIS2"].data.field("VCOORD") )
                
            self.Nbl.append( hdul["OI_VIS2"].header["NAXIS2"] )
            self.Ntr.append( hdul["OI_T3"].header["NAXIS2"] )
            self.dates.append( hdul['OI_FLUX'].header['DATE-OBS'] )
            

            
            if k == 0 :
                
                self.target_coords["RA"], self.target_coords["DEC"] = hdul['PRIMARY'].header['RA'], hdul['PRIMARY'].header['DEC']
                self.target_coords["pmRA"], self.target_coords["pmDEC"] = hdul['OI_TARGET'].data.field('PMRA')[0], hdul['OI_TARGET'].data.field('PMDEC')[0]
                self.dicname = {i:n for i,n in zip(hdul['OI_ARRAY'].data.field('STA_INDEX'),
                                             hdul['OI_ARRAY'].data.field('STA_NAME'))}
                self.BL_idx = [ (i,j) for i,j in hdul['OI_VIS'].data.field('STA_INDEX') ]
                self.TR_idx = [ (i,j,k) for i,j,k in hdul['OI_T3'].data.field('STA_INDEX')]
                self.BLs = [self.dicname[i]+'-'+self.dicname[j] for i,j in hdul['OI_VIS'].data.field('STA_INDEX')]
                self.TRs = [self.dicname[i]+'-'+self.dicname[j]+'-'+self.dicname[k] for i,j,k in hdul['OI_T3'].data.field('STA_INDEX')]
            
            
            hdul.close()
            ########### OUTSIDE THE FITS FILE
        
        # if all_equal(self.array) :
        #     self.array = [self.array[0]]
        
        if all_equal(self.Nwvl) and all_equal(self.Nbl) and all_equal(self.Ntr):
            self.Nwvl, self.Nbl, self.Ntr = self.Nwvl[0], self.Nbl[0], self.Ntr[0]
            self.u, self.v = np.asarray(self.u)[:,:,None]/np.asarray(self.wvl)[:,None,:], np.asarray(self.v)[:,:,None]/np.asarray(self.wvl)[:,None,:] ### Unit here is 1/rad
            self.wvl = np.asarray(self.wvl)#*1e6 ### wavelength becomes microns 
            self.Flux  = np.asarray(self.Flux)
            self.V2, self.e_V2 = np.asarray(self.V2), np.asarray(self.e_V2)
            self.Phi, self.e_Phi = np.asarray(self.Phi), np.asarray(self.e_Phi)
            self.CPhi, self.e_CPhi = np.asarray(self.CPhi), np.asarray(self.e_CPhi)
        else :
            self.Nwvl, self.Nbl, self.Ntr = np.asarray(self.Nwvl), np.asarray(self.Nbl), np.asarray(self.Ntr)
            ##### Pad the data in order to transform in arrays (TBD)
        
        self.wl0 = self.wvl.mean()  # Unit = microns
        self.freqs = np.hypot( self.u, self.v ) # Unit = cycles/rad
        
        
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
        
        
# =============================================================================
#         Setting floor/ceiling values on uncertainties
# =============================================================================
        self.e_V2[self.e_V2 < 1e-3] = 1e-3
        self.e_V2[self.e_V2 > 1] = 1
        self.e_Phi[self.e_Phi <= 0.1] = 0.1
        self.e_Phi[self.e_Phi > 180] = 180
        self.e_Phi[np.isnan(self.e_Phi)] = 10
        self.e_CPhi[self.e_CPhi <= 0.1] = 0.1
        self.e_CPhi[self.e_CPhi > 180] = 180
        self.e_CPhi[np.isnan(self.e_CPhi)] = 10
        
        
        if self.Nbl == 6:
            self.Ntel = 4
        elif self.Nbl == 10:
            self.Ntel = 5
        elif self.Nbl == 15:
            self.Ntel = 6
            
        self.clbr_Nbl = plt.cm.nipy_spectral_r(np.linspace(0.1, 0.88, self.Nbl))
        self.clbr_Nobs = plt.cm.nipy_spectral_r(np.linspace(0.1, 0.9, self.Nobs))
        self.clbr_Ntel = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Ntel))
        self.clbr_Ntr = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Ntr))
    
    # def spectral_crop(self, wl_range, inout=in):
        ### A function that extracts a spectral region inside or outside (inout) a given wavelength range wl_range
        ### Need to be able to pass several ranges (i.e. to define the continuum)




    def spectral_binning(self, bin_factor=5):
        """
        Weighted spectral binning with interpolation of NaNs.
        Compatible multi-dimensions (any shape that ends up with [..., channels]).
    
        Steps :
        1. interpolating NaNs on last axis
        2. centering to be a multiple of bin_factor
        3. binning weighted by 1/sigma^2
        """    
    
        def __interpolate_nan_1d(arr):
            """
            Interpolates a 1D vector containing NaNs.
            - Standard linear interpolation for finite values
            - If all NaNs : gives back arr
            - If NaN on the edges : linear extrapolation
            """
            x = np.arange(arr.size)
            mask = np.isfinite(arr)
    
            if mask.sum() == 0:
                return arr  # Nothing to do, all NaNs
    
            f = interp1d( x[mask], arr[mask], kind='linear', fill_value='extrapolate', bounds_error=False )
            return f(x)
    
        def __interpolate_nan_last_axis(arr):
            """
            Applies interpolate_nan_1d over the last axis,
            while preserving the shape of the general table N-D.
            """
            arr = np.asarray(arr)
            reshaped = arr.reshape(-1, arr.shape[-1])  # collapse all dims except last
            out = np.empty_like(reshaped)
    
            for i in range(reshaped.shape[0]):
                out[i] = __interpolate_nan_1d(reshaped[i])
    
            return out.reshape(arr.shape)
    
        def __centered_slice_nd(arr, factor):
            """
            Cuts the last axis of 'arr' in a centered way
            so the length is an intger scaling of 'factor'.
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
            Transforms arr[..., N] into arr[..., nbins, factor].
            """
            new_shape = arr.shape[:-1] + (arr.shape[-1] // factor, factor)
            return arr.reshape(new_shape)
    
            
        def __bin_weighted(values, errors, factor):
            # Centering
            v = __centered_slice_nd(values, factor)
            e = __centered_slice_nd(errors, factor)
    
            # groups for binning
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
        self.wvl = __interpolate_nan_last_axis(self.wvl)
        
        self.u, self.v = __interpolate_nan_last_axis(self.u), __interpolate_nan_last_axis(self.v)
        
        self.freqs   = __interpolate_nan_last_axis(self.freqs)
        self.freqsCPhi   = __interpolate_nan_last_axis(self.freqsCPhi)
        
        self.Phi   = __interpolate_nan_last_axis(self.Phi)
        self.e_Phi  = __interpolate_nan_last_axis(self.e_Phi)
        
        self.V2   = __interpolate_nan_last_axis(self.V2)
        self.e_V2  = __interpolate_nan_last_axis(self.e_V2)
    
        self.Flux  = __interpolate_nan_last_axis(self.Flux)
        self.e_Flux = __interpolate_nan_last_axis(self.e_Flux) ### Might be useless...
    
        self.CPhi   = __interpolate_nan_last_axis(self.CPhi)
        self.e_CPhi  = __interpolate_nan_last_axis(self.e_CPhi)
    
        # --------------- STEP 2 : WEIGHTED BINNING --------------- #
        self.wvl, duh = __bin_weighted(self.wvl, np.ones(self.wvl.shape),   bin_factor)
        
        self.u, duh = __bin_weighted(self.u, np.ones(self.u.shape),   bin_factor)
        self.v, duh = __bin_weighted(self.v, np.ones(self.v.shape),   bin_factor)
        
        self.freqs, duh = __bin_weighted(self.freqs, np.ones(self.freqs.shape),   bin_factor)
        self.freqsCPhi, duh = __bin_weighted(self.freqsCPhi, np.ones(self.freqsCPhi.shape),   bin_factor)
    
        self.Flux, duh = __bin_weighted(self.Flux, np.ones(self.Flux.shape), bin_factor)
        self.CPhi, self.e_CPhi = __bin_weighted(self.CPhi, self.e_CPhi, bin_factor)
        self.V2, self.e_V2   = __bin_weighted(self.V2, self.e_V2,   bin_factor)
        self.Phi, self.e_Phi   = __bin_weighted(self.Phi, self.e_Phi,   bin_factor)
        
        
        # self.V2   = __interpolate_nan_last_axis(self.V2) ### Was here but seems to be useless keep in case
        


    def merge_obs(self, merge_span=2):
        """ 
        A function that merges in the 'Nobs' dimension over 'merge_span' files to reduce 
        the uncertainty on interferometric quantities 
        Caveat : uncertainty decreased on frequencies if too long 
        """
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
        self.wvl = __merge_partial( self.wvl, np.ones(self.wvl.shape), merge_span, self.Nobs )
        self.Flux = __merge_partial( self.Flux, np.ones(self.Flux.shape), merge_span, self.Nobs )
        self.freqs, self.freqsCPhi = __merge_partial( self.freqs, np.ones(self.freqs.shape), merge_span, self.Nobs ), __merge_partial( self.freqsCPhi, np.ones(self.freqsCPhi.shape), merge_span, self.Nobs )
        self.V2, self.CPhi = __merge_partial( self.V2, self.e_V2, merge_span, self.Nobs ), __merge_partial( self.CPhi, self.e_CPhi, merge_span, self.Nobs )
        self.e_V2, self.e_CPhi = __uncertainty_improv( self.e_V2, merge_span, self.Nobs ), __uncertainty_improv( self.e_CPhi, merge_span, self.Nobs )
        self.Phi = __merge_partial( self.Phi, self.e_Phi, merge_span, self.Nobs )
        self.e_Phi = __uncertainty_improv( self.e_Phi, merge_span, self.Nobs )
        
        
        self.merged = True
        if self.Nobs//merge_span > 0 :
            self.Nobs = self.Nobs//merge_span
        else :
            self.Nobs = 1


    def plot_spectrum(self):
        fig, ax = plt.subplots()
        ax.set_prop_cycle( color=self.clbr_Nobs )
        if self.array == "CHARA":
            ax.plot((self.wvl*1e6).T, np.nanmedian(self.Flux,1).T, marker='o')
            ax.set_ylabel('Median flux (arbitrary units)')
        else :
            if self.wvl.ndim == self.Flux.ndim :
                ### When TELLURICS computed
                ax.plot((self.wvl*1e6).T, self.Flux.T)
                ax.set_ylabel('Normalized flux')
            else :
                ### When TELLURICS not computed
                ax.plot((self.wvl*1e6).T, np.nanmedian(self.Flux,axis=1).T)
                ax.set_ylabel('Median flux')
        ax.set_xlabel(r'Wavelength ($\mu$m)')
        return ax
    
    def plot_uv(self, axis_unit='MLamb', legend=False):
    ### A function that plots the full uv coverage with labels to identify the baselines
        fig, ax = plt.subplots()
        if axis_unit == '1/rad':
            u, v = self.u, self.v
            ax.set_xlabel(r'u (x10$^6$ rad$^{-1}$)'), ax.set_ylabel(r'v (x10$^6$ rad$^{-1}$)')
        elif axis_unit == 'meters':
            u, v = self.u*self.wvl[:,None,:], self.v*self.wvl[:,None,:]
            ax.set_xlabel(r'u (meters)'), ax.set_ylabel(r'v (meters)')
        else :
            u, v = self.u*self.wl0, self.v*self.wl0
            ax.set_xlabel(r'u (M$\lambda$)'), ax.set_ylabel(r'v (M$\lambda$)')
        lim = 1.1*max( abs(min(u.min(), v.min())), max(u.max(),v.max()) )
        rc = lim/5
        XC = np.cos( np.linspace(0,2*np.pi,100) )
        YC = np.sin( np.linspace(0,2*np.pi,100) )
        
        for i in range(self.Nbl):
            ax.scatter( u[:,i,:], v[:,i,:], color=self.clbr_Nbl[i], edgecolors='k', lw=0.1 ), ax.scatter( -u[:,i,:], -v[:,i,:], color=self.clbr_Nbl[i], edgecolors='k', lw=0.1, label=self.BLs[i] )
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
            freqs = self.freqs*1e-6
        elif axis_unit == "meters":
            ax.set_xlabel(r'Spatial frequency (meters)')
            freqs = self.freqs*self.wvl[:,None,:]*1e-6
        elif axis_unit == "1/mas":
            ax.set_xlabel(r'Spatial frequency (cycles / mas)')
            freqs = self.freqs /180*np.pi/3600/1000
        else :
            ax.set_xlabel(r'Spatial frequency (cycles / rad)')
            freqs = self.freqs

        ax.hlines(1, 0, 1.1*freqs.max(), color='k', lw=1.1)
        for j in range(self.Nbl):
            for i in range(self.Nobs):
                if i == 0 :
                    ax.errorbar(freqs[i,j], self.V2[i,j], yerr=self.e_V2[i,j], markerfacecolor=self.clbr_Nbl[j], markeredgecolor='None', ecolor='grey', ls='None', capsize=3, marker='o', markersize=5, label=self.BLs[j])
                else:
                    ax.errorbar(freqs[i,j], self.V2[i,j], yerr=self.e_V2[i,j], markerfacecolor=self.clbr_Nbl[j], markeredgecolor='None', ecolor='grey', ls='None', capsize=3, marker='o', markersize=5)
            
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
            freqs = self.freqsCPhi*1e-6
        elif axis_unit == "meters":
            ax.set_xlabel(r'Spatial frequency (meters)')
            freqs = self.freqsCPhi*self.wvl[:,None,:]*1e-6
        else :
            ax.set_xlabel(r'Spatial frequency (x10$^6$ rad$^{-1}$)')
            freqs = self.freqsCPhi
            
        ax.hlines(0,0,1.1*freqs.max(), ls='dashed', color='k')
        for j in range(self.Ntr):
            for i in range(self.Nobs):
                if i == 0 :
                    ax.errorbar(freqs[i,j], self.CPhi[i,j], yerr=self.e_CPhi[i,j], markerfacecolor=self.clbr_Ntr[j], markeredgecolor='None', elinewidth=0.1, ecolor='k', ls='None', capsize=3, marker='o', markersize=5, label=self.TRs[j])
                else :
                    ax.errorbar(freqs[i,j], self.CPhi[i,j], yerr=self.e_CPhi[i,j], markerfacecolor=self.clbr_Ntr[j], markeredgecolor='None', elinewidth=0.1, ecolor='k', ls='None', capsize=3, marker='o', markersize=5 )
        ax.grid(), ax.set_xlim(0,1.1*freqs.max()), ax.set_ylim(-180,180)
        ax.set_ylabel("Closure phase (degrees)")
        
        if legend :
            ax.legend(ncols=3, title='Triplets', loc='lower left')
        return ax
    
    
    # def plot_data(self):
        ### A summary plot showing a V curve (V2+Cphi), the uv-coverage and the flux  


        
class emission_line:
    """
    A class that allows to work on an emission line specifically
    """
    def __init__(self, dataset, line='BrG', vContMax=1000, vContMin=-1000, tellcorr=False):
        ### Extracts the region surrounding an emission line in a dataset
        # =============================================================================
        #         Define main properties
        # =============================================================================
        # self.original_data = dataset
        self.array, self.BLs = dataset.array, dataset.BLs
        self.Nobs, self.Nbl, self.Ntel = dataset.Nobs, dataset.Nbl, dataset.Ntel
        self.target_coords, self.dates = dataset.target_coords, dataset.dates
        # self.Ntr = dataset.Ntr  ### Useless for the moment
        self.line_defined, self.pl_comp = False, False
        if all(dataset.tellurics):
            self.tellurics = True
        else :
            self.tellurics = False
            
        self.clbr_Nbl = plt.cm.nipy_spectral_r(np.linspace(0.1, 0.88, self.Nbl))
        self.clbr_Nobs = plt.cm.nipy_spectral_r(np.linspace(0.1, 0.9, self.Nobs))
        self.clbr_Ntel = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Ntel))
        # self.clbr_Ntr = plt.cm.nipy_spectral_r(np.linspace(0., 1., self.Ntr))
        
        # =============================================================================
        #         First check if pmoired "TELLURICS" can/should be computed
        # =============================================================================
        if tellcorr and not all(dataset.tellurics) :
            if all_equal(dataset.inst) and dataset.inst[0] == 'GRAVITY' :
                import pmoired as pm
                print('Correcting tellurics...')
                for i, p in enumerate(dataset.path):
                    print('File nb ',i+1,'/',len(dataset.path))
                    pm.tellcorr.gravity(p)
                print('Done')
            else :
                raise ValueError('Cannot correct tellurics with pmoired if data is not from GRAVITY')
        elif tellcorr and all(dataset.tellurics) :
            print('Tellurics already corrected in the files')
        
        elif not tellcorr and not all(dataset.tellurics) :
            print("============= BEWARE ============= \n"
                  "all spectra should be normalized \n"
                  "before proceeding to differential analysis \n"
                  "WHICH DOES NOT SEEM TO BE THE CASE HERE\n"
                  "==================================")

        # =============================================================================
        #       Then, extract the relevant data
        # =============================================================================
        if line == 'BrG' :
            self.wvl0 = 2.1661178e-6
        elif line == 'HeI' :
            self.wvl0 = 2.0587e-6
        else :
            raise ValueError("Unknown line, try 'BrG', 'HeI'")
            
        self.vContMax = vContMax
        self.vContMin = vContMin
        self.wvlMin = self.wvl0 * ( 1 + self.vContMin*1000/csts.c )
        self.wvlMax = self.wvl0 * ( 1 + self.vContMax*1000/csts.c )
    
        mean_wvls = dataset.wvl.mean(axis=0)
        idx = (  (mean_wvls > self.wvlMin) & (mean_wvls < self.wvlMax)  )
        self.wvl = dataset.wvl[ :, idx ]
        self.Nwvl = np.nansum(idx)
        self.vel = csts.c/1000 * (self.wvl/self.wvl0 - 1)
        self.u, self.v, self.freqs = dataset.u[ :, :, idx ], dataset.v[ :, :, idx ], dataset.freqs[ :, :, idx ]
        self.B_norm, self.B_ang = np.hypot(self.u, self.v)*self.wvl[:,None,:], np.rad2deg(np.arctan2( self.v, self.u ))
        self.B = np.array( [self.B_norm*np.cos(np.deg2rad(self.B_ang)), self.B_norm*np.sin(np.deg2rad(self.B_ang))] )
        if not self.tellurics :
            self.Flux = np.nanmean(dataset.Flux, axis=1)[ :, idx ]
            self.e_Flux = np.nanmean(dataset.e_Flux, axis=1)[ :, idx ]
        else :
            self.Flux = dataset.Flux[ :, idx ]
            cut = self.Nwvl//6
            self.e_Flux = np.ones(self.Flux.shape) * rms( (self.Flux[:,:cut]-1),axis=(0,1) )
        self.V2 = dataset.V2[ :, :, idx ]
        self.e_V2 = dataset.e_V2[ :, :, idx ]
        self.Phi = dataset.Phi[ :, :, idx ]
        self.e_Phi = dataset.e_Phi[ :, :, idx ]
        

    def normalize_dphi(self, vmin, vmax, polyorder=1):
        ### Add the possibility to plot to check the region of normalization    
        corrected = np.empty_like(self.Phi, dtype=float)
        # coeffs = np.full((self.Nobs, self.Nbl, polyorder+1), np.nan)
        for iobs in range(self.Nobs):
            mask = ( ((self.vel[iobs] <= vmin) | (self.vel[iobs] >= vmax)) & np.isfinite(self.vel[iobs]) )
            if np.count_nonzero(mask) < polyorder+1 :
                corrected[iobs] = np.nan
                continue
    
            for ibl in range(self.Nbl):
                y = self.Phi[iobs, ibl]
                valid = mask & np.isfinite(y)
    
                if np.count_nonzero(valid) < polyorder+1 :
                    corrected[iobs, ibl] = np.nan
                    continue
                p = np.polyfit(self.wvl[iobs][valid], y[valid], deg=polyorder)
                # coeffs[iobs, ibl] = p
    
                poly = np.polyval(p, self.wvl[iobs])
                corrected[iobs, ibl] = y - poly
        self.Phi = corrected


    def correct_rv( self, systematic_rv, dist ) :
        for i in range( self.Nobs ) :
            loc = EarthLocation(lon='-24d37m39.5s',lat='-70d24m15.5s',height=2635.43*units.m)
            v_rad = systematic_rv*units.km/units.s  
            radeg, decdeg = self.target_coords['RA'], self.target_coords['DEC']
            pmra, pmdec = self.target_coords['pmRA'], self.target_coords['pmDEC']
            pmra2 = pmra*np.cos( np.deg2rad( decdeg ) )
            sc = SkyCoord( radeg*units.deg, decdeg*units.deg )
            t0 = Time( self.dates[i], format='isot', scale='utc' )          
            t0 = Time(t0.mjd, format='mjd', scale='utc')
            vcorr = sc.radial_velocity_correction(kind='barycentric', obstime=t0, location=loc)
            rv = v_rad + vcorr.to(units.km/units.s)
            obs = ICRS(ra=radeg*units.deg, dec=decdeg*units.deg,
                                  pm_ra_cosdec=pmra2*units.mas/units.yr, pm_dec= pmdec*units.mas/units.yr,
                                  radial_velocity=rv.value*units.km/units.s, distance = dist*units.pc)
            
            new_rv = obs.transform_to(LSR()).radial_velocity
            self.vel[i,:] += new_rv.value
            

    def define_line(self, vmin=None, vmax=None, Flux_threshold=0.25 ):
        if not self.tellurics :
            raise ValueError('Cannot define a line with a non-normalized flux')
        
        mean_vel = np.nanmean(self.vel, axis=0)
        if not vmin or not vmax :
            mean_Flux = np.nanmean(self.Flux, axis=0)
            max_Flux = np.nanmax( mean_Flux )
            vel_above = mean_vel[mean_Flux >= 1 + Flux_threshold*(max_Flux-1) ]
            self.vlineMin, self.vlineMax = np.nanmin(vel_above), np.nanmax(vel_above)
        else :
            self.vlineMin, self.vlineMax = vmin, vmax
        
        idx_vel = (( mean_vel <= self.vlineMax ) & ( mean_vel >= self.vlineMin ))
        self.wvl_line, self.vel_line = self.wvl[ :, idx_vel ], self.vel[ :, idx_vel ]
        self.Flux_line, self.e_Flux_line = self.Flux[ :, idx_vel ], self.e_Flux[ :, idx_vel ]
        self.V2_line, self.e_V2_line = self.V2[ :, :, idx_vel ], self.e_V2[ :, :, idx_vel ]
        self.Phi_line, self.e_Phi_line = self.Phi[ :, :, idx_vel ], self.e_Phi[ :, :, idx_vel ]
        self.B_line = self.B[ :, :, :, idx_vel ]
        
        self.Nvel = np.nansum(idx_vel)
        Vtilde = np.nanmax([abs(self.vlineMin), abs(self.vlineMax)])
        self.clbr_Nvel = plt.cm.seismic(np.linspace( (1-abs(np.nanmin(self.vel_line))/Vtilde)/2, abs(np.nanmax(self.vel_line))/Vtilde, self.Nvel ) )
        self.line_defined = True


    def compute_pureline( self ):
        if not self.line_defined :
            raise ValueError("Cannot derive pure-line quantities if line is not defined")
            
        
        if np.nanmax(self.Phi_line) >= 10 :
            print("============= BEWARE ============= \n"
                  "Differential phase seems out of the \n"
                  "validity domain for pure-line quantities \n"
                  "(>= 10 degrees), results should be \n"
                  "interpreted carefully, consider \n"
                  "using complex visibilities instead\n"
                  "==================================")
        
        mean_vel = np.nanmean( self.vel, axis=0 )
        idx_vel = (( (mean_vel > self.vContMin) & (mean_vel < self.vlineMin) ) | ((mean_vel < self.vContMax) & (mean_vel > self.vlineMax)) )
        Vcont = np.median( np.sqrt( self.V2[:,:,idx_vel] ), axis=-1 )
        e_Vcont = np.std( np.sqrt( self.V2[:,:,idx_vel] ), axis=-1 )
        specCorrFlux = self.Flux_line[:,None,:] * np.sqrt(self.V2_line) / Vcont[:,:,None]
                
        self.V_pl = Vcont[:,:,None] * (specCorrFlux-1)/(self.Flux_line[:,None,:]-1) 
        self.e_V_pl = self.V_pl * np.sqrt( (e_Vcont[:,:,None]/Vcont[:,:,None])**2 + (self.e_Flux_line[:,None,:]/self.Flux_line[:,None,:])**2 + (self.e_V2_line/self.V2_line/2)**2 )
        
        alpha = specCorrFlux/(specCorrFlux-1)
        self.Phi_pl = np.rad2deg( np.arcsin( alpha*np.sin( np.deg2rad(self.Phi_line)) ))
        
        term1 = e_Vcont[:,:,None]/Vcont[:,:,None]
        term2 = self.e_Flux_line[:,None,:]/self.Flux_line[:,None,:]
        term3 = self.e_V2_line/self.V2_line/2
        e_Phi_pl = np.rad2deg( np.tan(np.deg2rad(self.Phi_pl))*np.sqrt( [term1**2 + term2**2 + term3**2]/(alpha-1)**2 + (np.deg2rad(self.e_Phi_line)/np.tan(np.deg2rad(self.Phi_line)))**2 ))
        self.e_Phi_pl = np.squeeze(np.sqrt(e_Phi_pl**2))
        if self.e_Phi_pl.ndim == 2:
            self.e_Phi_pl = self.e_Phi_pl[None,:,:]
        self.pl_comp = True
        
    
    def compute_astrometry( self, BL_used={'U2-U1':True,'U3-U2':True,'U4-U3':True,'U3-U1':True,'U4-U2':True,'U4-U1':True} ) :
        """
        Computes the astrometric displacements from pure-line differential phases
        """
        if not self.pl_comp :
            raise ValueError("Need to compute pure-line quantities before computing astrometry")
        filterB = [True] * self.Nbl
        for key, b in zip(BL_used.keys(),BL_used.values()):
            for j, k in enumerate(self.BLs) :
                if k == key :
                    filterB[j] = b
                elif k == len(self.BLs)-1 :
                    raise ValueError("Please write the BLs as from the pipeline\n"
                                     f"i.e., {self.BLs}")
        
        x_line, y_line = [], []
        e_x_line, e_y_line = [], []
        for i in range( self.Nvel ) :
            #### In case computation shouldn't occur on median values
            # PHI = np.deg2rad( self.Phi_pl[:,filterB,i] )
            # dPHI = np.deg2rad( self.e_Phi_pl[:,filterB,i] )
            # Bi = np.linalg.pinv( self.B_line[ :, :, filterB, i] )
            # wvl = self.wvl_line[:,i]
            # P = -wvl[None,:]/2/np.pi * np.einsum('ikj,jk->ij', Bi, PHI)
            # dP = P * np.einsum('ikj,jk->ij', Bi, dPHI) / np.einsum('ikj,jk->ij', Bi, PHI)
            # x_line.append(P[0]), y_line.append(P[1])
            # e_x_line.append(np.sqrt(dP[0]**2)), e_y_line.append(np.sqrt(dP[1]**2))
            
            PHI = np.deg2rad( self.Phi_pl[:,filterB,i] )
            dPHI = np.deg2rad( self.e_Phi_pl[:,filterB,i] )
            PHI, dPHI = weighted_mean( PHI, dPHI, axis=0 )
            Bi = np.linalg.pinv( np.nanmean(self.B_line[ :, :, filterB, i], 1) ).T
            wvl = self.wvl_line[:,i].mean()
            P = -wvl/2/np.pi * np.dot( Bi, PHI)
            dP = P * np.dot( Bi, dPHI) / np.dot( Bi, PHI)
            x_line.append(P[0]), y_line.append(P[1])
            e_x_line.append(np.sqrt(dP[0]**2)), e_y_line.append(np.sqrt(dP[1]**2))
                        
            self.x_line, self.e_x_line = np.array(x_line).T, np.array(e_x_line).T
            self.y_line, self.e_y_line = np.array(y_line).T, np.array(e_y_line).T
            
        
    def prepare_pureline(self, vmin=-500, vmax=500, poly=1, rv=0, dist=150, Flx_thr=0.25):
        """
        Calls all the functions needed to go from raw line data to pure-line data
        """
        self.normalize_dphi( vmin=vmin, vmax=vmax, polyorder=poly)
        self.correct_rv( systematic_rv=rv, dist=dist )
        self.define_line( Flux_threshold=Flx_thr )
        self.compute_pureline()


    def plot_astrometry(self, vmin=-200, vmax=200, phimin=None, phimax=None, axis_lim=((150,-150),(-150,150)), axis_unit='muas'):
        # Must only apply to pure line data
        if not self.pl_comp :
            raise ValueError("Need to compute pure-line quantities and \n"
                             "astrometry before plotting astrometry")
        
        x, y = rad2mas(self.x_line), rad2mas(self.y_line)
        e_x, e_y = rad2mas(self.e_x_line), rad2mas(self.e_y_line)
        
               
        gs_kw = dict(width_ratios=[ 1, 1, 1, 2], height_ratios=[1, 1], hspace=0.02)
        fig, axd = plt.subplot_mosaic([[0, 1, 2, 'ax1'], [3, 4, 5, 'ax1']], gridspec_kw=gs_kw, figsize=(19.6, 6.5), \
                                      layout="constrained")
        ax1 = axd['ax1']

        ax1.hlines( 0, 5000, -5000, ls='dotted', color='k' )
        ax1.vlines( 0, -5000, 5000, ls='dotted', color='k' )
        if axis_unit == "muas" :
            x, y = x*1000, y*1000
            e_x, e_y = e_x*1000, e_y*1000
            ax1.set_xlabel(r'$\longleftarrow$ East (µas)'), ax1.set_ylabel(r'North (µas) $\longrightarrow$')
        else :
            ax1.set_xlabel(r'$\longleftarrow$ East (mas)'), ax1.set_ylabel(r'North (mas) $\longrightarrow$')
        
        mean_x, e_mean_x = x, e_x
        mean_y, e_mean_y = y, e_y
        
        ##### For future clean integration : continuum photocenter + star radius
        # circle1 = plt.Circle((-cont_baricenter[0]*pixscale/sampling*1000, -cont_baricenter[1]*pixscale/sampling*1000), 1*cont_err*pixscale/sampling*1000, fill=True, color='cyan', alpha=0.3, ec='k')
        # circle2 = plt.Circle((-cont_baricenter[0]*pixscale/sampling*1000, -cont_baricenter[1]*pixscale/sampling*1000), 2*cont_err*pixscale/sampling*1000, fill=True, color='cyan', alpha=0.2, ec='k')
        # circle3 = plt.Circle((-cont_baricenter[0]*pixscale/sampling*1000, -cont_baricenter[1]*pixscale/sampling*1000), 3*cont_err*pixscale/sampling*1000, fill=True, color='cyan', alpha=0.1, ec='k')
        # starRad = plt.Circle((0,0), auToMas(R,d)*1000, fill=False, color='k', ls='dashed' )
        # ax1.add_patch(starRad), ax1.add_patch(circle1), ax1.add_patch(circle2), ax1.add_patch(circle3)
        # ax1.scatter(-cont_baricenter[0]*pixscale/sampling*1000, -cont_baricenter[1]*pixscale/sampling*1000, marker='*', color='cyan', ec='k', s=1200, label='Star position', zorder=12)
        # ax1.plot( [-1000, 1000], [-np.tan(np.pi/2 - pa*np.pi/180)*1000, np.tan(np.pi/2 - pa*np.pi/180)*1000], ls='dashed', color='gold')
        # ax1.fill_between([-1000, 1000], [-np.tan(np.pi/2 - (pa-30)*np.pi/180)*1000, np.tan(np.pi/2 - (pa-30)*np.pi/180)*1000], [-np.tan(np.pi/2 - (pa+30)*np.pi/180)*1000, np.tan(np.pi/2 - (pa+30)*np.pi/180)*1000], color='gold', alpha=0.2 )
        for i in range( self.Nvel ):
            ax1.errorbar( mean_x[i], mean_y[i], xerr=e_mean_x[i], yerr=e_mean_y[i], linewidth=2.5, color='k', ecolor='k', mfc=self.clbr_Nvel[i], ms=10, linestyle='None', marker='o', capsize=6, capthick=2.5, label='Median Shifts' )
        ax1.set_xlim( axis_lim[0] ), ax1.set_ylim( axis_lim[1] )
        c1 = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(self.vlineMin,-self.vlineMax),vmax=max(self.vlineMax, -self.vlineMin)), \
                cmap=mpl.colormaps['seismic']), location='right', pad=0, ax=ax1)
        c1.set_label(label='Velocity (km/s)',labelpad=0)
        if not phimin or not phimax :
            ylim = 1.1+np.nanmax(np.nanmedian(self.Phi_pl,0))
        
        for i in range(self.Nbl):
            axd[i].plot(self.vel.T, self.Phi[:,i,:].T, color=self.clbr_Nbl[i], lw=0.5 )
            
            axd[i].hlines(0, vmin, vmax, ls='dashed', color='gold', zorder=0)
            axd[i].vlines(0, -180, 180, ls='dotted', color='k')
            for j in range(self.Nvel):
                if self.Nobs == 1 :
                    axd[i].errorbar(self.vel_line[:,j], self.Phi_pl[:,i,j], yerr=self.e_Phi_pl[:,i,j], ms=10, mfc=self.clbr_Nvel[j], color='k', marker='o', linestyle='None', lw=2, capsize=6, capthick=2, zorder=6)
                else :
                    meanv = np.nanmean(self.vel_line, axis=0 )
                    meanphi, e_meanphi = weighted_mean( self.Phi_pl[:,i,j], self.e_Phi_pl[:,i,j], axis=0 )
                    axd[i].errorbar(meanv[j], meanphi, yerr=e_meanphi, ms=10, mfc=self.clbr_Nvel[j], color='k', marker='o', linestyle='None', lw=2, capsize=6, capthick=2, zorder=6)
            axd[i].set_xlim(vmin, vmax)
            if not phimin or not phimax :
                axd[i].set_ylim( (-ylim, ylim) )
            else :
                axd[i].set_ylim( (phimin, phimax) )
            if ((i==0)|(i==3)) :
                axd[i].set_ylabel('Differential phase (degrees)')
            if ((i==3)|(i==4)|(i==5)):
                axd[i].set_xlabel('Velocity (km/s)')

    
    def plot_data(self, vmin, vmax):
        """ 
        Recap plot with uv-coverage, flux, differential phase and differential V2
        """
        XC = np.cos( np.linspace(0,2*np.pi,100) )
        YC = np.sin( np.linspace(0,2*np.pi,100) ) 
        gs_kw = dict(width_ratios=[1.5, 1, 1, 1, 2], height_ratios=[1, 1], hspace=0.02, wspace=.15)
        fig, axd = plt.subplot_mosaic([['ax1', 0, 1, 2, 'ax8'], ['ax1', 3, 4, 5, 'ax9']], gridspec_kw=gs_kw, figsize=(17.5, 9) )
        ax1, ax8, ax9 = axd['ax1'], axd['ax8'], axd['ax9']
        ax1.vlines(0, 0, 1, linestyle='dashed', color='k', alpha=0.7)
        ax1b = ax1.twiny() 
        for i in range(self.Nbl):
                ax1.plot(self.vel.T, self.V2[:,i,:].T, color=self.clbr_Nbl[i], linewidth=0.5, zorder=-1)
                ax1.errorbar(np.nanmean(self.vel,axis=0).T, np.nanmean(self.V2[:,i,:],axis=0).T, yerr=np.nanmean(self.e_V2[:,i,:], axis=0)/np.sqrt(self.Nobs), capsize=4, marker='o', color='k', mfc=self.clbr_Nbl[i], linestyle='None')
                
                axd[i].plot(self.vel.T, self.Phi[:,i,:].T, color=self.clbr_Nbl[i], lw=0.5 )
                axd[i].errorbar(self.vel.mean(axis=0), np.mean(self.Phi[:,i,:], axis=0), yerr=np.nanmean(self.e_Phi[:,i,:], axis=0)/np.sqrt(self.Nobs), capsize=4, marker='o', color='k', mfc=self.clbr_Nbl[i], zorder=6)
                axd[i].set_xlim( ( vmin, vmax ) )
                axd[i].set_ylim((-1.1*np.nanmax(abs(self.Phi)), 1.1*np.nanmax(abs(self.Phi))))
                if self.line_defined :
                    axd[i].fill_betweenx([-180,180], vmin, self.vlineMin, color='grey', alpha=0.2)
                    axd[i].fill_betweenx([-180,180], self.vlineMax, vmax, color='grey', alpha=0.2)
                axd[i].hlines(0, vmin, vmin, linestyle='dashed', color='k', alpha=0.7)
                axd[i].vlines(0, -180, 180, linestyle='dashed', color='k', alpha=0.7)
                axd[i].text( 0.55, 0.9, self.BLs[i], bbox=dict(facecolor=self.clbr_Nbl[i], edgecolor='k', boxstyle='round,pad=1', alpha=0.5),horizontalalignment='left', verticalalignment='top', transform = axd[i].transAxes)
                axd[i].tick_params(axis="y",direction="in", pad=-22)
                if ((i==0)|(i==3)) :
                    axd[i].set_ylabel('Differential phase (degrees)')
                if ((i==3)|(i==4)|(i==5)):
                    axd[i].set_xlabel('Velocity (km/s)')
        
        if self.line_defined :
            ax1.fill_betweenx( [-0.01,1.01], vmin, self.vlineMin, color='grey', alpha=0.2 )
            ax1.fill_betweenx( [-0.01,1.01], self.vlineMax, vmax, color='grey', alpha=0.2 )
        ax1.set_xlabel('Velocity (km/s)')
        ax1b.set_xlabel(r'Wavelength ($\mu$m)')
        ax1.set_ylabel('Visibility squared')
        ax1.set_xlim((vmin, vmax))
        ax1b.set_xlim((self.wvl0*(1+vmin/csts.c*1000)*1e6, self.wvl0*(1+vmax/csts.c*1000)*1e6))
        ax1.set_ylim(( np.nanmax([-0.01, 0.89*abs(np.nanmin(self.V2))]) , np.nanmin( [1.1, 1.1*np.nanmax(self.V2)]) ))

        ax8.xaxis.set_label_position('top')
        ax8.yaxis.set_label_position('right')
        ax8.xaxis.tick_top()
        ax8.yaxis.tick_right()
        ax8.hlines(0, -60, 60, linestyle='dashed', color='k')
        ax8.vlines(0, -60, 60, linestyle='dashed', color='k')
        for i in range(8):
            ax8.plot(10*(1+i)*XC, 10*(1+i)*YC, color='grey', ls='dotted')
        for i in range(self.Nbl):
            ax8.scatter(self.u[:,i]*1e-6, self.v[:,i]*1e-6, marker='o', color=self.clbr_Nbl[i])
            ax8.scatter(-self.u[:,i]*1e-6, -self.v[:,i]*1e-6, marker='o', color=self.clbr_Nbl[i])
        ax8.set_xlim((60, -60))
        ax8.set_ylim((-60, 60))
        ax8.set_ylabel(r'v (M$\lambda$)')
        ax8.set_xlabel(r'u (M$\lambda$)')

        ax9.yaxis.set_label_position('right')
        ax9.yaxis.tick_right()
        if self.tellurics:
            ax9.set_ylim( ( 0.9,1.05*np.nanmax(self.Flux) ) )
            ax9.hlines( 1, -500, 500, linestyle='dashed', color='k' )
            ax9.vlines( 0, 0, 5, linestyle='dashed', color='k' )
            ax9.set_ylabel('Normalized flux')
        else:
            ax9.set_ylim( ( 0.9*np.nanmin(self.Flux),1.05*np.nanmax(self.Flux) ) )
            ax9.set_ylabel('Total flux (arbitrary units)')
        ax9.plot(self.vel.T, self.Flux.T, color='grey', linewidth=0.4, zorder=15)
        ax9.errorbar( self.vel.mean(0), self.Flux.mean(0), yerr=self.e_Flux.mean(0), capsize=2, marker='o', mfc='white', color='k', linestyle='None', label='Data', zorder=15)
        if self.line_defined :
            for i in range(self.Nvel):
                ax9.errorbar(self.vel_line[:,i], self.Flux_line[:,i], yerr=self.e_Flux_line[:,i], ms=7, mfc=self.clbr_Nvel[i], color='k', marker='o', linestyle='None', zorder=16)
            ax9.fill_betweenx([-5,5], vmin, self.vlineMin, color='grey', alpha=0.2)
            ax9.fill_betweenx([-5,5], self.vlineMax, vmax, color='grey', alpha=0.2)
        ax9.set_xlim(( vmin, vmax))
        ax9.set_xlabel('Velocity (km/s)')
        
        return fig
    

    def plot_spectrum(self, x_unit='micron', vmin=-1000, vmax=1000):
        fig, ax = plt.subplots()
        if x_unit == 'nano' :
            x = self.wvl*1e9
            xliminf, xlimsup = self.wvl0*1e9 * ( 1 + vmin*1000/csts.c ), self.wvl0*1e9 * ( 1 + vmax*1000/csts.c )
        elif x_unit == 'meters' :
            x = self.wvl
            xliminf, xlimsup = self.wvl0 * ( 1 + vmin*1000/csts.c ), self.wvl0 * ( 1 + vmax*1000/csts.c )
        elif x_unit == 'vel' :
            x = (self.wvl/self.wvl0 - 1)*csts.c/1000
            xliminf, xlimsup = vmin, vmax
        else : #Microns
            x = self.wvl*1e6
            xliminf, xlimsup = self.wvl0*1e6 * ( 1 + vmin*1000/csts.c ), self.wvl0*1e6 * ( 1 + vmax*1000/csts.c )
        
        ax.set_prop_cycle( color=self.clbr_Nobs )
        if all_equal(self.array) == "CHARA":
            ax.plot(x.T, np.nanmedian(self.Flux,1).T, marker='o')
            ax.set_ylabel('Median flux (arbitrary units)')
        else :
            if self.tellurics :
                ax.plot( x.T, self.Flux.T)
                ax.hlines(1.0,np.nanmin(x), np.nanmax(x), ls='dashed', color='k')
                ax.set_ylabel('Normalized flux')
            else :
                ax.plot(np.nanmedian( x, axis=1 ).T, np.nanmedian(self.Flux,axis=1).T, color='k', label='Median spectrum')
                for i in range(self.Nobs):
                    ax.plot( x[i].T, self.Flux[i].T)
                ax.set_ylabel('Total uncalibrated flux')
                ax.legend()
            
        ax.set_xlabel(r'Wavelength ($\mu$m)')
        ax.set_xlim(xliminf, xlimsup)
        return ax



    def plot_Vcurve(self, axis_unit="1/rad", logscale=False, legend=True):
        ### A function that plots V2 as a function of frequency
        fig, ax = plt.subplots()
        if axis_unit == "MLamb":
            ax.set_xlabel(r'Spatial frequency (M$\lambda$)')
            freqs = self.freqs*1e-6
        elif axis_unit == "meters":
            ax.set_xlabel(r'Spatial frequency (meters)')
            freqs = self.freqs*self.wvl[:,None,:]*1e-6
        elif axis_unit == "1/mas":
            ax.set_xlabel(r'Spatial frequency (cycles / mas)')
            freqs = self.freqs /180*np.pi/3600/1000
        else :
            ax.set_xlabel(r'Spatial frequency (cycles / rad)')
            freqs = self.freqs

        ax.hlines(1, 0, 1.1*freqs.max(), color='k', lw=1.1)
        for j in range(self.Nbl):
            for i in range(self.Nobs):
                if i == 0 :
                    ax.errorbar(freqs[i,j], self.V2[i,j], yerr=self.e_V2[i,j], markerfacecolor=self.clbr_Nbl[j], markeredgecolor='None', ecolor='grey', ls='None', capsize=3, marker='o', markersize=5, label=self.BLs[j])
                else:
                    ax.errorbar(freqs[i,j], self.V2[i,j], yerr=self.e_V2[i,j], markerfacecolor=self.clbr_Nbl[j], markeredgecolor='None', ecolor='grey', ls='None', capsize=3, marker='o', markersize=5)
            
        ax.grid(), ax.set_xlim(0,1.1*freqs.max())
        if logscale :
            ax.set_yscale('log'), ax.set_ylim(1e-4,1.5)
        else :
            ax.set_ylim(0,1.05)
        ax.set_ylabel("Visibility squared")
        if legend :
            ax.legend(ncols=2, title='Baselines')
        return ax








