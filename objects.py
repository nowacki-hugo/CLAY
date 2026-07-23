#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONTINUUM AND LINE ANALYSIS OF YSOS -- CLAY
objects.py -- Allows to create basic objects and synthetic data
Author : H. Nowacki  (hugo.nowacki@oca.eu)
Version : 0.2.0 (07/2026)
Licence : Creative Commons (CC)
No reference for the code yet -- please contact for academic use 
"""

import numpy as np
import scipy.constants as csts
from scipy.stats import norm
from scipy import special
import matplotlib.pyplot as plt
from itertools import combinations

G_SI = csts.G       # SI
G_cgs = G_SI*1000   # cgs
c = csts.c/1000     # km/s
MSUN = 1.989E30     # kg
RSUN = 696342000    # m
YEAR = 31540000     # sec

def daysToSec(days):
    sec = days*86400
    return(sec)
def secToDays(sec):
    days = sec/86400
    return(days)

def mas2rad(mas):
    rad = mas/1000/3600*np.pi/180
    return(rad)
def rad2mas(rad):
    mas = rad*180/np.pi*3600*1000
    return(mas)

def compute_CPhi( freqs, Vtot, BL_idx, TR_idx ):
    #### Need arrays of size (Nobs, Nbl, Nwl)
    freqsCPhi, CPhi = [], []
    Phi = np.angle(Vtot, deg=True)
    
    for idx in TR_idx :
        a = list(combinations(idx, 2))
        f = []
        c = []
        for test in a :
            i = BL_idx.index(test)
            c.append(Phi[:,i,:])
            f.append(freqs[:,i,:])
        i = np.argmax( np.asarray(f).mean(axis=(1,2)) )
        cp = c[0] + c[1] - c[2]
        # print(np.shape(Phi),np.shape(c), np.shape(cp))
        freqsCPhi.append( f[i] ), CPhi.append(cp)
    
    freqsCPhi = np.transpose(np.asarray(freqsCPhi),(1,0,2))
    CPhi = np.transpose(np.asarray(CPhi),(1,0,2))
    
    return freqsCPhi, CPhi


class YSO:
    """
    Define the protostar itself and its properties
    Input :
        'params' is a dictionary listing stellar properties
    Need : a series of methods that derive some other properties (Rco, Rt, ...)
    """
    def __init__(self, params: dict):
        self.expected_keys = ["M", "e_M", "R", "e_R", "Prot", "e_Prot", "Bdip", "e_Bdip", "Mdot", "e_Mdot"]
        """
        # For each expected key, an attribute is initialized,
        # either with the value from 'params',
        # or with None if the key is not in 'params'
        """
        for key in self.expected_keys:
            setattr(self, key, params.get(key, None))
    
    def setParam(self, key, value):
        if key in (self.expected_keys) :
            setattr(self, key, value)
        else :
            raise ValueError(f'Key "{key}" not recognized, try "expected_keys"')
        
    
    def compute_Rco(self):
        """
        Computes the corotation radius (i.e. radius at which Keplerian orbit matches rotation rate)

        """
        if not (self.M and self.R and self.Prot) :
            raise ValueError("Not enough params for Rco,\n"
                  "Need : M, R, Prot and their errors \n"
                  "Set to default = None")
        else :
            self.Rco = (G_SI*self.M*MSUN*daysToSec(self.Prot)**2/(4*np.pi**2))**(1/3)/(self.R*RSUN)
            self.e_Rco = self.Rco * np.sqrt( (self.e_M/self.M/3)**2 + (self.e_Prot/self.Prot*2/3)**2 +(self.e_R/self.R)**2 )
            return self.Rco, self.e_Rco
    
    def compute_Rt(self, method='P'):
        """
        Computes the truncation radius, where gas pressure equals magnetic pressure. 
        i.e. where the material is lifted up from the disk's midplane
        Method == 'P' uses Pantolmos+ (in prep.) prescription, 'B' uses Bessolaz et al. (2008)
        """
        if ((method != 'B') and (method != 'P')):
            print("Method not recognized, either 'B' or 'P'. \n"
                  "Assumed to be 'P' in the following.")
        elif method == 'B':
            ### Bessolaz et al. (2008) formula, not accounting for rotation.
            if not (self.M and self.e_M and self.R and self.e_R and self.Mdot and self.e_Mdot and self.Bdip and self.e_Bdip):
                print("Missing parameter (or error) to compute Rt, see doc.")
            else :
                self.Rt = 2*(self.Bdip/2/140)**(4/7) * (self.Mdot/1E-8)**(-2/7) \
                *(self.M/0.8)**(-1/7) * (self.R/2)**(5/7) 
                self.e_Rt = self.Rt * np.sqrt( (self.e_M/self.M/7)**2 + (self.e_R/self.R*5/7)**2 
                                              +(self.e_Mdot/self.Mdot*2/7)**2 + (self.e_Bdip/self.Bdip*4/7)**2 )
        else:
            ### Pantolmos et al. (in prep) formula, accounting for rotation.
            if not (self.M and self.e_M and self.R and self.e_R and self.Mdot and self.e_Mdot and self.Bdip and self.e_Bdip and self.Prot and self.e_Prot):
                print("Missing parameter (or error) to compute Rt, see doc.")
            else :
                fs = 2*np.pi *self.R*RSUN / daysToSec(self.Prot) / np.sqrt(G_SI*self.M*MSUN/(self.R*RSUN))
                Ys = (self.Bdip/2*self.R*(RSUN*100))**2 / (4*np.pi*self.Mdot*(MSUN*1000/YEAR)) / np.sqrt(2*G_cgs*(self.M*MSUN*1000)/(self.R*RSUN*100))
                dfs = fs* np.sqrt( (3/2*self.e_R/self.R)**2 + (1/2*self.e_M/self.M)**2 + (self.e_Prot/self.Prot)**2 )
                dYs = Ys* np.sqrt((5/2*self.e_R/self.R)**2 + (2*self.e_Bdip/self.Bdip)**2 + (self.e_Mdot/self.Mdot)**2 + (1/2*self.e_M/self.M)**2)
                self.Rt = 0.88 * fs**(-0.57) * Ys**(0.057)
                self.e_Rt = self.Rt * np.sqrt( (0.57*dfs/fs)**2 + (0.057*dYs/Ys)**2 )

        return self.Rt, self.e_Rt
                
                
    def __repr__(self):
        return (
            f"===========================================\n"
            f"YSO object (as defined in CLAY) \n"
            f"Main properties :\n"
            f"M= {self.M} Msun | R= {self.R} Rsun | Bdip= {self.Bdip} G \n"
            f"===========================================\n"
        )

# class dataSet():
#     ### Regroups the set of interferometric observations, need to make it possible for both VLTI(1st) and CHARA(2nd) data
#     ### Need : methods that compute pure-line quantities
#     def __init__(self, listOfFiles, dorigin="VLTI"):
#         # Function that extracts everything needed, depending on "dorigin"



class model :
    """
    Allows to compute synthetic observations
    Can be fitted to a dataset
    Needs : point, background, rings, disk, Gaussian, Lorentzian, ...
    """
    def __init__(self, u=None, v=None, wvl=None, model=None, params={}):
        if type(wvl) == None :
            self.wvl = 2.16e-6
            self.wvl0 = 2.16e-6
        else :
            self.wvl = wvl
            self.wvl0 = np.nanmean(self.wvl)
        if type(u) == None  :
            self.u = np.linspace(-150,150,101)/self.wl0 # Unit = 1/rad
        else :
            self.u = u
        if type(v) == None :
            self.v = np.linspace(-150,150,101)/self.wl0 # Unit = 1/rad
        else :
            self.v = v
            
        self.uu, self.vv = np.meshgrid(self.u,self.v)
        self.freqs = np.hypot( self.u, self.v )
        
        if type(model) == dict :
            self.modeltype = model.get("type",'Cont')
            self.model = model.get('model')
        
        else :
            self.modeltype = 'Cont'
            self.model = model
        
        
        
        if model=='Punct' :
            self.params = {'x0':params.get("x0",0), 'y0':params.get("y0",0)}
            self.keys = self.params.keys
            self.Vtot = self.__pointSource(params)
        
        elif model == 'Gauss' :
            self.Vtot = self.__elongGauss( params )
            self.Model_params = {"x0":params.get('x0',0), "y0":params.get('y0',0), "a":params.get('a',1), "inc":params.get('inc',0), "pa":params.get('pa',0)}
            
        elif model =='LDD' :
            self.Vtot = self.__sqrtLD(params)
            self.params = {}
            self.keys = {}
            
        elif model=='Ellip' :
            self.Vtot = self.__Ellipsoid()
            self.Model_params = {"x0":self.x0,"y0":self.y0,"flor":self.flor,"a":self.a,"inc":self.inc,"pa":self.pa}
        
        elif model=='Sharp' :
            ### TO BE FIXED
            self.Vtot = self.__sharpDisk()
        
        elif model=='Ring' :
            Vr = self.__elongRing()
            self.Model_params = {"x0":self.x0,"y0":self.y0,"a":self.a,"inc":self.inc,"pa":self.pa,"c1":self.c1,"s1":self.s1}
            if self.type == "None":
                self.Vtot = Vr
            elif self.type == "Uniform":
                params_ker = self.params.get("params_ker", {"a":1})   
                self.Model_params.update({"params_ker", {"a":params_ker["a"]  }})
                self.Vtot = Vr * self.__UniformDisk(params_ker)
            elif self.type == "Smooth":
                params_ker = self.params.get("params_ker", {"flor":0,"a":1})
                self.Model_params.update({"params_ker", {"flor":self.flor,"a":params_ker["a"]}})
                self.Vtot = Vr * self.__Ellipsoid(params_ker)
            else :
                print('Unrecognized type of ring, set to "Smooth".')
                print('Possible types include : "None", "Uniform" and "Smooth".')
                params_ker = self.params.get("params_ker", {"a":1})  
                self.Vtot = Vr * self.__Ellipsoid(params_ker)
                            
        elif model=='Unif' :
            self.Vtot = self.__UniformDisk()
            self.Model_params = {"x0":self.x0,"y0":self.y0,"a":self.a,"inc":self.inc,"pa":self.pa,"c1":self.c1,"s1":self.s1}
        
        elif model=='Lazar' :
            self.Model_params = {"fs":params.get('fs',0),"fc":params.get('fc',0),"la":params.get('la',0), "lk":params.get('lk',0),"inc":np.deg2rad(params.get('inc',0)),"pa":np.deg2rad(params.get('pa',0)),"c1":params.get('c1',0),"s1":params.get('s1',0), "kc":params.get('kc',-1.6), "flor":params.get('flor',1)}    
            self.Vtot = self.__Lazareff(params)
        else :
            print('Model not recognized, set to "Lazar".')
            print('Possible models : "Punct", "Ellip", "Sharp", "Ring", "Unif" or "Lazar"')
        
        
    def __shiftFourier(self, Vc_in, x0, y0):
        """
        Shift the image (apply a phasor in Fourier space).
        """
        Vc_out = Vc_in * np.exp(-2j * np.pi * (self.u * x0 + self.v * y0))
        return Vc_out
    
    def __pointSource(self, params={}):
        """
        Compute complex visibility of a point source.
        Params:
        -------
        x0, y0: {float}
            Shift along x and y position [rad].
        """
        x0, y0 = mas2rad(params.get("x0", 0)), mas2rad(params.get("y0", 0))
        Vc_centered = np.ones(self.u.shape)
        Vc = self.__shiftFourier( Vc_centered, x0, y0 )
        return Vc
    
    def __elongLorentz(self, params={}):
        """
        Return the complex visibility of an ellongated Lorentzian
        of size a along the major axis = PA, 'a' is the radius
        and a x cos(i) along the minor axis, 'i' the inclination
        position angle PA, from North to East.
        """
        a = mas2rad(params.get("a", 1))
        inc, pa = np.deg2rad(params.get("inc", 0)), np.deg2rad(params.get("pa", 0))
        x0, y0 = params.get("x0", 0), params.get("y0", 0)
        
        rPA = np.pi/2 - pa
        uM = self.u * np.cos(rPA) - self.v * np.sin(rPA)
        um = self.u * np.sin(rPA) + self.v * np.cos(rPA)
        aq = np.sqrt( ( a*uM )**2 + ( a*um*np.cos(inc) )**2 )
        Vc = np.exp( -( 2*np.pi*aq )/np.sqrt(3) )
        if ((x0 != 0) or (y0 != 0)):
            return self.__shiftFourier(Vc, x0, y0)
        else :
            return Vc.astype(complex)
    
    def __elongGauss(self, params={}):
        """
        Return the complex visibility of an ellongated Gaussian
        of size a cosi (a is the radius),
        position angle PA, East from North.
        """
        a = mas2rad(params.get("a", 1))
        inc, pa = np.deg2rad(params.get("inc", 0)), np.deg2rad(params.get("pa", 0))
        x0, y0 = mas2rad(params.get("x0", 0)), mas2rad(params.get("y0", 0))
        
        rPA = np.pi/2 - pa
        uM = self.u * np.cos(rPA) - self.v * np.sin(rPA)
        um = self.u * np.sin(rPA) + self.v * np.cos(rPA)
        aq2 = ( a*uM )**2 + ( a*um*np.cos(inc) )**2
        
        Vc = np.exp(-np.pi**2 * aq2 / (np.log(2)))
        if ((x0 != 0) or (y0 != 0)):
            return self.__shiftFourier(Vc, x0, y0)
        else :
            return Vc.astype(complex)
        
    
    def __elongRing(self,params={}):
        """
        Return the complex visibility of an elongated ring
        of size a.cos(i) & position angle PA, East from North
        """
        a = mas2rad(params.get("a", 1))
        inc, pa = np.deg2rad(params.get("inc", 0)), np.deg2rad(params.get("pa", 0))
        c1, s1 = params.get("c1", 0), params.get("s1", 0)
        x0, y0 = mas2rad(params.get("x0", 0)), mas2rad(params.get("y0", 0))
        # Squeeze and rotation
        rPA = np.pi/2 - pa
        uM = self.u * np.cos(rPA) - self.v * np.sin(rPA)
        um = self.u * np.sin(rPA) + self.v * np.cos(rPA)
        # Polar coordinates (check angle)
        z = 2.0 * np.pi * a * np.hypot(uM, um*np.cos(inc))
        psi = np.arctan2(um, uM)
        # Modulation in polar
        rho1 = np.sqrt(c1**2 + s1**2)
        phi1 = np.pi/2 + np.arctan2(-c1, s1)
        # phi1 = np.arctan2(s1, c1)
        mod = -1.0j * rho1 * np.cos(psi - phi1) * special.jv(1, z)
        Vc = special.jv(0, z) + mod
        if ((x0 != 0) or (y0 != 0)):
            return self.__shiftFourier(Vc, x0, y0)
        else :
            return Vc.astype(complex)
    
    def __Ellipsoid(self, params={}):
        """
        Compute complex visibility of an ellipsoid.
        Params:
        -------
        `hwhm` {float}:
            HWHM of the disk [mas],\n
        `incl` {float}:
            Inclination of the disk [deg],\n
        `pa` {float}:
            Orientation of the disk [deg],\n
        `flor` {float}:
            Hybridation between purely gaussian (flor=0)
            and Lorentzian radial profile (flor=1).
        """
        flor = params.get("flor", 1)
        Vgauss = self.__elongGauss(params)
        Vlor = self.__elongLorentz(params) 
        Vc = (1 - flor) * Vgauss + flor * Vlor
        return Vc
        
    def __sharpDisk(self, params={}):
        #### TO BE FIXED
        """
        Compute complex visibility of a disk that has one sharp edge
        Params:
        -------
        `a` {float}:
            HWHM of the disk [mas],\n
        `inc` {float}:
            Inclination of the disk [deg],\n
        `pa` {float}:
            Orientation of the disk [deg],\n
        `x0, y0` {float}:
            shift of the disk in the image plan [mas],\n
        """
        # a = mas2rad(params.get("a", rad2mas(self.a)))
        inc, pa = np.deg2rad(params.get("inc", self.inc)), np.deg2rad(params.get("pa", self.pa))
        # x0, y0 = mas2rad(params.get("x0", rad2mas(self.x0))), mas2rad(params.get("y0", rad2mas(self.y0)))
        params_ker = params.get("params_ker", {"a":rad2mas(self.a)/3})
        rPA = np.pi/2 - pa
        uM = self.u * np.cos(rPA) - self.v * np.sin(rPA)
        um = self.u * np.sin(rPA) + self.v * np.cos(rPA)
        aq2 = ( mas2rad(params_ker["a"])*uM )**2 + ( mas2rad(params_ker["a"])*um*np.cos(inc) )**2
        
        Vker = np.exp(-np.pi**2 * aq2 / (np.log(2)))
        Vring = self.__elongRing(params)
        Vc = Vker * Vring
        # Vc[np.sqrt(aq2) < self.a] = 0
        
        return Vc
    
    def __UniformDisk( self, params={} ):
        #### TO BE FIXED
        """
        Compute complex visibility of a uniform disk
        Params:
        -------
        diam: {float}
            Radius of the disk [mas],\n
        x0, y0: {float}
            Position along x and y position [mas].
        """
        diam = 2* mas2rad(params.get("a", rad2mas(self.a)))
        inc, pa = np.deg2rad(params.get("inc", self.inc)), np.deg2rad(params.get("pa", self.pa))
        x0, y0 = mas2rad(params.get("x0", rad2mas(self.x0))), mas2rad(params.get("y0", rad2mas(self.y0)))
        
        rPA = np.pi/2 - pa
        uM = self.u * np.cos(rPA) - self.v * np.sin(rPA)
        um = self.u * np.sin(rPA) + self.v * np.cos(rPA)
        r = np.hypot( diam*uM , diam*um*np.cos(inc) )
        
        Vc = 2 * special.j1(np.pi * r * diam) / (np.pi * r * diam)
        filt = ~np.isfinite(Vc)
        Vc[filt] = 1
        if ((x0 != 0) or (y0 != 0)):
            return self.__shiftFourier(Vc, x0, y0)
        else :
            return Vc
        
    def __sqrtLD( self, params={} ):
        """
        Compute complex visibility of a Limb-darkened disk
        Params:
        -------
        LDD: {float}
            Limb-darkened diameter of the disk [mas],\n
        c, d: {float}
            Linear and square root limb darkening coefficients, respectively [no_unit].
        """
        nu = np.hypot( self.uu, self.vv )
        R = mas2rad(params.get("LDD", 1))/2
        alpha = 2*np.pi*nu*R
        c, d = params.get("c", -0.1445), params.get("d", 0.7511)
        
        unif_term = (1-c-d) * special.jv(1, alpha)/alpha                                        # i=0
        sqrt_term = d * 2**(1/4) * special.gamma(5/4) * special.jv(5/4,alpha) / alpha**(5/4)    # i=1
        linr_term = c * 2**(2/4) * special.gamma(3/2) * special.jv(3/2,alpha) / alpha**(3/2)    # i=2
        denom = (1-c-d)/2 + c/3 + 2*d/5 
        
        Vc = (unif_term+linr_term+sqrt_term)/denom
        
        return Vc
    
    def __Lazareff(self, params={}):
        """
        Compute complex visibility of a Lazareff model (star + thick ring + resolved
        halo). The halo contribution is computed with 1 - fc - fs.
        Params:
        -------
        `la` {float}:
            Half major axis of the disk (log),\n
        `lk` {float}:
            Kernel half light (log),\n
        `flor` {float}:
            Weighting for radial profile (0 gaussian kernel,
            1 Lorentizian kernel),\n
        `incl` {float}:
            Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)) [deg],\n
        `pa` {float}:
            Orientation of the disk (from north to East) [deg],\n
        `fs` {float}:
            Flux contribution of the star [%],\n
        `fc` {float}:
            Flux contribution of the disk [%],\n
        `ks` {float}:
            Spectral index compared to reference wave at 2.2 µm,\n
        `c1`, `s1` {float}:
            Cosine and sine amplitude for the mode 1 (azimutal changes),\n
        """
        wvl0, wvl = params.get("wvl0", self.wvl0), params.get("wvl", self.wvl)
        ks, kc = params.get("ks", -1.2), params.get("kc", 0)
        fs, fc = params.get("fs", 0.5), params.get("fc", 0.5)
        # fh = params.get("fh", 1-(fs+fc))
        la, lk = params.get("la", 1), params.get("lk", 0)
        
        kr = 10.**lk
        ar = 10.**la / np.sqrt(1 + kr**2)
        ak = ar * kr
        params_ker = {"a":ak} 
        params_ring = {"a":ar}
        Vkernel = self.__Ellipsoid(params_ker)    
        Vring = self.__elongRing(params_ring) * Vkernel
        
        fs_lambda = fs * (wvl0 / wvl) ** ks
        fc_lambda = fc * (wvl0 / wvl) ** kc
        fh_lambda = (1-fs-fc) * (wvl0 / wvl) ** ks
                
        if not (wvl.shape==(51,)) :
            F1 = fs_lambda[:, None,:] * self.__pointSource(params)
            F2 = fc_lambda[:, None,:] * Vring
            ftot = fs_lambda + fh_lambda + fc_lambda
            
            Vc = (F1 + F2) / ftot[:, None,:]
            return Vc.astype(complex)
        
        else :
            F1 = fs_lambda[None, None,:] * self.__pointSource(params)[:,:,None]
            F2 = fc_lambda[None, None,:] * Vring[:,:,None]
            ftot = fs_lambda + fh_lambda + fc_lambda
            Vc = (F1 + F2) / ftot[None, None,:]
            
            return Vc.astype(complex)
        
    
    
    def __update_params(self, params={}):
        self.fs = params.get("fs", self.fs)
        self.fc = params.get("fc", self.fc)
        self.fh = params.gte("fh", 1 - (self.fc+self.fs) )
        self.la = params.get("la", self.la)
        self.lk = params.get("lk", self.lk)
        if not ( self.la and self.lk ):
            self.a = mas2rad(params.get("a", self.a))
        else :
            kr = 10.0 **self.lk
            ar = 10**self.la / (np.sqrt(1 + kr**2))
            ak = ar * kr
            self.ar, self.ak = mas2rad(ar), mas2rad(ak)
        self.flor = params.get("flor", self.flor)
        self.pa = params.get("pa", self.pa)
        self.inc = params.get("inc", self.inc)
        self.x0 = mas2rad(params.get("x0", rad2mas(self.x0)))
        self.y0 = mas2rad(params.get("y0", rad2mas(self.y0)))
        self.ks = params.get("ks", self.ks)
        self.kc = params.get("kc", self.kc)
        self.c1 = params.get("c1", self.c1)
        self.s1 = params.get("s1", self.s1)
    # def setParam(self, key, value):
    #     setattr(self, key, value)
    
     
    def plot_V2map(self, axis_unit='MLamb'):
        #Plots the Visibility map
        
        if axis_unit == 'meters' :
            xaxis, yaxis = self.u*self.wl0, self.v*self.wl0
            xlabel, ylabel = 'u (meters)', 'v (meters)'
        elif axis_unit =='MLamb':
            xaxis, yaxis = self.u*1e-6, self.v*1e-6
            xlabel, ylabel = r'u (M$\lambda$)', r'v (M$\lambda$)'
        elif axis_unit =='1/rad':
            xaxis, yaxis = self.u/1e6, self.v/1e6
            xlabel, ylabel = r'u (x$10^{6}$ rad$^{-1}$)', r'v (x$10^{6}$ rad$^{-1}$)'
        else :
            print("axis_unit unknown, set to 1/radians")
            xaxis, yaxis = self.u, self.v
            xlabel, ylabel = r'u (x$10^{-6}$ rad$^{-1}$)', r'v (x$10^{-6}$ rad$^{-1}$)'
            
        fig, ax = plt.subplots()
        pl = ax.pcolor(xaxis, yaxis, abs(self.Vtot[:,:,len(self.wvl)//2])**2, vmin=0, vmax=1, cmap='nipy_spectral')
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        ax.set_title(r'$|V|^2$ map in the u-v plan'), ax.set_xlim(xaxis.max(), xaxis.min())
        fig.colorbar(pl)
        return fig
    
    def plot_V2curve(self, axis_unit='MLamb', angle=0):
        #Plots the Visibility map
        filtre = np.arctan2(self.vv,self.uu) == np.deg2rad(90-angle)
        if axis_unit == 'meters' :
            xaxis = np.hypot(self.uu, self.vv)[filtre]*self.wl0
            xlabel= 'freq (meters)'
        elif axis_unit =='MLamb':
            xaxis = np.hypot(self.uu, self.vv)[filtre]*1e-6
            xlabel = r'freq (M$\lambda$)'
        elif axis_unit =='1/rad':
            xaxis = np.hypot(self.uu, self.vv)[filtre]/1e6
            xlabel = r'freq (x$10^{6}$ rad$^{-1}$)'
        else :
            print("axis_unit unknown, set to 1/radians")
            xaxis= np.hypot(self.uu, self.vv)[filtre]/1e6
            xlabel = r'u (x$10^{-6}$ rad$^{-1}$)'
        fig, ax = plt.subplots()
        ax.plot(xaxis, abs(self.Vtot[filtre])**2)
        ax.set_xlabel(xlabel), ax.set_ylabel(r'$V^2$')
        # ax.set_title(r'$|V|^2$ map in the u-v plan'), ax.set_xlim(xaxis.min(), xaxis.max()), ax.set_ylim(0, 1.05)
        return fig
        
    def plot_image(self, axis_unit='mas', d=150, fromV2=True):
        # Function that plots the model image
        ### FromV2 = False NOT IMPLEMENTED YET means you have an image that doesn't use the FFT of the V2
        Umax, Umin, Vmax, Vmin = self.u.max(), self.u.min(), self.v.max(), self.v.min()
        Nu, Nv = self.u.size, self.v.size
        uscale, vscale = (Umax-Umin)/(Nu-1), (Vmax-Vmin)/(Nv-1)
        if axis_unit == 'rad' :
            xaxis, yaxis = np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale)), np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale))
            xlabel, ylabel = 'East (rad)', 'North (rad)'
        elif axis_unit =='mas':
            xaxis, yaxis = rad2mas(np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale))), rad2mas(np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale)))
            xlabel, ylabel = r'East (mas)', r'North (mas)'
        elif axis_unit =='au':
            xaxis, yaxis = rad2mas(np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale)))*d/1000, rad2mas(np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale)))*d/1000
            xlabel, ylabel = r'East (a.u.)', r'North (a.u.)'
        else :
            print("axis_unit unknown, set to mas")
            xaxis, yaxis = rad2mas(np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale))), rad2mas(np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale)))
            xlabel, ylabel = r'East (mas)', r'North (mas)'
            
        fig, ax = plt.subplots()
        if self.model == 'Lazar' :
            ax.pcolor(xaxis, yaxis, abs(np.fft.fftshift(np.fft.ifft2(self.Vtot[len(self.wvl)//2]))), cmap='turbo')
        else :
            ax.pcolor(xaxis, yaxis, abs(np.fft.fftshift(np.fft.ifft2(self.Vtot))), cmap='turbo')
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        return( ax )
    
    
    def __repr__(self):
        return (
            "===========================================\n"
            "YSO model (as defined in CLAY) \n"
            f"The type of model is : {self.model}\n"
            "===========================================\n"
        )


class complexModel:
    def __init__(self, models=[], f_ratios=[]):
        self.models = []
        self.fr = []
        if ((models[:] == []) or (f_ratios[:] == [])):
            print("Empty list of models / flux ratios")
            self.Vtot = 0
        else :
            self.Vtot = np.zeros(models[0].Vtot.shape).astype(complex)
            self.u, self.v = models[0].u, models[0].v
            self.uv_grid = (models[0].uu, models[0].vv)
            self.wl0 = models[0].wl0
            for m, f in zip( models, f_ratios ):
                self.Vtot += f * m.Vtot
                self.models.append(m.model), self.fr.append(f)
            self.Vtot /= sum(self.fr)
            
    def plot_V2map(self, axis_unit='MLamb'):
        #Plots the Visibility map
        
        if axis_unit == 'meters' :
            xaxis, yaxis = self.u*self.wl0, self.v*self.wl0
            xlabel, ylabel = 'u (meters)', 'v (meters)'
        elif axis_unit =='MLamb':
            xaxis, yaxis = self.u*1e-6, self.v*1e-6
            xlabel, ylabel = r'u (M$\lambda$)', r'v (M$\lambda$)'
        elif axis_unit =='1/rad':
            xaxis, yaxis = self.u/1e6, self.v/1e6
            xlabel, ylabel = r'u (x$10^{6}$ rad$^{-1}$)', r'v (x$10^{6}$ rad$^{-1}$)'
        else :
            print("axis_unit unknown, set to 1/radians")
            xaxis, yaxis = self.u, self.v
            xlabel, ylabel = r'u (x$10^{-6}$ rad$^{-1}$)', r'v (x$10^{-6}$ rad$^{-1}$)'
            
        fig, ax = plt.subplots()
        if "Lazar" in self.models :
            pl = ax.pcolor(xaxis, yaxis, abs(self.Vtot[len(self.wvl)//2])**2, vmin=0, vmax=1, cmap='nipy_spectral')
        else :
            pl = ax.pcolor(xaxis, yaxis, abs(self.Vtot)**2, vmin=0, vmax=1, cmap='nipy_spectral')
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        ax.set_title(r'$|V|^2$ map in the u-v plan'), ax.set_xlim(xaxis.max(), xaxis.min())
        fig.colorbar(pl)
        return fig
        
    def plot_Phimap(self, axis_unit='MLamb', deg=True):
        # Function that plots the model image
        if axis_unit == 'meters' :
            xaxis, yaxis = self.u*self.wl0, self.v*self.wl0
            xlabel, ylabel = 'u (meters)', 'v (meters)'
        elif axis_unit =='MLamb':
            xaxis, yaxis = self.u*1e-6, self.v*1e-6
            xlabel, ylabel = r'u (M$\lambda$)', r'v (M$\lambda$)'
        elif axis_unit =='1/rad':
            xaxis, yaxis = self.u/1e6, self.v/1e6
            xlabel, ylabel = r'u (x$10^{6}$ rad$^{-1}$)', r'v (x$10^{6}$ rad$^{-1}$)'
        else :
            print("axis_unit unknown, set to 1/radians")
            xaxis, yaxis = self.u, self.v
            xlabel, ylabel = r'u (x$10^{-6}$ rad$^{-1}$)', r'v (x$10^{-6}$ rad$^{-1}$)'
            
        fig, ax = plt.subplots()
        pl = ax.pcolor(xaxis, yaxis, np.angle(self.Vtot, deg=deg), cmap='gray')
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        fig.colorbar(pl)
        return(fig)
    
    
    def plot_image(self, axis_unit='mas', d=150, fromV2=True):
        # Function that plots the model image
        ### FromV2 = False NOT IMPLEMENTED YET means you have an image that doesn't use the FFT of the V2
        Umax, Umin, Vmax, Vmin = self.u.max(), self.u.min(), self.v.max(), self.v.min()
        Nu, Nv = self.u.size, self.v.size
        uscale, vscale = (Umax-Umin)/(Nu-1), (Vmax-Vmin)/(Nv-1)
        if axis_unit == 'rad' :
            xaxis, yaxis = np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale)), np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale))
            xlabel, ylabel = 'East (rad)', 'North (rad)'
        elif axis_unit =='mas':
            xaxis, yaxis = rad2mas(np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale))), rad2mas(np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale)))
            xlabel, ylabel = r'East (mas)', r'North (mas)'
        elif axis_unit =='au':
            xaxis, yaxis = rad2mas(np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale)))*d/1000, rad2mas(np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale)))*d/1000
            xlabel, ylabel = r'East (a.u.)', r'North (a.u.)'
        else :
            print("axis_unit unknown, set to mas")
            xaxis, yaxis = rad2mas(np.fft.fftshift(np.fft.fftfreq(self.u.size, d=uscale))), rad2mas(np.fft.fftshift(np.fft.fftfreq(self.v.size, d=vscale)))
            xlabel, ylabel = r'East (mas)', r'North (mas)'
            
        fig, ax = plt.subplots()
        ax.pcolor(xaxis, yaxis, abs(np.fft.fftshift(np.fft.ifft2(self.Vtot))), cmap='turbo')
        ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
        return( ax )
        
    def __repr__(self):
        return (
            "===========================================\n"
            "YSO complex model (as defined in CLAYS) \n"
            f"The models included are : {self.models}\n"
            f"With respective weights : {self.fr}\n"
            "===========================================\n"
        )