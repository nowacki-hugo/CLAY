"""
Part of the CLAY code (Continuum and Line Analysis of YSOs)
Used to model a dataset with a synthetic model
"""
import emcee
import numpy as np
import matplotlib.pyplot as plt
import clay.objects as objs
import clay.datahandling as dh
import scipy.stats as stats
import corner

def compute_chi2():
    chi2 = sum(  )

def plot_chain(model, sampler):
        labels = list(model.model.keys)
        flat_samples = sampler.get_chain()
        for x in range(model.ndim):
            figchain, ax = plt.subplots(figsize=(8, 4))
            ax.plot(flat_samples[:, :, x], "k", alpha=0.3)
            ax.set_ylabel(labels[x])
        return ax
    
def plot_corner(model, sampler):
    labels = list(model.model.keys)
    flat_samples = sampler.get_chain(discard=int(0.5*model.nsteps), thin=15, flat=True)
    results = []
    for i in range(model.ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        results.append(mcmc[1])
    figc = corner.corner(flat_samples, quantiles=[0.16,0.5,0.84], labels=labels,show_titles=True, title_fmt=".4f",
                     truths=[*results])
    return figc

def get_bestModel(model, sampler):
    flat_samples = sampler.get_chain(discard=int(0.5*model.nsteps), thin=15, flat=True)
    results = []
    m_error = []
    p_error = []
    
    for i in range(model.ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        results.append(mcmc[1])
        m_error.append(mcmc[1]-mcmc[0])
        p_error.append(mcmc[2]-mcmc[1])
    dict_params = { i:n for i,n in zip( model.model.keys,results ) }
    bestmodel = objs.model(model.u, model.v, model.Lambda, model.model.model, dict_params)
    return ( bestmodel )

# def eval_model(model):
#     N_fp = model.ndim
#     N = model.Vtot.size
#     V2chi2 = compute_chi2()
#     EV2chi2 = np.sqrt((np.sum((N*(((pV_med**2 - modelV2.T)/ (2*pV_med*err_pV_med))**2))**2)/(N*(N-1))) - (V2chi2**2)/(N-1))
#     nu = (N-1)-N_fp 
#     V2chi2r = V2chi2/nu
#     EV2chi2r = EV2chi2/nu

#     #N = modelCP.size
#     #CPchi2 = np.sum(((cph - modelCP)/ cph_err)**2)
#     #ECPchi2 = np.sqrt((np.sum((N*(((cph - modelCP)/ cph_err)**2))**2)/(N*(N-1))) - (CPchi2**2)/(N-1))
#     #nu = (N-1)-N_fp 
#     #CPchi2r = CPchi2/nu
#     #ECPchi2r = ECPchi2/nu
    
#     Nv, Nc = modelV2.size,  modelCP.size
#     chi2 = V2chi2 + CPchi2
#     echi2 = np.sqrt((EV2chi2)**2+(ECPchi2)**2)
#     nu = (Nv+Nc-1)-N_fp                                 
#     chi2r = chi2/nu
#     echi2r = echi2/nu
#     return f'chi2r={np.round(chi2r,decimals=2)} +/- {np.round(echi2r,decimals=2)}'


class fitter:
    ### A class that gathers data and model to eventually fit the model to the data
    ### methods to : initialize priors, run a fit, NOT plot the results
    ### interesting Plots :
    ### Data alone / Model alone / Model vs data (ALL PRIOR TO FIT, "fit" CLASS OTHERWISE)
    def __init__(self, data, model):
        self.u, self.v = data.u*1e6, data.v*1e6
        self.freqs, self.freqsCP = data.freqs, data.freqsCPhi
        self.Lambda = data.Lambda
        self.BL_idx, self.TR_idx = data.BL_idx, data.TR_idx
        self.Vtotmod = model.Vtot
        self.V2dat, self.e_V2dat = data.V2, data.e_V2
        self.CPhidat, self.e_CP = data.CPhi, data.e_CPhi
        # self.mod_params
        self.model = model
    # self.args = (argx, argy, arge)
    
    def __log_likelihood(self, params, x, ydat, yerr):
        dict_params = {i:n for i,n in zip(self.model.keys,params)}
        # print(dict_params["la"])
        upd_mod = objs.model(self.u, self.v, self.Lambda, self.model.model, dict_params)
        if not( ('V2' in self.fitted) or ('CPhi' in self.fitted)):
            print('Nothing to fit, need V2 or CPhi, or both.')
        elif not ('V2' in self.fitted) :
            _, CP = objs.compute_CPhi( self.freqs, upd_mod.Vtot, self.BL_idx, self.TR_idx )
            ymod = CP.flatten()
        elif not ('CPhi' in self.fitted) :
            ymod = (abs(upd_mod.Vtot)**2).flatten()
        else :
            _, CP = objs.compute_CPhi( self.freqs, upd_mod.Vtot, self.BL_idx, self.TR_idx )
            ymod = np.concatenate( ((abs(upd_mod.Vtot)**2).flatten(), CP.flatten()) )
        
        pr = -0.5*np.sum( (ydat-ymod)**2 / yerr**2 )
        return pr
    
    def __log_prior(self, params):
        for p, l, L in zip(params, self.lims[0],self.lims[1]):
            if not (l <= p <= L):
                return -np.inf
        return 0.0
    
    def __log_probability(self, params, x, y, yerr):
        lp = self.__log_prior( params )
        if not np.isfinite( lp ):
            return -np.inf
        return lp + self.__log_likelihood( params, x, y, yerr )
    
    def prepare_run( self, nwalkers, nsteps, pos_init=[], prior=[], lims=[], sigma=[], fitted=['V2','CPhi'] ):
        ### nwalkers [int] (1) = number of walkers used.
        ### nstep [int] (1) = number of steps performed by each walker overall.
        ### pos_init [float] (Nparams) = [pos1, pos2] list of values taken as initial guess for each parameter 
        ### prior[str] (Nparams/1) = [prior1, prior2]  string or list of string detailing the priors applied 
        ###                                 to each parameter (list) or for all (str). Possible values
        ###                                 include "gaussian" or "uniform".
        ### sigma [float] (Nparams) = Gaussian dispersion around the initial guess (not used for "uniform").
        self.nwalkers, self.nsteps = nwalkers, nsteps 
        self.pos_init = pos_init
        self.ndim = len(pos_init.values())
        self.lims = lims
        self.fitted=fitted
        if not (len(pos_init.values()) == 0) :
            if prior == "gaussian":
                self.pos = np.zeros((self.nwalkers, self.ndim)) 
                for p in self.pos:
                    p[:] = stats.truncnorm((np.asarray(self.lims[0]) - np.asarray(list(pos_init.values())))/sigma, 
                                           (np.asarray(self.lims[1]) - np.asarray(list(pos_init.values())))/sigma, 
                                           loc=list(pos_init.values()), scale=sigma).rvs()
            elif prior == "uniform" :
                self.pos = np.random.uniform( low=self.lims[0], high=self.lims[1], size=( self.nwalkers, self.ndim))
            else :
                print( 'Error : type of initial conditions unknown, sets uniform' )
                self.pos = np.random.uniform( low=self.lims[0], high=self.lims[1], size=(self.nwalkers, self.ndim) )
            

        else :
            return "Error : no prior given, cannot prepare the run"
        
        if not( ('V2' in fitted) or ('CPhi' in fitted)):
            print('Nothing to fit, need V2 or CPhi, or both.')
        elif not ('V2' in fitted) :
            argsx = self.freqsCP.flatten()
            argsy = self.CPhidat.flatten()
            argse = self.e_CP.flatten()
        elif not ('CPhi' in fitted) :
            argsx = self.freqs.flatten()
            argsy = self.V2dat.flatten()
            argse = self.e_V2dat.flatten()
        else :
            argsx = np.concatenate( (self.freqs.flatten(), self.freqsCP.flatten()) )
            argsy = np.concatenate( (self.V2dat.flatten(), self.CPhidat.flatten()) )
            argse = np.concatenate( (self.e_V2dat.flatten(), self.e_CP.flatten()) )
        
        self.args = ( argsx, argsy, argse )
    
    def run_fit(self, filename = "CLAY_MCMC_Run.h5") :
        backend = emcee.backends.HDFBackend( filename )
        backend.reset( self.nwalkers, self.ndim )
        sampler = emcee.EnsembleSampler( self.nwalkers, self.ndim, self.__log_probability, args=self.args )
        sampler.run_mcmc( self.pos, self.nsteps, progress=True )
        return sampler

    # def set_fixed_params(self, params={}):
    #     self.free_params
    
    # def set_free_params(self, params={}):
    #     self.free_params

class fit:
    ### A class that recovers the result file from a previous fit to play with it
    ### 
    def __init__(self, file_path):
        self.fpath = file_path