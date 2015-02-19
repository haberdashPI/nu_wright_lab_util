import os.path
import pandas as pd
import pickle
import pkgutil
import pystan
import numpy as np
import appdirs

from patsy import dmatrix, dmatrices
from scipy.stats import sem

class SampledModel:
    def __init__(self,fit,formula,mms):
        self.formula = formula
        self.fit = fit
        self.mms = mms
        self.samples = fit.extract()

    def summary(self):
        def coef_row(i,name,samples):
            p_value = max(2*np.mean(np.sign(samples['coefs'][:,i]) != \
                                    np.sign(np.mean(samples['coefs'][:,i]))),
                          2.0/len(samples['coefs']))
            sig = ' '
            if p_value < 0.001: sig = '***'
            elif p_value < 0.01: sig = '**'
            elif p_value < 0.05: sig = '*'
            elif p_value < 0.1:  sig = '.'

            return {'name': name,
                    'mean': np.mean(samples['coefs'][:,i]),
                    'SE': sem(samples['coefs'][:,i]),
                    'p-value': p_value, ' ': sig}

        table = pd.DataFrame([coef_row(i,name,self.samples)
                              for i,name in
                              enumerate(self.mms[1].design_info.column_names)])
        table = table[['name','mean','SE','p-value',' ']]
        print table

class OptModel:
    def __init__(self,fit,formula,mms):
        self.fit = fit
        self.formula = formula
        self.mms = mms

    def params(self):
        return {name: self.fit['coefs'][i]
                for i,name in enumerate(self.mms[1].design_info.column_names)}

def get_model(file_name,use_cache=True):

    if isinstance(use_cache,basestring):
        object_dir = '.'
        object_file = use_cache
    else:
        object_dir = appdirs.user_cache_dir("pylab_util","David Little")
        object_file = os.path.join(object_dir,file_name+'.o')
    
    if not os.path.isfile(object_file) or not use_cache:
        model_file = pkgutil.get_data('pylab_util','stan/'+file_name+'.stan')
        model = pystan.StanModel(model_code = model_file.decode())

        try: os.makedirs(object_dir)
        except OSError:
            if not os.path.isdir(object_dir): raise

        with open(object_file,'wb') as f: pickle.dump(model,f)
    else:
        with open(object_file,'rb') as f: model = pickle.load(f)

    return model
    
        
def blm(formula,data,optimize=False,use_cache=True,**keys):
    mms = dmatrices(formula,data,eval_env=1)
    model = get_model('mean.model',use_cache)

    if not optimize:
        return SampledModel(model.sampling(data={"N": mms[1].shape[0],
                                                 "K": mms[1].shape[1],
                                                 "x": mms[1],"y": mms[0]},
                                                 **keys),
                                                 formula,mms)
    else:
        return OptModel(model.optimizing(data={"N": mms[1].shape[0],
                                               "K": mms[1].shape[1],
                                               "x": mms[1],
                                               "y": np.squeeze(mms[0])},**keys),
                                               formula,mms)

def read_blmm(file):
    return StoredSampledMultiModel(file)

class StoredSampledMultiModel:
    def __init__(self,file):
        store = pd.HDFStore(file,'r')
        self.y_hat = store['y_hat']
        self.error = store['error']
        self.log_prob = store['log_prob']
        self.ind_coefs = store['ind_coefs']
        self.group_coefs = store['group_coefs']
        self.group_cov = store['group_cov']
        self.formula = store.get_storer('ind_coefs').attrs.formula
        self.group_formula = store.get_storer('group_coefs').attrs.formula
        self.groupby = store.get_storer('group_coefs').attrs.groupby
        store.close()

class SampledMultiModel:
    def __init__(self,fit,formula,group_formula,groupby,ind_mms,group_mm,group_keys):
        self.fit = fit
        self.formula = formula
        self.group_formula = group_formula
        self.groupby = groupby

        samples = fit.extract()

        self.y_hat = pd.DataFrame(samples['y_hat'])
        self.error = pd.Series(samples['sigma'],name='error')
        self.log_prob = pd.Series(samples['lp__'],name='log_prob')
        
        self.ind_coefs = self.__ind_coefs(samples,ind_mms,group_keys)
        self.group_coefs = self.__group_coefs(samples,ind_mms,group_mm)
        self.group_cov = self.__group_cov(samples,ind_mms)

    def to_hdf(self,file,*params,**kwparams):
        store = pd.HDFStore(file,*params,**kwparams)
        store['y_hat'] = self.y_hat
        store['error'] = self.error
        store['log_prob'] = self.log_prob
        store['ind_coefs'] = self.ind_coefs
        store['group_coefs'] = self.group_coefs
        store['group_cov'] = self.group_cov
        store.get_storer('ind_coefs').attrs.formula = self.formula
        store.get_storer('group_coefs').attrs.formula = self.group_formula
        store.get_storer('group_coefs').attrs.groupby = self.groupby
        store.close()
        
    def __ind_coefs(self,samples,ind_mms,group_keys):
        beta = np.array(samples['beta'])
        n_samples = beta.shape[0]
        beta = beta.flatten()
        coefs = ind_mms[1].design_info.column_names

        indices = np.tile(np.repeat(group_keys.index,len(coefs)),n_samples)
        group_labels = group_keys.ix[indices].reset_index(drop=True)
        betam = {'value': beta,
                 'index': np.repeat(np.arange(n_samples),
                                    len(coefs)*len(group_keys)),
                 'coef': np.tile(coefs,len(group_keys) * n_samples)}
        betad = pd.DataFrame(betam)
        return pd.concat([betad,group_labels],axis=1)

    def __group_coefs(self,samples,ind_mms,group_mm):
        gamma = np.array(samples['gamma'])
        n_samples = gamma.shape[0]
        ind_coefs = ind_mms[1].design_info.column_names
        group_coefs = group_mm.design_info.column_names

        gammad = pd.DataFrame({'value': gamma.flatten(),
                               'index': np.repeat(np.arange(n_samples),
                                                  len(ind_coefs)*len(group_coefs)),
                               'coef': np.tile(ind_coefs,len(group_coefs)*n_samples),
                               'gcoef': np.tile(np.repeat(group_coefs,len(ind_coefs)),
                                                n_samples)})

        return gammad

    def __group_cov(self,samples,ind_mms):
        gcov = np.array(samples['Sigma_beta'])
        n_samples = gcov.shape[0]
        coefs = ind_mms[1].design_info.column_names
        
        gcovd = pd.DataFrame({'value': gcov.flatten(),
                            'index': np.repeat(np.arange(n_samples),
                                               len(coefs)**2),
                            'coef1': np.tile(coefs,n_samples*len(coefs)),
                            'coef2': np.tile(np.repeat(coefs,len(coefs)),
                                             n_samples)})

        return gcovd


def unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]
    
def blmm(formula,data,groupby,group_formula="",optimize=False,use_cache=True,
         group_init = None,group_mean_prior = 5,group_var_prior = 2.5,
         group_cor_prior = 2,prediction_error_prior = 1,**keys):
    
    ind_mms = dmatrices(formula,data,eval_env=1)
    grouped = data.groupby(groupby)
    group_mm = dmatrix(group_formula,grouped.head(1),eval_env=1)
    group_labels = np.array(grouped.grouper.labels).T
    unique_labels = unique_rows(group_labels)
    group_index_map = {tuple(labels): i
                       for i,labels in enumerate(unique_labels)}
    group_indices = np.array([group_index_map[tuple(labels)]
                              for labels in group_labels])+1
    
    group_keys = pd.DataFrame([ {grouped.grouper.names[j]:
                                 grouped.grouper.levels[j][labels[j]]
                                 for j in range(len(labels))}
                               for labels in unique_labels ])

    model = get_model('ind.model',use_cache)
    
    if not group_init is None:
        def init_fn():
            return {"gamma": np.vstack([np.matrix(group_init),
                                        np.random.rand(group_mm.shape[1]-1,
                                                       ind_mms[1].shape[1])]),
                    "z": np.random.rand(ind_mms[1].shape[1],
                                        group_mm.shape[0])+0.001,
                    "L_Omega": np.zeros((ind_mms[1].shape[1],
                                         ind_mms[1].shape[1])),
                    "tau": np.random.rand(ind_mms[1].shape[1])+0.001,
                    "sigma": np.random.rand()}
    else:
        def init_fn():
            return {"gamma": np.random.rand(group_mm.shape[1],ind_mm.shape[1]),
                    "z": np.random.rand(ind_mms[1].shape[1],
                                        group_mm.shape[0])+0.001,
                    "L_Omega": np.zeros(ind_mms[1].shape[1],
                                        ind_mms[1].shape[1]),
                    "tau": np.random.rand(ind_mms[1].shape[1])+0.001,
                    "sigma": np.random.rand()}

    fit = model.sampling(data={"N": ind_mms[1].shape[0],
                               "K": ind_mms[1].shape[1],
                               "J": group_mm.shape[0], "L": group_mm.shape[1],
                               "jj": group_indices,
                               "x": ind_mms[1], "u": group_mm,
                               "y": np.squeeze(ind_mms[0]),
                               "prediction_error_prior": prediction_error_prior,
                               "group_mean_prior": group_mean_prior,
                               "group_var_prior": group_var_prior,
                               "group_cor_prior": group_cor_prior},
                               init = init_fn,
                               **keys)

    return SampledMultiModel(fit,formula,group_formula,groupby,ind_mms,group_mm,group_keys)
