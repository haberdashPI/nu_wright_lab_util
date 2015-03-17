import re
import os.path
import pandas as pd
import pickle
import pkgutil
import pystan
import numpy as np
import appdirs
import scipy
import patsy
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
    mms = patsy.dmatrices(formula,data,eval_env=1)
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
    
def read_multi_fit(file):
    store = pd.HDFStore(file,'r')
    ind_formula = read_formula(store,'ind_coefs')
    group_formula = read_formula(store,'group_coefs')
    model = \
      SampledMultiModel(data = store['data'],
                        y_hat = store['y_hat'],
                        error = store['error'],
                        log_prob = store['log_prob'],

                        ind_coefs = store['ind_coefs'],
                        group_coefs = store['group_coefs'],
                        group_cov = store['group_cov'],

                        ind_formula = ind_formula,
                        group_formula = group_formula,
                        groupby = store.get_storer('group_coefs').attrs.groupby)
    store.close()

    return model

def create_multi_fit(data,fit,ind_formula,group_formula,groupby,ind_mms,group_mm,group_keys):
    samples = fit.extract()
    model = \
        SampledMultiModel(data = data,
                          y_hat = pd.DataFrame(samples['y_hat']),
                          error = pd.Series(samples['sigma'],name='error'),
                          log_prob = pd.Series(samples['lp__'],name='log_prob'),

                          ind_coefs = __ind_coefs(samples,ind_mms,groupby,group_keys),
                          group_coefs = __group_coefs(samples,ind_mms,group_mm),
                          group_cov = __group_cov(samples,ind_mms),

                          ind_formula = ind_formula,
                          group_formula = group_formula,
                          groupby = groupby)
    model.fit = fit
    return model

def __ind_coefs(samples,ind_mms,groupby,group_keys):
    beta = np.array(samples['beta'])
    n_samples = beta.shape[0]
    beta = beta.flatten()
    coefs = ind_mms[1].design_info.column_names

    indices = np.tile(np.repeat(group_keys.index.values,len(coefs)),n_samples)
    group_labels = group_keys.ix[indices].reset_index(drop=True)
    betam = {'value': beta,
             'sample': np.repeat(np.arange(n_samples),
                                len(coefs)*len(group_keys)),
             'coef': np.tile(coefs,len(group_keys) * n_samples)}
    betad = pd.DataFrame(betam)
    betad.coef = betad.coef.astype('category')
    betad.coef.cat.reorder_categories(coefs,inplace=True)

    ind_coefs = pd.concat([betad,group_labels],axis=1)
    
    if isinstance(groupby,list):
        return ind_coefs.set_index(['sample'] + groupby + ['coef'])
    else:
        return ind_coefs.set_index(['sample',groupby,'coef'])

def __group_coefs(samples,ind_mms,group_mm):
    gamma = np.array(samples['gamma'])
    n_samples = gamma.shape[0]
    ind_coefs = ind_mms[1].design_info.column_names
    group_coefs = group_mm.design_info.column_names

    gammad_map = \
      {'value': gamma.flatten(),
       'sample': np.repeat(np.arange(n_samples),
                           len(ind_coefs)*len(group_coefs)),
       'coef': np.tile(ind_coefs,len(group_coefs)*n_samples),
       'gcoef': np.tile(np.repeat(group_coefs,len(ind_coefs)),
                        n_samples)}
    gammad = pd.DataFrame(gammad_map)
    gammad.coef = gammad.coef.astype('category')
    gammad.coef.cat.reorder_categories(ind_coefs)
    gammad.gcoef = gammad.gcoef.astype('category')
    gammad.gcoef.cat.reorder_categories(group_coefs)

    return gammad.set_index(['sample','gcoef','coef'])

def __group_cov(samples,ind_mms):
    gcov = np.array(samples['Sigma_beta'])
    n_samples = gcov.shape[0]
    coefs = ind_mms[1].design_info.column_names

    gcovd = pd.DataFrame({'value': gcov.flatten(),
                          'sample': np.repeat(np.arange(n_samples),
                                              len(coefs)**2),
                          'coef1': np.tile(coefs,n_samples*len(coefs)),
                          'coef2': np.tile(np.repeat(coefs,len(coefs)),
                                           n_samples)})
    gcovd.coef1 = gcovd.coef1.astype('category')
    gcovd.coef1.cat.reorder_categories(coefs)
    gcovd.coef2 = gcovd.coef2.astype('category')
    gcovd.coef2.cat.reorder_categories(coefs)

    return gcovd.set_index(['sample','coef1','coef2'])

class SampledMultiModel:
    def __init__(self,data,y_hat,error,log_prob,ind_coefs,group_coefs,
                 group_cov,ind_formula,group_formula,groupby):
    
        self.data = data
        self.y_hat = y_hat
        self.error = error
        self.log_prob = log_prob

        self.ind_coefs = ind_coefs
        self.group_coefs = group_coefs
        self.group_cov = group_cov

        self.ind_formula = ind_formula
        self.group_formula = group_formula
        self.groupby = groupby
        
        
    def to_hdf(self,file,*params,**kwparams):
        store = pd.HDFStore(file,*params,**kwparams)
        store['data'] = self.data

        store['y_hat'] = self.y_hat
        store['error'] = self.error
        store['log_prob'] = self.log_prob

        store['ind_coefs'] = self.ind_coefs
        store['group_coefs'] = self.group_coefs
        store['group_cov'] = self.group_cov

        self.ind_formula.to_store(store,'ind_coefs')
        self.group_formula.to_store(store,'group_coefs')
        store.get_storer('group_coefs').attrs.groupby = self.groupby
        store.close()
        
    def predict_individuals(self,data):
        n_samples = len(self.ind_coefs.index.levels[0])
        n_coefs = len(self.ind_coefs.index.levels[2])
        data['row'] = range(data.shape[0])
        
        def find_group(data_for_group):
            try:
                return tuple(data_for_group[self.groupby].iloc[0].values)
            except AttributeError:
                return tuple(data_for_group[[self.groupby]].iloc[0])

        def find_coef_samples(group):
            group_slice = (slice(None),) + group + (slice(None),)
            coefs = self.ind_coefs.loc[group_slice,:].values.view()
            coefs.shape = (n_samples,n_coefs)

            return coefs.T

        def predict_group_ind(data_for_group):
            coefs = find_coef_samples(find_group(data_for_group))
            mm = self.ind_formula.dmatrices(data_for_group)[1]
            predictions = mm.dot(coefs)

            pred_cols = \
              pd.DataFrame({'prediction': predictions.flatten(),
                            'sample': np.tile(self.ind_coefs.index.levels[0],
                                              predictions.shape[0])})
            data_cols = \
              data_for_group.loc[np.repeat(data_for_group.index.values,
                                 predictions.shape[1])].reset_index(drop=True)
            del data_cols[self.groupby]
            return pd.concat([data_cols,pred_cols],axis=1)

        result = data.groupby(self.groupby).apply(predict_group_ind)
        result.sort(columns=['sample','row'],inplace=True)
        return result

    def __group_coefs(self,sample_index,gshape):
            gcoefs = self.group_coefs.loc[sample_index].reset_index()
            gcoefs = gcoefs.value.values.view()
            gcoefs.shape = gshape
            return gcoefs

    def __group_cov(self,sample_index,cov_shape):
            cov = self.group_cov.loc[sample_index].values.view()
            cov.shape = cov_shape
            return cov

    def group_mean_prediction(self,data):
        # setup group predictors
        grouped = data.groupby(self.groupby)
        group_cols = grouped.head(1).reset_index(drop=True)
        gmm = self.group_formula.dmatrix(group_cols)
        
        # setup individual predictors
        mm = self.ind_formula.dmatrices(data)[1]
        group_indices,_  = _organize_group_labels(grouped)
        gshape = (len(self.group_coefs.index.levels[1]),
                  len(self.group_coefs.index.levels[2]))

        def helper(sample_index):
            ind_coefs = gmm.dot(self.__group_coefs(sample_index,gshape))
            predictions = np.einsum('ij,ij->i',mm,ind_coefs[group_indices,:])
            
            d = data.copy()
            d['group_mean'] = predictions
            d['sample'] = sample_index
            return d
        return pd.concat([helper(i) for i in self.group_coefs.index.levels[0]])

    def sample_group(self,data):
        # setup group predictors
        grouped = data.groupby(self.groupby)
        group_cols = grouped.head(1).reset_index(drop=True)
        gmm = self.group_formula.dmatrix(group_cols)
        
        # setup individual predictors
        ##mm = self.ind_formula.dmatrices(data)[1]
        group_indices,_  = _organize_group_labels(grouped)
        gshape = (len(self.group_coefs.index.levels[1]),
                  len(self.group_coefs.index.levels[2]))
        cov_shape = (len(self.group_coefs.index.levels[2]),
                     len(self.group_coefs.index.levels[2]))

        def sample_ind_coefs(sample_index):
            mean_coefs = gmm.dot(self.__group_coefs(sample_index,gshape))
            cov = self.__group_cov(sample_index,cov_shape)
            coef_offsets = \
              np.random.multivariate_normal(np.zeros(mean_coefs.shape[1]),cov,
                                            mean_coefs.shape[0])
            return mean_coefs + coef_offsets
        
        def sample_groups(sample_index):
            ind_coefs = sample_ind_coefs(sample_index)
            coef_cols = pd.DataFrame(ind_coefs[group_indices,:],
                                     columns=self.ind_coefs.index.levels[2])
            coef_cols['sample'] = sample_index

            return pd.concat([group_cols,coef_cols],axis=1)

        return pd.concat(sample_groups(i)
                         for i in self.group_coefs.index.levels[0])

    def WAIC_fn(self):
        y = self.ind_formula.dmatrices(self.data)[0]
        N = len(y)
        samples = \
          np.array([scipy.stats.norm.pdf(self.y_hat.iloc[:,n] - y[n],self.error)
                    for n in range(N)])
        
        def waic(weights=1):
            try:
                if len(weights.shape) < 2: weights = weights[:,np.newaxis]
            except AttributeError: pass
            return -2*(np.log(np.mean(samples * weights,axis=1)) - \
                       np.std(np.log(samples) * weights,axis=1,ddof=1))

        waic.boot_size = N
        return waic

    def WAIC(self):
        fn = self.WAIC_fn()
        return fn()

def unique_rows(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return a[ui]

def _organize_group_labels(grouped):
    group_labels = np.array(grouped.grouper.labels).T
    unique_labels = unique_rows(group_labels)
    group_index_map = {tuple(labels): i
                       for i,labels in enumerate(unique_labels)}

    group_indices = np.array([group_index_map[tuple(labels)]
                              for labels in group_labels])
    
    group_keys = pd.DataFrame([ {grouped.grouper.names[j]:
                                 grouped.grouper.levels[j][labels[j]]
                                 for j in range(len(labels))}
                               for labels in unique_labels ])
    return group_indices,group_keys

def _init_fn(group_init,ind_mms,group_mm):
    if not group_init is None:
        def init():
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
        def init():
            return {"gamma": np.random.rand(group_mm.shape[1],ind_mm.shape[1]),
                    "z": np.random.rand(ind_mms[1].shape[1],
                                        group_mm.shape[0])+0.001,
                    "L_Omega": np.zeros(ind_mms[1].shape[1],
                                        ind_mms[1].shape[1]),
                    "tau": np.random.rand(ind_mms[1].shape[1])+0.001,
                    "sigma": np.random.rand()}
    return init

def read_formula(store,key,error_if_absent=True):
    import dill
    builder = dill.loads(store.get_storer(key).attrs.formula)
    return SaveableFormula(builder)

def saveable_formulas(formula,data,eval_env):
    return SaveableFormula(patsy.incr_dbuilders(formula,lambda: iter([data]),
                           eval_env=eval_env+1))
def saveable_formula(formula,data,eval_env):
    return SaveableFormula(patsy.incr_dbuilder(formula,lambda: iter([data]),
                            eval_env=eval_env+1))

class SaveableFormula:
    def __init__(self,builders):
        self.builders = builders
    def dmatrix(self,data):
        return patsy.build_design_matrices([self.builders],data)[0]
    def dmatrices(self,data):
        return patsy.build_design_matrices(self.builders,data)
    def to_store(self,store,key):
        import dill
        store.get_storer(key).attrs.formula = dill.dumps(self.builders)

# def blmm(formula,data,use_cache=True,
#          fixed_mean_prior = 2.5,group_init = None,group_mean_prior = 5,
#          group_var_prior = 2.5,group_cor_prior = 2,prediction_error_prior = 1,
#          eval_env=0,**keys):

#     match = re.match(r'^(?P<outcome>\S+)\s*~\s*(?P<fixed>\S.+)'
#                      r'(+\s*\((?P<individual>\S[^\|]+)|'
#                      r'(?P<groupby>[^[]]+)'
#                      r'((?P<group>[^]]+)])?\s*\))?\s*$',formula)

#     outcome_str = match.group('outcome')
#     fixed_str = outcome_str + '~' + match.group('fixed')
#     ind_str = match.group('individual')
#     if match.group('group'):
#         group_formula = match.group('group')
#     else:
#         group_formula = ''
#     groupby = match.group('groupby').split(r'\s*+\s*',groupbystr)

#     fixed_formula = saveable_formulas(fixed_str,eval_env+1)
#     fixed_mms = fixed_formula.dmatrices(data)

#     ind_formula = saveable_formulas(ind_str,data,eval_env+1)
#     ind_mm = ind_formula.dmatrix(data)

#     grouped = data.groupby(groupby)
#     group_data = grouped.head(1)
#     group_formula = saveable_formula(group_formula,group_data,eval_env+1)
#     group_mm = group_formula.dmatrix(group_data)
#     group_indices,group_keys = _organize_group_labels(grouped)

#     model = get_model('mixed.model',use_cache)
        
#     fit = model.sampling(data={"N": ind_mm.shape[0],
#                                "K": ind_mm.shape[1],
#                                "J": group_mm.shape[0],
#                                "L": group_mm.shape[1],
#                                "M": fixed_mms[1].shape[1]
#                                "jj": group_indices+1,
#                                "x": ind_mms[1], "xf": fixed_mm, "u": group_mm,
#                                "y": np.squeeze(ind_mms[0]),
#                                "prediction_error_prior": prediction_error_prior,
#                                "group_mean_prior": group_mean_prior,
#                                "group_var_prior": group_var_prior,
#                                "group_cor_prior": group_cor_prior},
#                                init = _init_fn(group_init,ind_mms,group_mm),
#                                **keys)

#     return create_multi_fit(data,fit,fixed_formula,ind_formula,group_formula,groupby,
#                             ind_mms,group_mm,group_keys)

def regex_formula(df,pattern,formula,extract,eval_env=0):
    formula = saveable_formula(re.match(pattern,formula).group(extract),
                               df,eval_env+1)

    X = formula.dmatrix(df)
    return formula,X,X.shape[0],X.shape[1]

def regex_formulas(df,pattern,formula,extract,eval_env=0):
    formula = saveable_formulas(re.match(pattern,formula).group(extract),
                                df,eval_env+1)

    y,X = formula.dmatrices(df)
    return formula,np.squeeze(y),X,X.shape[0],X.shape[1]

def regex_get_groups(df,pattern,splitby,formula,extract):
    groups_str = re.match(pattern,formula).group(extract)
    groupby = map(lambda x: x.strip(),re.split(splitby,groups_str.strip()))
    group_df = df.groupby(groupby)
    group_indices,group_keys = _organize_group_labels(group_df)

    return group_df.head(1),group_indices,group_keys

def write_samples(samples,file,formulae,groupers,*params,**kwparams):
    store = pd.HDFStore(file,*params,**kwparams)
    for key in samples.keys():
        indices = list(np.where(np.ones(samples[key].shape)))
        index_str = ['sample'] + ['index%02d' % i for i in range(len(indices)-1)]
        values = samples[key].flatten()
        columns = index_str + ['value']
        store[key] = pd.DataFrame(dict(zip(columns,indices + [values])),columns = columns)

        if formulae.has_key(key):
            print "storing formula: ",key
            formulae[key].to_store(store,key)
        if groupers.has_key(key):
            print "storing grouper: ",key
            store.get_storer(key).attrs.grouper = groupers[key]

def read_samples(file,*params,**kwparams):
    store = pd.HDFStore(file,*params,**kwparams)
    samples = {}
    formulae = {}
    groupers = {}
    for key in store.keys():
        samples[key] = store[key]
        if hasattr(store.get_storer(key).attrs,'formula'):
            formulae[key] = read_formula(store,key)
        if hasattr(store.get_storer(key).attrs,'grouper'):
            groupers[key] = store.get_storer(key).attrs.grouper

    return samples,formulae,groupers
