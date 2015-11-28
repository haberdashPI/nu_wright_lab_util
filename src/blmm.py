import pkgutil
import appdirs
import h5py
import os.path
import pandas as pd
import pickle
import pystan
import numpy as np
import patsy
from misc import unique_rows

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


def dmatrix(df,extracted,eval_env=0):
    if not extracted: extracted = ''

    formula = saveable_formula(extracted,df,eval_env+1)
    X = formula.dmatrix(df)
    return formula,X


def dmatrices(df,extracted,eval_env=0):
    if not extracted: extracted = ''

    formula = saveable_formulas(extracted,df,eval_env+1)
    y,X = formula.dmatrices(df)
    return formula,np.squeeze(y),X


def _organize_group_labels(grouped):
    group_labels = np.array(grouped.grouper.labels).T
    unique_labels = unique_rows(group_labels)
    group_index_map = {tuple(labels): i
                       for i,labels in enumerate(unique_labels)}

    group_indices = np.array([group_index_map[tuple(labels)]
                              for labels in group_labels])

    group_keys = pd.DataFrame([{grouped.grouper.names[j]:
                                grouped.grouper.levels[j][labels[j]]
                                for j in range(len(labels))}
                               for labels in unique_labels])
    return group_indices,group_keys


def setup_groups(df,groups):
    group_df = df.groupby(groups)
    group_indices,group_keys = _organize_group_labels(group_df)

    return group_df.head(1),group_indices,group_keys


def write_samples(samples,file,*params,**kwparams):
    with h5py.File(file,'w',*params,**kwparams) as store:
        for key in samples.keys():
            store[key] = samples[key]


def read_samples(file,*params,**kwparams):
    samples = {}
    with h5py.File(file,'r',*params,**kwparams) as store:
        for key in store.keys():
            x = np.zeros(store[key].shape,dtype=store[key].dtype)
            store[key].read_direct(x)
            samples[key] = x

    return samples


def load_model(prefix,use_package_cache=False,nocache=False):
    if nocache:
        clear_cache(prefix,use_package_cache)
    if use_package_cache:
        cache_dir = appdirs.user_cache_dir("pylab_util","David Little")
        object_file = os.path.join(cache_dir,prefix+".o")
        model_code = pkgutil.get_data('pylab_util','stan/'+prefix+'.stan')
    else:
        model_file = prefix+".stan"
        object_file = prefix+".o"

    if not os.path.isfile(object_file):
        print ("WARNING: Saving cached model to "+object_file+" if you have"+
               " trouble after changing model code, you may need to delete "+
               "this file.")
        if use_package_cache:
            model = pystan.StanModel(model_code=model_code.decode())
        else:
            model = pystan.StanModel(model_file)
        with open(object_file,'wb') as f: pickle.dump(model,f)
    else:
        with open(object_file,'rb') as f: model = pickle.load(f)

    return model


def clear_cache(prefix,use_package_cache=False):
    if use_package_cache:
        cache_dir = appdirs.user_cache_dir("pylab_util","David Little")
        os.remove(os.path.join(cache_dir,prefix+".o"))
    else:
        os.remove(os.path.join(prefix+".o"))
