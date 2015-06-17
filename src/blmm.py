import h5py
import os.path
import pandas as pd
import pickle
import pystan
import numpy as np
import patsy


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


def load_model(prefix):
    model_file = prefix+".stan"
    object_file = prefix+".o"
    if not os.path.isfile(object_file):
        model = pystan.StanModel(model_file)
        with open(object_file,'wb') as f: pickle.dump(model,f)
    else:
        with open(object_file,'rb') as f: model = pickle.load(f)

    return model
