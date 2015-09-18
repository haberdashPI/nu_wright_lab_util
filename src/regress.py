import collections
import os
import patsy
import pandas as pd
import numpy as np
import scipy
import pystan

import blmm
from sample_stats import *

linear_model = blmm.load_model('linear',use_package_cache=True)
robit_model = blmm.load_model('robit',use_package_cache=True)
robit2_model = blmm.load_model('robit2',use_package_cache=True)

default_stats = collections.OrderedDict()
default_stats['min'] = min
default_stats['min95'] = lambda x: np.percentile(x, 02.5)
default_stats['min68'] = lambda x: np.percentile(x, 34.1)
default_stats['rms'] = lambda x: np.sqrt(np.mean(x**2))
default_stats['max68'] = lambda x: np.percentile(x, 65.9)
default_stats['max95'] = lambda x: np.percentile(x, 97.5)
default_stats['max'] = max
default_stats['skewness'] = lambda x: scipy.stats.skew(x)
default_stats['kurtosis'] = lambda x: scipy.stats.kurtosis(x)


def linear(formula,df,coef_prior=10,error_prior=100,cache_file=None,
           **sample_kws):
  y,A = patsy.dmatrices(formula,df,return_type='dataframe',eval_env=1)
  y = np.squeeze(y)

  if cache_file is None or not os.path.isfile(cache_file):
    fit = linear_model.sampling({'y': y, 'A': A, 'n': A.shape[0],
                                 'k': A.shape[1],
                                 'fixed_mean_prior': coef_prior,
                                 'prediction_error_prior': error_prior},
                                **sample_kws)
    if cache_file:
      blmm.write_samples(fit.extract(),cache_file)
  else:
    fit = blmm.read_samples(cache_file)

  return Linear(fit,y,A,df)


class BaseRegressResults(object):
  def __init__(self,fit,y,A,df):
    self.fit = fit
    self.y = y
    self.A = A
    self.df = df

  def cache(self,cache_file):
    if issubclass(self.fit,pystan.StanFit4Model):
      blmm.write_samples(self.fit.extract(),cache_file)
    else: blmm.write_samples(fit,cache_file)

  def summary(self,coefs=None):
    if coefs is None:
      return coef_table(self.fit['alpha'],self.A.columns)
    else:
      return coef_table(self.fit['alpha'][:,coefs],self.A.columns[coefs])

  def contrasts(self,coefs=None,correct=True):
    if coefs is None:
      return contrast_table(self.fit['alpha'],self.A.columns,correct=correct)
    else:
      return contrast_table(self.fit['alpha'][:,coefs],self.A.columns[coefs],
                            correct=correct)

  def predict(self,df=None):
    if df is None: return self._predict_helper(self.A)
    else:
      A = patsy.dmatrix(self.A.design_info,df)
      return self._predict_helper(A)

  def linear_tests(self,names,X,coefs=None,rhs=0,correct=True):
    if coefs is None: coefs = slice(0,self.fit['alpha'].shape[1])
    tests = np.dot(self.fit['alpha'][:,coefs],X.T) - rhs
    table = coef_table(tests,names)
    if correct:
      return mcorrect(pd.DataFrame(tests),table)
    else:
      return table

  def validate(self,stats=default_stats):
    p = self.predict()
    tests = ppp(self.y,p.T,self.error_fn(),stats)

    g = tests.groupby('type')
    summary = g.mean()
    summary['p_val'] = g.apply(lambda d: p_value(d.real - d.fake))
    return summary

  def WAIC(self):
    log_post = self.log_posterior(self.y)
    p_waic = np.std(log_post,axis=1,ddof=1)
    lpd = scipy.misc.logsumexp(log_post,axis=1) - np.log(log_post.shape[1])

    waic = -2*np.sum(lpd - p_waic)
    sd = 2*np.sqrt(lpd.shape[0] * np.std(lpd - p_waic))
    return waic,sd,np.sum(p_waic)


class Linear(BaseRegressResults):
  def _predict_helper(self,A):
    p = np.einsum('ij,kj->ik',A,self.fit['alpha'])
    return p

  def log_posterior(self,y):
    eps = self.fit['eps'][np.newaxis,:]
    y = y[:,np.newaxis]
    p = self.predict()

    return scipy.stats.norm.pdf(y,p,eps)

  def error_fn(self):
    error = self.fit['eps']
    def fn(y_hat,i,error=error):
        return np.random.normal(scale=error[i],size=y_hat.shape[0])
    return fn


def robit(formula,df,coef_prior=5,error_prior=100,r=1e-10,cache_file=None,
          **sample_kws):

  y,A = patsy.dmatrices(formula,df,return_type='dataframe',eval_env=1)
  y = np.squeeze(y)

  if cache_file is None or not os.path.isfile(cache_file):
    fit = robit_model.sampling({'y': y, 'A': A, 'n': A.shape[0],
                                'k': A.shape[1], 'fixed_mean_prior': coef_prior,
                                'r': r,
                                'prediction_error_prior': error_prior},
                               **sample_kws)
    if cache_file:
      blmm.write_samples(fit.extract(),cache_file)
  else:
    fit = blmm.read_samples(cache_file)

  return RobustLogit(r,fit,y,A,df)


class RobustLogit(BaseRegressResults):
  def __init__(self,r,*params):
    super(RobustLogit,self).__init__(*params)
    self.r = r

  def _predict_helper(self,A):
    p = np.einsum('ij,kj->ik',A,self.fit['alpha'])
    p = 1 / (1 + np.exp(-p))
    p = (p - self.r/2) / (1-self.r)

    return p

  def log_posterior(self,y):
    scale = self.fit['scale'][np.newaxis,:]

    r = self.r
    y = y[:,np.newaxis]
    p = r/2 + self.predict()*(1-r)

    return scipy.stats.beta.logpdf(r/2 + y*(1-r),p*scale,(1-p)*scale)

  def error_fn(self):
    scale = self.fit['scale']
    r = self.r

    def fn(y_hat,i,scale=scale,r=r):
      p = r/2 + y_hat*(1-r)
      pr = np.random.beta(p*scale[i],(1-p)*scale[i])
      pr = (pr - r/2) / (1-r)

      return pr - y_hat

    return fn


def robit2(formula,df,coef_prior=5,error_prior=100,cache_file=None,r_prior=0.05,
           **sample_kws):

  y,A = patsy.dmatrices(formula,df,return_type='dataframe',eval_env=1)
  y = np.squeeze(y)

  if cache_file is None or not os.path.isfile(cache_file):
    fit = robit2_model.sampling({'y': y, 'A': A, 'n': A.shape[0],
                                 'k': A.shape[1],
                                 'fixed_mean_prior': coef_prior,
                                 'r_prior': r_prior,
                                 'prediction_error_prior': error_prior},
                                **sample_kws)
    if cache_file:
      blmm.write_samples(fit.extract(),cache_file)
  else:
    fit = blmm.read_samples(cache_file)

  return RobustLogit2(fit,y,A,df)


class RobustLogit2(BaseRegressResults):
  def __init__(self,*params):
    super(RobustLogit2,self).__init__(*params)

  def _predict_helper(self,A):
    p = np.einsum('ij,kj->ik',A,self.fit['alpha'])
    r = self.fit['r']
    p = 1 / (1 + np.exp(-p))
    p = (p - r/2) / (1-r)

    return p

  def log_posterior(self,y):
    scale = self.fit['scale'][np.newaxis,:]
    r = self.fit['r'][np.newaxis,:]

    y = y[:,np.newaxis]
    p = r/2 + self.predict()*(1-r)

    return scipy.stats.beta.logpdf(r/2 + y*(1-r),p*scale,(1-p)*scale)

  def error_fn(self):
    scale = self.fit['scale']
    r = self.fit['r']

    def fn(y_hat,i,scale=scale,r=r):
      p = r[i]/2 + y_hat*(1-r[i])
      pr = np.random.beta(p*scale[i],(1-p)*scale[i])
      pr = (pr - r[i]/2) / (1-r[i])

      return pr - y_hat

    return fn
