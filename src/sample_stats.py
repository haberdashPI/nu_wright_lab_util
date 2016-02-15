import scipy
import numpy as np
import pandas as pd
from pymc.utils import hpd


def p_value(xs,axis=None):
  diffs = np.sign(xs) != np.sign(np.nanmedian(xs,axis=axis))
  return np.minimum(1.0,2*np.mean(diffs,axis=axis))


def sig_indicator(p_val):
    if p_val < 0.001:
         return '***'
    elif p_val <= 0.01:
        return ' **'
    elif p_val <= 0.05:
        return '  *'
    elif p_val <= 0.1:
        return '  ~'
    else:
        return '   '


def coef_table(samples,names=None,round=3,alpha=0.318,sig=True):  # 0.318 = 1 SEM
    if names is None: data = pd.DataFrame(samples,columns=names)
    else: data = pd.DataFrame(samples,columns=names)

    if sig:
      return data.apply(lambda xs: coef_stats(xs,round,alpha)).T
    else:
      return data.apply(lambda xs: coef_stats_ns(xs,round,alpha)).T


def contrast_table(samples,names=None,round=3,correct=True,alpha=0.05):
    if names is None: data = pd.DataFrame(samples,columns=names)
    else: data = pd.DataFrame(samples,columns=names)

    columns = data.columns.tolist()

    comp_data = pd.DataFrame()
    for i in range(data.shape[1]-1):
        for j in range(i+1,data.shape[1]):
            comp_data[str(columns[i])+' - '+str(columns[j])] = \
              data.iloc[:,i] - data.iloc[:,j]

    if correct:
      stats = comp_data.apply(lambda xs: coef_stats(xs,round,alpha)).T
      return mcorrect(comp_data,stats,round)
    else:
      return comp_data.apply(lambda xs: coef_stats(xs,round,alpha)).T


# correct for multiple comparisons by looking at the joint
# probability of all comparisons with a lower p-value than each
# given comparison.
def mcorrect(samples,stats,round=3):
  if stats.shape[1] <= 2:
    return stats
   
  order = np.argsort(stats.p_value)
  final_p_vals = np.zeros(len(order))

  S = samples.values[:,order]
  signs = np.sign(np.median(S,axis=0))
  sign_changes = S * -signs[np.newaxis,:]

  for i,j in enumerate(order):
    sign_change_freq = np.mean(np.any(sign_changes[:,:(i+1)] > 0,axis=1))
    final_p_vals[j] = np.around(min(1.0,2*sign_change_freq),3)

  stats.p_value = final_p_vals
  stats.sig = stats.p_value.apply(sig_indicator)

  return stats


def coef_stats_ns(xs,round=3,alpha=0.318):
  return coef_stats(xs,round,alpha,show_sig=False)


def coef_stats(xs,round=3,alpha=0.318,show_sig=True):  # 0.318 = 1 SEM
    lower, upper = np.around(hpd(xs,alpha),round)

    if not show_sig:
      return pd.Series(np.array([np.around(np.mean(xs),round),
                                 np.around(np.std(xs),round),
                                 np.around(scipy.stats.sem(xs),round),
                                 lower,upper,
                                 np.around(p_value(xs),round)]),
                       index=['mean','SE','error','lower','upper','p_value'],
                       name='stats')
    else:
      return pd.Series(np.array([np.around(np.mean(xs),round),
                                 np.around(np.std(xs),round),
                                 np.around(scipy.stats.sem(xs),round),
                                 lower,upper,
                                 np.around(p_value(xs),round),
                                 sig_indicator(p_value(xs))]),
                       index=['mean','SE','error','lower','upper','p_value','sig'],
                       name='stats')




def normal_error_fn(error):
    def fn(y_hat,indices):
        return np.random.normal(scale=error[indices,np.newaxis],
                                size=(len(indices),y_hat.shape[1]))
    return fn


def stat_fn(name,fn):
  return lambda x,a: pd.DataFrame({'type': name, 'value': fn(x,a)})

default_stats = [stat_fn('min',lambda x,a: np.min(x,axis=a)),
                 stat_fn('max',lambda x,a: np.max(x,axis=a)),
                 stat_fn('min95', lambda x,a: np.percentile(x, 02.5,axis=a)),
                 stat_fn('min68', lambda x,a: np.percentile(x, 34.1,axis=a)),
                 stat_fn('median', lambda x,a: np.percentile(x,50,axis=a)),
                 stat_fn('rms', lambda x,a: np.sqrt(np.mean(x**2,axis=a))),
                 stat_fn('max68', lambda x,a: np.percentile(x, 65.9,axis=a)),
                 stat_fn('max95', lambda x,a: np.percentile(x, 97.5,axis=a)),
                 stat_fn('skewness', lambda x,a: scipy.stats.skew(x,axis=a)),
                   stat_fn('kurtosis', lambda x,a: scipy.stats.kurtosis(x,axis=a))]


def ppp_T(T_real,T_fake,stats):
  results = []
  for stat in stats:
    real = stat(T_real,1)
    fake = stat(T_fake,1)
    results.append(pd.DataFrame({'real': real.value, 'fake': fake.value,
                                 'type': real.type}))
  return pd.concat(results)

def ppp(y,y_hat,error_fn,stats=default_stats,N=1000):
  if N is None:
    indices = np.arange(y_hat.shape[0])
  else:
    indices = np.random.randint(y_hat.shape[0],size=N)

  diffs = y[np.newaxis,:]-y_hat[indices,:]
  fake_diffs = error_fn(y_hat,indices)

  return ppp_T(diffs,fake_diffs,stats)

def bootstrap_samples(stat_fn,bootstrap=1000):
    if isinstance(bootstrap,(np.ndarray,np.matrix)):
        bootstrap_weights = bootstrap
        bootstrap = bootstrap_weights.shape[1]
    else:
        N = stat_fn.boot_size
        bootstrap_weights = np.random.dirichlet(np.ones(N),size=bootstrap) * N

    bootstrap_stat = np.array([np.sum(stat_fn(bootstrap_weights[i,:]))
                               for i in range(bootstrap)])
    return bootstrap_stat,bootstrap_weights


def ci_boot(stat_fn,bootstrap=1000):
    bootstrap_stat,_ = bootstrap_samples(stat_fn,bootstrap)
    return (np.mean(bootstrap_stat),
            np.percentile(bootstrap_stat,[97.2,2.5]).tolist())


def ci_normal(stat_fn):
    xs = stat_fn()
    se = np.sqrt(len(xs) * np.std(xs,ddof=1))
    mean = np.sum(xs)
    return mean,[mean-2*se,mean+2*se]
