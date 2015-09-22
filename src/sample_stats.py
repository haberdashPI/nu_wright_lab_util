import scipy
import collections
import numpy as np
import pandas as pd


def p_value(xs):
    xs = xs[~np.isnan(xs)]
    return min(1.0,2*np.mean(np.sign(xs) != np.sign(np.median(xs))))


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


def coef_table(samples,names=None,round=3):
    if names is None: data = pd.DataFrame(samples,columns=names)
    else: data = pd.DataFrame(samples,columns=names)
    return data.apply(lambda xs: coef_stats(xs,round)).T


def contrast_table(samples,names=None,round=3,correct=True):
    if names is None: data = pd.DataFrame(samples,columns=names)
    else: data = pd.DataFrame(samples,columns=names)

    columns = data.columns.tolist()

    comp_data = pd.DataFrame()
    for i in range(data.shape[1]-1):
        for j in range(i+1,data.shape[1]):
            comp_data[str(columns[i])+' - '+str(columns[j])] = \
              data.iloc[:,i] - data.iloc[:,j]

    if correct:
      stats = comp_data.apply(lambda xs: coef_stats(xs,round)).T
      return mcorrect(comp_data,stats,round)
    else:
      return comp_data.apply(lambda xs: coef_stats(xs,round)).T


# correct for multiple comparisons by looking at the joint
# probability of all comparisons with a lower p-value than each
# given comparison.
def mcorrect(samples,stats,round=3):
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


def coef_stats(xs,round=3):
    return pd.Series(np.array([np.around(np.mean(xs),round),
                               np.around(np.std(xs),round),
                               np.around(np.percentile(xs,02.5),round),
                               np.around(np.percentile(xs,97.5),round),
                               np.around(p_value(xs),round),
                               sig_indicator(p_value(xs))]),
                     index=['mean','SE','lower','upper','p_value','sig'],
                     name='stats')


def normal_error_fn(error):
    def fn(y_hat,i):
        return np.random.normal(scale=error[i],size=y_hat.shape[0])
    return fn

def stat_fn(name,fn):
  return lambda x: pd.DataFrame([{'type': name, 'value': fn(x)}])

default_stats = [stat_fn('min',min),
                 stat_fn('max',max),
                 stat_fn('min95', lambda x: np.percentile(x, 02.5)),
                 stat_fn('min68', lambda x: np.percentile(x, 34.1)),
                 stat_fn('rms', lambda x: np.sqrt(np.mean(x**2))),
                 stat_fn('max68', lambda x: np.percentile(x, 65.9)),
                 stat_fn('max95', lambda x: np.percentile(x, 97.5)),
                 stat_fn('skewness', lambda x: scipy.stats.skew(x)),
                 stat_fn('kurtosis', lambda x: scipy.stats.kurtosis(x))]


def ppp(y,y_hats,error_fn,stats=default_stats,N=1000):
    def test(y,y_hat,error_fn,i):
        diffs = y - y_hat
        fake_diffs = error_fn(y_hat,i)

        results = []
        for stat in stats:
          real = stat(diffs)
          fake = stat(fake_diffs)
          results.append(pd.DataFrame({'real': real['value'],
                                       'fake': fake['value'],
                                       'type': real['type']}))
        return pd.concat(results)

    if N is None:
        return pd.concat([test(y,y_hats[i,:],error_fn,i)
                          for i in range(y_hats.shape[0])])
    else:
        samples = np.random.randint(y_hats.shape[0],size=N)
        return pd.concat([test(y,y_hats[i,:],error_fn,i) for i in samples])


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
