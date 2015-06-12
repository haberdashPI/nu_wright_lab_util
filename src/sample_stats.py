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


def contrast_table(samples,names=None,round=3):
    if names is None: data = pd.DataFrame(samples,columns=names)
    else: data = pd.DataFrame(samples,columns=names)

    comp_data = pd.DataFrame()
    for i in range(data.shape[1]-1):
        for j in range(i+1,data.shape[1]):
            comp_data[data.columns[i]+' - '+data.columns[j]] = \
              data.iloc[:,i] - data.iloc[:,j]

    return comp_data.apply(lambda xs: coef_stats(xs,round)).T


def coef_stats(xs,round=3):
    return pd.Series(np.array([np.around(np.mean(xs),round),
                               np.around(np.percentile(xs,02.5),round),
                               np.around(np.percentile(xs,97.5),round),
                               np.around(p_value(xs),round),
                               sig_indicator(p_value(xs))]),
                     index=['mean','lower','upper','p_value','sig'],
                     name='stats')


def normal_error_fn(error):
    def fn(y_hat,i):
        return np.random.normal(scale=error[i],size=y_hat.shape[0])
    return fn


def ppp(y,y_hats,error_fn,stats,N=1000):
    def test(y,y_hat,error_fn,i):
        diffs = y_hat - y
        fake_diffs = error_fn(y_hat,i)

        return pd.DataFrame([{'real': stat(diffs),'fake': stat(fake_diffs),
                              'type': name}
                             for name, stat in stats.iteritems()])

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
