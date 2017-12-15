import blmm
from misc import stdout_redirected
import numpy as np
import scipy

model = blmm.load_model('psych_curve',use_package_cache=True)



# dbxs = [ 6.  ,  6.5 ,  6.5 ,  6.5 ,  6.  ,  6.  ,  6.5 ,  6.5 ,  7.  ,
#         7.  ,  7.5 ,  7.5 ,  7.5 ,  7.25,  7.25,  7.25,  7.  ,  7.25,
#         7.25,  7.25,  7.5 ,  7.5 ,  7.5 ,  7.75,  7.75,  7.75,  8.  ,
#         8.25,  8.25,  8.25,  8.5 ,  8.5 ,  8.75,  8.75,  8.75,  8.5 ,
#         8.5 ,  8.75,  9.  ,  9.  ,  9.  ,  8.75,  8.75,  8.75,  8.5 ,
#         8.5 ,  8.5 ,  8.25,  8.25,  8.25]


def levitt(log_delta,reversals=7,drop=3):
  dirs = np.sign(np.diff(log_delta))
  indices = np.where(dirs != 0)[0]
  changes = np.diff(dirs[indices])
  xs = log_delta[indices[np.where(changes != 0)[0]+1]]

  if len(xs) < reversals:
    return {'thresh': float('NaN'), 'thresh_se': float('NaN')}

  if len(xs) % 2 == 0: xs = xs[(drop+1):len(xs)]
  else: xs = xs[drop:len(xs)]

  return {'thresh': np.mean(xs), 'thresh_se': np.std(xs)}


def psych_curve(log_delta,correct,N,theta_range,sigma,miss,threshold=0.79,
                fast=False):
  with stdout_redirected():
        #print data[['sid','block','day','condition']].head(1)
        sample_input = {"n": len(log_delta),
                        "log_delta": log_delta,
                        "tmin": theta_range[0],
                        "tmax": theta_range[1],
                        "sigma": sigma,
                        "miss": miss,
                        "correct": correct.astype('int64'),
                        "totals": N.astype('int64')}

        ixs = range((len(log_delta)/2),(len(log_delta)))
        init = np.mean(log_delta[ixs])
        def init_fn():
            return {"theta": init * (1 + np.random.uniform(high=0.01))}

        if fast:
          fit = model.optimizing(data=sample_input,init=init_fn)
        else:
          fit = model.sampling(data=sample_input,init=init_fn,iter=2000,n_jobs=1)

  theta = fit['theta']
  dprime = scipy.stats.norm.ppf((threshold-miss/2)/(1-miss))
  thresh_off = -np.log(dprime)/sigma

  if fast:
    return {'thresh': theta + thresh_off, 'thresh_se': float('NaN')}
  else:
    thresh = (theta + thresh_off).mean()
    thresh_sd = (theta + thresh_off).std()

    return {'thresh': thresh, 'thresh_se': thresh_sd}
    