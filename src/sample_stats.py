import numpy as np
import pandas as pd

def p_value(xs): return 2*np.mean(np.sign(xs) != np.sign(np.mean(xs)))

def ppp(y,y_hats,error,stats,N=1000):
    def test(y,y_hat,error):
        diffs = y_hat - y
        fake_diffs = np.random.normal(scale=error,size=len(y_hat))

        return pd.DataFrame([{'real': stat(diffs),'fake': stat(fake_diffs),
                              'type': name}
                              for name, stat in stats.iteritems()])

    if N is None:
        return pd.concat([test(y,y_hats[i,:],error[i])
                          for i in range(len(error))])
    else:
        samples = np.random.randint(y_hats.shape[0],size=N)
        return pd.concat([test(y,y_hats[i,:],error[i]) for i in samples])
