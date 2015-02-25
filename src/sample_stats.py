import numpy as np
import pandas as pd

def p_value(xs): return 2*np.mean(np.sign(xs) != np.sign(np.mean(xs)))

def ppp(y,model,stats,N=1000):
    def test(y,y_hat):
        diffs = y_hat - y
        fake_diffs = np.random.normal(scale=model.error[y_hat.name],
                                      size=len(y_hat))

        return pd.DataFrame([{'real': stat(diffs),
                              'fake': stat(fake_diffs),
                              'type': name}
                             for name, stat in stats.iteritems()])

    samples = np.random.randint(model.y_hat.shape[0],size=N)
    return pd.concat([test(y,model.y_hat.iloc[i,:]) for i in samples])

