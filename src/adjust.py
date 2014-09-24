from statsmodels.formula.api import ols

def covariate_adjust(df,x,y,covariate,by=[],prefix='adj_'):
    all_by = by + [x]
    slopes = df.groupby(all_by).\
          apply(lambda x: ols(y+' ~ '+covariate,x).fit().params[1])
    slopes.name = prefix + 'slope_' + y
    df = df.join(slopes,on=all_by)

    if len(by) > 0:
        df[prefix + y] = df[y] - df[slopes.name] * \
            (df.set_index(by)[covariate] -
             df.groupby(by)[covariate].mean()).reset_index(drop=True)
    else:
        df[prefix + y] = df[y] - \
            df[slopes.name] * (df[covariate] - df[covariate].mean())

    del df[slopes.name]

    return df
