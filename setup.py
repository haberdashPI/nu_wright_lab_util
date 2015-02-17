from  setuptools  import  setup
from  setuptools.command.install import install as _install
import pickle, os, datetime

class pystan_compile(_install):
    def run(self):
        import pystan

        if os.stat('src/stan/mean.model.stan').st_mtime > \
            os.stat('src/stan/mean.model.o').st_mtime:

            model = pystan.StanModel(file='src/stan/mean.model.stan')
            with open('src/stan/mean.model.o','wb') as f: pickle.dump(model,f)
        
        if os.stat('src/stan/ind.model.stan').st_mtime > \
            os.stat('src/stan/ind.model.o').st_mtime:

            model = pystan.StanModel(file='src/stan/ind.model.stan')
            with open('src/stan/ind.model.o','wb') as f: pickle.dump(model,f)

        _install.run(self)

setup(cmdclass = {'install': pystan_compile},
      name='pylab_util',version='0.0.2',author='David Little',
      packages=['pylab_util'],
      package_dir={'pylab_util': 'src'},
      setup_requires = ['pystan>=2.5.0'],
      package_data={'pylab_util': ['stan/*.o']},
      requires=['statsmodels(>=0.5.0)',
                'pystan(>=2.5.0)',
                'patsy(>=0.3.0)',
                'numpy','pandas'])
