from distutils.core import setup
import pickle, os, datetime


setup(name='pylab_util',version='0.0.5',author='David Little',
      packages=['pylab_util'],
      package_dir={'pylab_util': 'src'},
      package_data={'pylab_util': ['stan/*.stan']},
      requires=['statsmodels(>=0.5.0)',
                'pystan(>=2.7.0)',
                'patsy(>=0.3.0)','matplotlib',
                'numpy','pandas','appdirs'])
