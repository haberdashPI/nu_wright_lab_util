from distutils.core import setup

setup(name='pylab_util',version='0.0.1',author='David Little',
      packages=['pylab_util'],
      package_dir={'pylab_util': 'src'},
      requires=['statsmodels(>=0.5.0)'])
