from setuptools import setup

 setup(
   name='opentraj',
   version='1.0',
   author='Javad Amirian',
   author_email='amiryan.j@gmail.com',
   packages=['opentraj'],
   scripts=['bin/script1','bin/script2'],
   url='https://github.com/crowdbotp/OpenTraj',
   license='MIT',
   description='Tools for analyzing trajectory datasets',
   long_description=open('README.md').read(),
   install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "tqdm",
        "pykalman", 
        "PyYAML",       
   ],
   extras_require={
        'test': [
            "pylint",
            "pytest",
        ],
        'plot': [
            "matplotlib",
            "seaborn",
        ]
   }
)
