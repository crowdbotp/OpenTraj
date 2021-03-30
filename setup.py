from setuptools import setup

 setup(
   name='opentraj',
   version='0.1.0',
   author='Javad Amirian',
   author_email='amiryan.j@gmail.com',
   packages=['package_name', 'package_name.test'],
   scripts=['bin/script1','bin/script2'],
   url='https://github.com/crowdbotp/OpenTraj',
   license='MIT',
   description='Tools for working with trajectory datasets',
   long_description=open('README.txt').read(),
   install_requires=[
       "pytest",
   ],
)
