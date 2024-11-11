from setuptools import setup, find_packages

setup(
    name='l2augment',
    version='0.1',    
    description='Code for learning augmentation pattern for test-time adaptation method',
    url='https://github.com/robflynnyh/learning-to-augment',
    author='Rob Flynn',
    author_email='rjflynn2@sheffield.ac.uk',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['torch',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',   
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.8',
    ],
)