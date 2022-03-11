# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(name='Feature-CRF-AE',
      version='0.0.1',
      author='Houquan Zhou',
      author_email='Jacob_Zhou@outlook.com',
      description='Feature CRF-AE Model',
      long_description=open('README.md', 'r').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/Jacob-Zhou/FeatureCRFAE',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Text Processing :: Linguistic'
      ],
      setup_requires=[
          'setuptools>=18.0',
      ],
      install_requires=[
          'h5py==3.1.0', 'matplotlib==3.3.1', 'nltk==3.5', 'numpy==1.19.1',
          'overrides==3.1.0', 'scikit_learn==1.0.2', 'seaborn==0.11.0',
          'torch==1.6.0', 'tqdm==4.49.0', 'transformers==3.5.1',
          'allennlp==1.2.2'
      ],
      entry_points={
          'console_scripts': [
              'crf-ae=tagger.cmds.crf_ae:main',
              'feature-hmm=tagger.cmds.feature_hmm:main',
          ]
      },
      python_requires='>=3.7',
      zip_safe=False)
