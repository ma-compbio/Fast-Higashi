# from setuptools import setup, find_packages
# print (find_packages())
# setup(
#     name='fast-higashi',
#     version='0.0.1a0',
#     description='Fast-Higashi: Ultrafast and interpretable single-cell 3D genome analysis',
#     url='https://github.com/ma-compbio/Fast-Higashi',
#     include_package_data=True,
#     python_requires='>=3.9',
#     packages=find_packages(),
#     install_requires=[
#         'numpy>=1.21.2',
#         'scipy==1.7.3',
#         'pandas==1.3.4',
#         'cython>=0.29.24',
#         'torch>=1.8.0',
#         'scikit-learn>=0.23.2',
#         'tqdm',
#         'h5py',
#         'seaborn>=0.11.2',
#         'umap-learn>=0.5',
#         'opt_einsum',
#         'pybedtools',
#         'pytorch',
#         'psutil'
#     ],
#     extras_require={},
#     author='Ruochi Zhang',
#     author_email='ruochiz@andrew.cmu.edu',
#     license='MIT'
# )

import setuptools

if __name__ == "__main__":
    setuptools.setup(name="fasthigashi")
