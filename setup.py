from setuptools import setup, find_packages
print (find_packages())
setup(
    name='fast-higashi',
    version='0.0.1a0',
    description='Fast-Higashi: Ultrafast and interpretable single-cell 3D genome analysis',
    url='https://github.com/ma-compbio/Fast-Higashi',
    include_package_data=True,
    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=[],
    extras_require={},
    author='Ruochi Zhang',
    author_email='ruochiz@andrew.cmu.edu',
    license='MIT'
)
