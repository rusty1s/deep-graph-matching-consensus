from setuptools import setup, find_packages

__version__ = '1.0.0'
url = 'https://github.com/rusty1s/deep-graph-matching-consensus'

install_requires = []
setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov']

setup(
    name='dgmc',
    version=__version__,
    description='Implementation of Deep Graph Matching Consensus in PyTorch',
    author='Matthias Fey',
    author_email='matthias.fey@tu-dortmund.de',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'geometric-deep-learning',
        'graph-neural-networks',
        'graph-matching',
        'neighborhood-consensus',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
