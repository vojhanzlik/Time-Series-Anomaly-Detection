Distributing the package: 

pip install setuptools, twine

python setup.py bdist_wheel sdist

Local pip install:

pip install .


Upload to test-Pypi

twine check dist/*

twine upload -r testpypi dist/*