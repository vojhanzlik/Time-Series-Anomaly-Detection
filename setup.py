from setuptools import setup, find_packages


setup(
    name="anomaly-det-model",
    version="1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Anomaly detection in time series model",
    long_description="Long Description",
    packages=find_packages(),
    url="https://github.com/vojhanzlik/Time-Series-Anomaly-Detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'numpy',
        'pandas',
        'tslearn',
        'scipy',
        'scikit-learn',
        'h5py'
    ],
)
