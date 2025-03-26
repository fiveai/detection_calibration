from setuptools import find_packages
from distutils.core import setup
import os

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''


with open(os.path.join(current_directory, "requirements.txt")) as fh:
    requirements = fh.readlines()

setup(
    name='detection_calibration',
    package_dir={"": "src"},
    packages=find_packages('src'),
    version='0.0.1',
    license='',
    description='detection_calibration library for post-hoc calibration and reliability benchmarking of object detectors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Selim Kuzucu',
    author_email='selim686kuzucu@gmail.com',
    url='https://github.com/fiveai',
    download_url='https://github.com/fiveai/detection_calibration',
    keywords=[
            'Calibration',
            'Object Detection',
            'Performance Evaluation'
    ],
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: POSIX :: Linux",
    ],
)
