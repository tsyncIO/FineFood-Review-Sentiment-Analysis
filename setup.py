from setuptools import setup, find_packages

setup(
    name="emotion-recognition-affectnet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch==1.10.0",
        "torchvision==0.11.1",
        "matplotlib==3.5.0",
        "opencv-python==4.5.4.60",
        "numpy==1.21.4",
        "pandas==1.3.3",
        "scikit-learn==0.24.2"
    ],
    entry_points={
        'console_scripts': [
            'emotion-recognition=src.predict:main',
        ],
    },
)
