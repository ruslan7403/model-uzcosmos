from setuptools import setup, find_packages

setup(
    name="traffic-sign-recognition",
    version="1.0.0",
    description="FaceID-like Traffic Sign Recognition System using embedding similarity",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
)
