from setuptools import find_packages, setup

setup(
    name="cytotraj",
    version="0.0.1",
    author="Erik Serrano",
    author_email="erik.serrano@cuanschutz.edu",
    description="CytoTrajectory: Drug Screening Trajectory Tracking with\
                 Image-based Profile Interpretability using Variational Auto\
                 Encoders (VAEs)",
    url="https://github.com/WayScience/CytoTrajectory",
    packages=find_packages(),
    python_requires=">=3.10, <3.12",
)
