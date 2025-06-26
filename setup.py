# setup.py
from setuptools import setup, find_packages

setup(
    name="contextual_insight",
    version="0.1.0",
    author="M2hesh",
    url="https://github.com/M2hesh/Contextual-Insight-Generator",
    description="Colab-friendly clustering wizard with business prompts",
    packages=find_packages(),            # THIS must find your 'contextual_insight' folder
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "gdown",
    ],
    python_requires=">=3.7",
)
