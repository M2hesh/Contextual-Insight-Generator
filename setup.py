from setuptools import setup, find_packages

setup(
    name="contextual_insight",
    version="0.1.0",
    author="Your Name",
    url="https://github.com/yourusername/Contextual-Insight-Generator",
    description="Lightweight Colab wizard for clustering + business prompts",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "gdown",
    ],
    python_requires=">=3.7",
)
