"""Setup script for flow-based generative models package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flow-based-generative-models",
    version="0.1.0",
    author="AI Projects",
    author_email="ai@example.com",
    description="Flow-based generative models implementation with RealNVP and Glow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/flow-based-generative-models",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "ruff>=0.0.280",
            "pytest>=7.4.0",
            "pre-commit>=3.3.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flow-train=scripts.train:main",
            "flow-sample=scripts.sample:main",
            "flow-evaluate=scripts.evaluate:main",
        ],
    },
)
