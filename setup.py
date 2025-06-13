from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lnp_mod",
    version="0.1.0",
    author="Owen Ip",
    author_email="ipowen10@gmail.com",
    description="LNP-MOD: Lipid Nanoparticle Morphology Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/owenip/LNP_MOD",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lnp-mod=lnp_mod.core.inference:main",
        ],
    },
    extras_require={
        "dev": [line.strip() 
                for line in open("requirements-dev.txt")
                if line.strip() and not line.startswith("#")]
    },
)