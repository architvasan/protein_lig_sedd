from setuptools import setup, find_packages

setup(
    name="protlig_dd",  # Replace with your package name
    version="0.1.0",           # Initial version
    author="Archit Vasan",        # Your name
    author_email="avasan@anl.gov",  # Your email
    description="Package to train protein-ligand disc diff models",  # Short package description
    long_description=open("README.md").read(),         # Detailed description from README
    long_description_content_type="text/markdown",     # Specify the format of the README
    packages=find_packages(),                          # Automatically find and include all packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',                           # Specify compatible Python versions
)

