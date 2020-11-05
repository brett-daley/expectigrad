import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="expectigrad",
    version="0.0.0",
    author="Brett Daley",
    author_email="b.daley@northeastern.edu",
    description="Expectigrad optimizer for Pytorch and TensorFlow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brett-daley/expectigrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
