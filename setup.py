import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="activeeval",
    version="0.1.0",
    author="Neil Marchant",
    author_email="ngmarchant@gmail.com",
    description="A package for active evaluation of classifiers",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ngmarchant/activeeval",
    install_requires=["numpy>=1.16", "scipy>=1.0.0", "treelib>=1.6.1"],
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
