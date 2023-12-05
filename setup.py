from setuptools import setup, find_packages

exec(open("stackscope/_version.py", encoding="utf-8").read())

LONG_DESC = open("README.rst", encoding="utf-8").read()

setup(
    name="stackscope",
    version=__version__,
    description="Unusually detailed Python stack introspection",
    url="https://github.com/oremanj/stackscope",
    long_description=LONG_DESC,
    author="Joshua Oreman",
    author_email="oremanj@gmail.com",
    license="MIT -or- Apache License 2.0",
    packages=find_packages(),
    install_requires=["exceptiongroup >= 1.0.0; python_version < '3.11'"],
    include_package_data=True,
    keywords=["async", "debugging", "trio", "asyncio"],
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
)
