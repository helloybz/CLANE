import setuptools

setuptools.setup(
    name="clane-helloybz",
    version="0.0.1",
    author="Youngbeom Choi",
    author_email="helloybz@gmail.com",
    description="A package for the implementation of my work.",
    url="https://github.com/helloybz/CLANE",
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['clane = clane.__main__:main']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)