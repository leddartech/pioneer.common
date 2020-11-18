import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pioneer_common", # Replace with your own username
    version="0.4.0",
    author="Leddartech",
    description="Pioneer team common utilities",
    packages=[
        'pioneer', 
        'pioneer.common',
        'pioneer.common.types'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'matplotlib',
        'future',
        'transforms3d',
        'scipy',
        'shapely',
        'sk-video',
        'ruamel.std.zipfile',
        'scikit-build',
        'open3d==0.10'
    ],
    python_requires='>=3.6'
)