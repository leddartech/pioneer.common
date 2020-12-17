import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

install_reqs = parse_requirements('requirements.txt')

setuptools.setup(
    name="pioneer_common", # Replace with your own username
    version="1.0.0",
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
    install_requires=install_reqs,
    python_requires='>=3.6'
)