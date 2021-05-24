import setuptools

long_description = "I hope You get good luck and do good things for human beings."


setuptools.setup(
    name = "centerface",
    version = "1.0.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="centerface",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/Build-Your-Own-Face-Model",
    packages=setuptools.find_packages(),
    package_data = {
        'centerface': ['final.pth'],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)