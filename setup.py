from setuptools import setup, find_packages

setup(
    name='RuAG',
    version='1.0.0',
    author='ReedZyd',
    author_email='y.zhang5@tue.nl',
    description='ruag',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        "openai",
        "matplotlib",
        "opencv_python",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)