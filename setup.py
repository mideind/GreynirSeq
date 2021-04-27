import sys

import numpy
from Cython.Build import build_ext, cythonize
from setuptools import Extension, find_packages, setup

############
# Source: https://bit.ly/2NLVsgE
args = sys.argv[1:]
if "clean" in args:
    print("deleting Cython files...")
    import subprocess

    subprocess.run(["rm -f src/greynirseq/*/*.so src/greynirseq/**/*.so src/greynirseq/**/*.pyd"], shell=True)
else:
    # We want to always use build_ext --inplace
    if args.count("build_ext") > 0 and args.count("--inplace") == 0:
        sys.argv.insert(sys.argv.index("build_ext") + 1, "--inplace")
############


setup(
    name="greynirseq",
    description="Natural language processing for Icelandic",
    version="0.1",
    author="Miðeind ehf",
    license="AGPLv3+",
    author_email="tauganet@mideind.is",
    url="https://github.com/mideind/GreynirSeq",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    setup_requires=["cython"],
    install_requires=["cython"],
    ext_modules=cythonize(
        [
            Extension(
                name="greynirseq.nicenlp.chart_parser",
                sources=["src/greynirseq/nicenlp/utils/constituency/chart_parser.pyx"],
                language="c++",
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="greynirseq.nicenlp.tree_dist",
                sources=["src/greynirseq/nicenlp/utils/constituency/tree_dist.pyx"],
                language="c++",
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"],
                include_dirs=[numpy.get_include()],
            ),
        ]
    ),
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Natural Language :: Icelandic",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
        "Environment :: GPU :: NVIDIA CUDA :: 10.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Environment :: GPU :: NVIDIA CUDA :: 11.1",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords=["nlp", "pos", "ner", "icelandic"],
)
