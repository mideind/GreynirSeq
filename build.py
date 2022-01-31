"""Build cython files included in the project.

This module is called by poetry during the installation process."""
import os
import sys

do_build = False
# See if Cython is installed
try:
    from Cython.Build import cythonize

    # Cython is installed.
    # We only support building for linux.
    if sys.platform.startswith("linux"):
        do_build = True
# Do nothing if Cython is not available
except ImportError:
    pass

if not do_build:
    # Got to provide this function. Otherwise, poetry will fail
    print("Cython is not or installed or different platform from linux. Skipping build.", file=sys.stderr)

    def build(setup_kwargs):
        pass

else:
    from distutils.command.build_ext import build_ext

    import numpy
    from setuptools import Extension

    # This function will be executed in setup.py:
    def build(setup_kwargs):
        # gcc arguments hack: enable optimizations
        os.environ["CFLAGS"] = "-O3"

        # Build
        setup_kwargs.update(
            {
                "ext_modules": cythonize(
                    module_list=[
                        Extension(
                            name="greynirseq.nicenlp.utils.constituency.chart_parser",
                            sources=["src/greynirseq/nicenlp/utils/constituency/chart_parser.pyx"],
                            language="c++",
                            extra_compile_args=["-fopenmp"],
                            extra_link_args=["-fopenmp"],
                            include_dirs=[numpy.get_include()],
                        ),
                        Extension(
                            name="greynirseq.nicenlp.utils.constituency.tree_dist",
                            sources=["src/greynirseq/nicenlp/utils/constituency/tree_dist.pyx"],
                            language="c++",
                            extra_compile_args=["-fopenmp"],
                            extra_link_args=["-fopenmp"],
                            include_dirs=[numpy.get_include()],
                        ),
                    ]
                ),
                "cmdclass": {"build_ext": build_ext},
            }
        )
