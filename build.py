import os

# See if Cython is installed
try:
    import numpy
    from Cython.Build import cythonize
# Do nothing if Cython is not available
except ImportError:
    # Got to provide this function. Otherwise, poetry will fail
    def build(setup_kwargs):
        pass


# Cython is installed. Compile
else:
    from distutils.command.build_ext import build_ext

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
