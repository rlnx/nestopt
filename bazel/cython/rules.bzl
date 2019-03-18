# Adapted with modifications from
# tensorflow/tensorflow/core/platform/default/build_config.bzl
def pyx_library(name, pyx_srcs=[], pyx_deps=[], cc_deps=[], **kwargs):
    """Compiles a group of .pyx / .pxd files.

    First runs Cython to create .cpp files for each input .pyx. Then builds a
    shared object for each, passing "deps" to each cc_binary rule. Finally,
    creates a filegroup rule with the native python extensions.
    Args:
        name: Name for the rule.
        deps: C/C++ dependencies of the Cython (e.g. Numpy headers).
        srcs: .pyx, or .pxd files to either compile or pass through.
    """
    for src in pyx_srcs:
        if not src.endswith('.pyx'):
            fail("Only .pyx files are allowed in 'srcs'")

    stems = [ x[:-3] for x in pyx_srcs ]
    pybins = [ ':' + x + 'so' for x in stems ]

    for stem in stems:
        native.genrule(
            name = stem + 'cythonize',
            srcs = [ stem + 'pyx' ],
            outs = [ stem + 'cpp' ],
            cmd = ('PYTHONHASHSEED=0 python -m cython -X language_level=3 ' +
                   '--cplus $(SRCS) --output-file $(OUTS)'),
            tools = pyx_deps,
        )
        native.cc_binary(
            name = stem + 'so',
            srcs = [ stem + 'cpp' ],
            deps = cc_deps,
            linkopts = [ '-Wl,-undefined,dynamic_lookup' ],
            linkshared = 1,
        )

    native.py_library(
        name = name,
        data = pybins,
        **kwargs
    )
