# Adapted with modifications from tensorflow/third_party/py/
"""Repository rule for Python autoconfiguration.
`python_configure` depends on the following environment variables:
  * `PYTHON_BIN_PATH`: location of python binary.
  * `PYTHON_LIB_PATH`: Location of python libraries.
"""

_BAZEL_SH = "BAZEL_SH"
_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"
_PYTHON_LIB_PATH = "PYTHON_LIB_PATH"

def _tpl(repository_ctx, tpl, substitutions={}):
    repository_ctx.template("BUILD",
                            Label("@//tools/templates:%s.BUILD.tpl" % tpl),
                            substitutions)


def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))


def _is_windows(repository_ctx):
    """Returns true if the host operating system is windows."""
    os_name = repository_ctx.os.name.lower()
    return os_name.find("windows") != -1


def _execute(repository_ctx,
             cmdline,
             error_msg=None,
             error_details=None,
             empty_stdout_fine=False):
    """Executes an arbitrary shell command.
    Args:
        repository_ctx: the repository_ctx object
        cmdline: list of strings, the command to execute
        error_msg: string, a summary of the error if the command fails
        error_details: string, details about the error or steps to fix it
        empty_stdout_fine: bool, if True, an empty stdout result is fine, otherwise
        it's an error
    Return:
        the result of repository_ctx.execute(cmdline)
  """
    result = repository_ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        _fail("\n".join([
            error_msg.strip() if error_msg else "Repository command failed",
            result.stderr.strip(), error_details if error_details else ""
        ]))
    else:
        return result


def _read_dir(repository_ctx, src_dir):
    """Returns a string with all files in a directory.
  Finds all files inside a directory, traversing subfolders and following
  symlinks. The returned string contains the full path of all files
  separated by line breaks.
  """
    if _is_windows(repository_ctx):
        src_dir = src_dir.replace("/", "\\")
        find_result = _execute(
            repository_ctx,
            ["cmd.exe", "/c", "dir", src_dir, "/b", "/s", "/a-d"],
            empty_stdout_fine=True)
        # src_files will be used in genrule.outs where the paths must
        # use forward slashes.
        return find_result.stdout.replace("\\", "/")
    else:
        find_result = _execute(
            repository_ctx, ["find", src_dir, "-follow", "-type", "f"],
            empty_stdout_fine=True)
        return find_result.stdout


def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.
  Genrule executes the given command and produces the given outputs.
  """
    return ('genrule(\n' + '    name = "' + genrule_name + '",\n' +
            '    outs = [\n' + outs + '\n    ],\n' + '    cmd = """\n' +
            command + '\n   """,\n' + ')\n')


def _normalize_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path


def _symlink_genrule_for_dir(repository_ctx,
                             src_dir,
                             dest_dir,
                             genrule_name,
                             src_files=[],
                             dest_files=[]):
    """Returns a genrule to symlink(or copy if on Windows) a set of files.
  If src_dir is passed, files will be read from the given directory; otherwise
  we assume files are in src_files and dest_files
  """
    if src_dir != None:
        src_dir = _normalize_path(src_dir)
        dest_dir = _normalize_path(dest_dir)
        files = '\n'.join(
            sorted(_read_dir(repository_ctx, src_dir).splitlines()))
        # Create a list with the src_dir stripped to use for outputs.
        dest_files = files.replace(src_dir, '').splitlines()
        src_files = files.splitlines()
    command = []
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = '$(@D)/' + dest_dir + dest_files[i] if len(
                dest_files) != 1 else '$(@D)/' + dest_files[i]
            # On Windows, symlink is not supported, so we just copy all the files.
            cmd = 'cp -f' if _is_windows(repository_ctx) else 'ln -s'
            command.append(cmd + ' "%s" "%s"' % (src_files[i], dest))
            outs.append('        "' + dest_dir + dest_files[i] + '",')
    return _genrule(src_dir, genrule_name, " && ".join(command),
                    "\n".join(outs))


def _get_python_bin(repository_ctx):
    """Gets the python bin path."""
    python_bin = repository_ctx.os.environ.get(_PYTHON_BIN_PATH)
    if python_bin != None:
        return python_bin
    python_bin_path = repository_ctx.which("python")
    if python_bin_path != None:
        return str(python_bin_path)
    _fail("Cannot find python in PATH, please make sure " +
          "python is installed and add its directory in PATH, or --define " +
          "%s='/something/else'.\nPATH=%s" %
          (_PYTHON_BIN_PATH, repository_ctx.os.environ.get("PATH", "")))


def _get_bash_bin(repository_ctx):
    """Gets the bash bin path."""
    bash_bin = repository_ctx.os.environ.get(_BAZEL_SH)
    if bash_bin != None:
        return bash_bin
    else:
        bash_bin_path = repository_ctx.which("bash")
        if bash_bin_path != None:
            return str(bash_bin_path)
        else:
            _fail(
                "Cannot find bash in PATH, please make sure " +
                "bash is installed and add its directory in PATH, or --define "
                + "%s='/path/to/bash'.\nPATH=%s" %
                (_BAZEL_SH, repository_ctx.os.environ.get("PATH", "")))


def _get_python_lib(repository_ctx, python_bin):
    """Gets the python lib path."""
    python_lib = repository_ctx.os.environ.get(_PYTHON_LIB_PATH)
    if python_lib != None:
        return python_lib
    print_lib = (
        "<<END\n" + "from __future__ import print_function\n" +
        "import site\n" + "import os\n" + "\n" + "try:\n" +
        "  input = raw_input\n" + "except NameError:\n" + "  pass\n" + "\n" +
        "python_paths = []\n" + "if os.getenv('PYTHONPATH') is not None:\n" +
        "  python_paths = os.getenv('PYTHONPATH').split(':')\n" + "try:\n" +
        "  library_paths = site.getsitepackages()\n" +
        "except AttributeError:\n" +
        " from distutils.sysconfig import get_python_lib\n" +
        " library_paths = [get_python_lib()]\n" +
        "all_paths = set(python_paths + library_paths)\n" + "paths = []\n" +
        "for path in all_paths:\n" + "  if os.path.isdir(path):\n" +
        "    paths.append(path)\n" + "if len(paths) >=1:\n" +
        "  print(paths[0])\n" + "END")
    cmd = '%s - %s' % (python_bin, print_lib)
    result = repository_ctx.execute([_get_bash_bin(repository_ctx), "-c", cmd])
    return result.stdout.strip('\n')


def _check_python_lib(repository_ctx, python_lib):
    """Checks the python lib path."""
    cmd = 'test -d "%s" -a -x "%s"' % (python_lib, python_lib)
    result = repository_ctx.execute([_get_bash_bin(repository_ctx), "-c", cmd])
    if result.return_code == 1:
        _fail("Invalid python library path: %s" % python_lib)


def _check_python_bin(repository_ctx, python_bin):
    """Checks the python bin path."""
    cmd = '[[ -x "%s" ]] && [[ ! -d "%s" ]]' % (python_bin, python_bin)
    result = repository_ctx.execute([_get_bash_bin(repository_ctx), "-c", cmd])
    if result.return_code == 1:
        _fail("--define %s='%s' is not executable. Is it the python binary?" %
              (_PYTHON_BIN_PATH, python_bin))


def _get_python_include(repository_ctx, python_bin):
    """Gets the python include path."""
    result = _execute(
        repository_ctx, [
            python_bin, "-c", 'from __future__ import print_function;' +
            'from distutils import sysconfig;' +
            'print(sysconfig.get_python_inc())'
        ],
        error_msg="Problem getting python include path.",
        error_details=(
            "Is the Python binary path set up right? " + "(See ./configure or "
            + _PYTHON_BIN_PATH + ".) " + "Is distutils installed?"))
    return result.stdout.splitlines()[0]


def _get_python_import_lib_name(repository_ctx, python_bin):
    """Get Python import library name (pythonXY.lib) on Windows."""
    result = _execute(
        repository_ctx, [
            python_bin, "-c",
            'import sys;' + 'print("python" + str(sys.version_info[0]) + ' +
            '      str(sys.version_info[1]) + ".lib")'
        ],
        error_msg="Problem getting python import library.",
        error_details=("Is the Python binary path set up right? " +
                       "(See ./configure or " + _PYTHON_BIN_PATH + ".) "))
    return result.stdout.splitlines()[0]

def _get_numpy_include(repository_ctx, python_bin):
  """Gets the numpy include path."""
  return _execute(repository_ctx,
                  [python_bin, "-c",
                   'from __future__ import print_function;' +
                   'import numpy;' +
                   ' print(numpy.get_include());'],
                  error_msg="Problem getting numpy include path.",
                  error_details="Is numpy installed?").stdout.splitlines()[0]

def _python_repository_impl(repository_ctx):
    """Creates the repository containing files set up to build with Python."""
    python_bin = _get_python_bin(repository_ctx)
    _check_python_bin(repository_ctx, python_bin)
    python_lib = _get_python_lib(repository_ctx, python_bin)
    _check_python_lib(repository_ctx, python_lib)
    python_include = _get_python_include(repository_ctx, python_bin)
    numpy_include = _get_numpy_include(repository_ctx, python_bin) + '/numpy'
    python_include_rule = _symlink_genrule_for_dir(
        repository_ctx, python_include, 'python_include', 'python_include')
    python_import_lib_genrule = ""
    # To build Python C/C++ extension on Windows,
    # we need to link to python import library pythonXY.lib
    # See https://docs.python.org/3/extending/windows.html
    if _is_windows(repository_ctx):
        python_include = _normalize_path(python_include)
        python_import_lib_name = _get_python_import_lib_name(
            repository_ctx, python_bin)
        python_import_lib_src = python_include.rsplit(
            '/', 1)[0] + "/libs/" + python_import_lib_name
        python_import_lib_genrule = _symlink_genrule_for_dir(
            repository_ctx, None, '', 'python_import_lib',
            [python_import_lib_src], [python_import_lib_name])
    numpy_include_rule = _symlink_genrule_for_dir(
        repository_ctx, numpy_include, 'numpy_include/numpy', 'numpy_include')
    _tpl(repository_ctx, "python", {
        "%{PYTHON_INCLUDE_GENRULE}": python_include_rule,
        "%{PYTHON_IMPORT_LIB_GENRULE}": python_import_lib_genrule,
        "%{NUMPY_INCLUDE_GENRULE}": numpy_include_rule,
    })


def _parent_path(path):
    return path.rsplit('/', 1)[0]


def _subpath(full, root):
    root = root.strip('/') + '/'
    full = full.strip('/') + '/'
    len_root = len(root)
    if (len_root <= len(full) and
        full[:len_root] == root):
        return full[len_root:].rstrip('/')
    return full.rstrip('/')


def _is_subpath_of(full, root):
    root = root.strip('/') + '/'
    return full[:len(root)] == root


def _path_in_package(my_ctx, path):
    path_in_build_dir = _subpath(path, my_ctx.build_dir_path)
    if not _is_subpath_of(path_in_build_dir, my_ctx.src_pack_dir_name):
        fail('All py_package files must be located in {}'.format(
             my_ctx.build_dir_path + '/' + my_ctx.src_pack_dir_name))
    path_in_root_dir = _subpath(path_in_build_dir, my_ctx.src_pack_dir_name)
    return my_ctx.dest_pack_dir_name + '/' + path_in_root_dir


def _copy_to_package(my_ctx, file):
    path_in_pack = _path_in_package(my_ctx, file.short_path)
    file_copy = my_ctx.actions.declare_file(path_in_pack)
    my_ctx.actions.run_shell(
        inputs  = [ file ],
        outputs = [ file_copy ],
        command = "cp {} {}".format(file.path, file_copy.path)
    )
    return file_copy


def _py_package_impl(ctx):
    my_ctx = struct(
        actions = ctx.actions,
        src_pack_dir_name = ctx.attr.root,
        dest_pack_dir_name = ctx.attr.name,
        build_dir_path = _parent_path(ctx.build_file_path),
    )
    files = [ _copy_to_package(my_ctx, src)
              for src in ctx.files.srcs ]
    return DefaultInfo(files = depset(files))


# Adapted with modifications from
# tensorflow/tensorflow/core/platform/default/build_config.bzl
def pyx_binaries(name, pyx_srcs=[], pyx_deps=[], cc_deps=[], **kwargs):
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
    pybins = [ x + 'so' for x in stems ]

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
            linkshared = 1,
        )

    native.filegroup(
        name = name,
        srcs = pybins,
        **kwargs
    )


python_repository = repository_rule(
    implementation=_python_repository_impl,
    environ=[
        _BAZEL_SH,
        _PYTHON_BIN_PATH,
        _PYTHON_LIB_PATH,
    ],
)


py_package = rule(
    implementation = _py_package_impl,
    attrs = {
        'srcs': attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        'root': attr.string(),
    },
)
