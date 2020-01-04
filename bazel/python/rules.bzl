def _parent_path(path):
    return path.rsplit('/', 1)[0]

def _subpath(full, root):
    root = root.strip('/') + '/'
    full = full.strip('/') + '/'
    len_root = len(root)
    if (len_root <= len(full) and full[:len_root] == root):
        return full[len_root:].rstrip('/')
    return full.rstrip('/')

def _is_subpath_of(full, root):
    root = root.strip('/') + '/'
    return full[:len(root)] == root

def _path_in_package(ctx, path):
    build_dir_path = _parent_path(ctx.build_file_path)
    path_in_build_dir = _subpath(path, build_dir_path)
    return path_in_build_dir

def _copy_to_package(ctx, file):
    path_in_package = _path_in_package(ctx, file.short_path)
    file_copy = ctx.actions.declare_file(path_in_package)
    ctx.actions.run_shell(
        inputs  = [ file ],
        outputs = [ file_copy ],
        command = "cp {} {}".format(file.path, file_copy.path)
    )
    return file_copy

def _py_package_impl(ctx):
    target_files = []
    for file in ctx.files.srcs:
        file_copy = _copy_to_package(ctx, file)
        target_files.append(file_copy)

    for src in ctx.attr.srcs:
        rfiles = src[DefaultInfo].default_runfiles.files
        files = [ f for f in rfiles.to_list() if not f.is_source ]
        target_files += files

    return DefaultInfo(
        files = depset(target_files),
        runfiles = ctx.runfiles(files=target_files)
    )

py_package = rule(
    implementation = _py_package_impl,
    attrs = {
        'srcs': attr.label_list(
            mandatory = True,
            allow_files = True,
        )
    },
)
