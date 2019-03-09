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
