# Adapted with modifications from georgeliaw/rules_wheel/wheel
"""Bazel rule for building a python wheel"""

def _generate_setup_py(ctx):
    classifiers = '[{}]'.format(','.join(['"{}"'.format(i) for i in ctx.attr.classifiers]))
    install_requires = '[{}]'.format(','.join(['"{}"'.format(i) for i in ctx.attr.install_requires]))
    setup_py = ctx.actions.declare_file("{}/setup.py".format(ctx.attr.name))

    # create setup.py
    ctx.actions.expand_template(
        template=ctx.file._setup_py_template,
        output=setup_py,
        substitutions={
            "{name}": ctx.attr.name,
            "{version}": ctx.attr.version,
            "{description}": ctx.attr.description,
            "{classifiers}": classifiers,
            "{platforms}": str(ctx.attr.platform),
            "{package_data}": str(ctx.attr.data),
            "{include_package_data}": str(ctx.attr.include_package_data),
            "{install_requires}": install_requires
        },
        is_executable=True
    )

    return setup_py

def _generate_manifest(ctx, package_name):
    manifest_text = '\n'.join([i for i in ctx.attr.manifest]).format(package_name=package_name)

    manifest = ctx.actions.declare_file("{}/MANIFEST.in".format(ctx.attr.name))
    ctx.actions.expand_template(
        template=ctx.file._manifest_template,
        output=manifest,
        substitutions={
            "{manifest}": manifest_text
        },
        is_executable=True
    )

    return manifest

def _bdist_wheel_impl(ctx):
    # use the rule name in the work dir path in case multiple wheels are declared in the same BUILD file
    work_dir = "{}/wheel".format(ctx.attr.name)
    build_file_dir = ctx.build_file_path.rstrip('/BUILD')

    package_dir = ctx.actions.declare_directory(work_dir)
    package_name = package_dir.dirname.split('/')[-1]

    setup_py_dest_dir = '/'.join([
        package_dir.path,
        '/'.join(build_file_dir.split('/')[:-1]),
        ctx.attr.strip_src_prefix.strip('/')
    ])
    backtrack_path = '/'.join(['..' for i in setup_py_dest_dir.split('/') if i])

    setup_py = _generate_setup_py(ctx)
    manifest = _generate_manifest(ctx, package_name)

    source_list = ' '.join([src.path for src in ctx.files.srcs])

    ctx.actions.run_shell(
        mnemonic="CreateWorkDir",
        outputs=[package_dir],
        inputs=[],
        command="mkdir -p {package_dir}".format(package_dir=package_dir.path)
    )

    command = "chmod 0775 {package_dir} " \
              + "&& rsync -R {source_list} {package_dir} " \
              + "&& cp {setup_py_path} {setup_py_dest_dir} " \
              + "&& cp {manifest_path} {setup_py_dest_dir} " \
              + "&& cd {setup_py_dest_dir} " \
              + "&& python setup.py bdist_wheel --dist-dir {dist_dir} "

    ctx.actions.run_shell(
        mnemonic="BuildWheel",
        outputs=[ctx.outputs.wheel],
        inputs=[src for src in ctx.files.srcs] + [package_dir, setup_py, manifest],
        command=command.format(
            source_list=source_list,
            setup_py_path=ctx.outputs.setup_py.path,
            manifest_path=ctx.outputs.manifest.path,
            package_dir=package_dir.path,
            setup_py_dest_dir=setup_py_dest_dir,
            bdist_dir=package_dir.path + "/build",
            dist_dir=backtrack_path + "/" + ctx.outputs.wheel.dirname,
        )
    )

    return DefaultInfo(files=depset([ctx.outputs.wheel]))

bdist_wheel = rule(
    implementation = _bdist_wheel_impl,
    executable = False,
    attrs = {
        "srcs": attr.label_list(
            allow_files=[".py"],
            mandatory=True,
            allow_empty=False
        ),
        "strip_src_prefix": attr.string(
            mandatory=False
        ),
        "version": attr.string(
            default='0.0.1',
            mandatory=False
        ),
        "description": attr.string(
            mandatory=False
        ),
        "classifiers": attr.string_list(
            mandatory=False
        ),
        "platform": attr.string_list(
            default=['any'],
            mandatory=False
        ),
        "data": attr.string_list_dict(
            mandatory=False
        ),
        "manifest": attr.string_list(
            default=['recursive-include {package_name} *'],
            mandatory=False
        ),
        "include_package_data": attr.bool(
            default=False,
            mandatory=False
        ),
        "install_requires": attr.string_list(
            mandatory=False
        ),
        "_setup_py_template": attr.label(
            default=Label("@//third_party/templates:setup.py.tpl"),
            allow_single_file=True
        ),
        "_manifest_template": attr.label(
            default=Label("@//third_party/templates:MANIFEST.in.tpl"),
            allow_single_file=True
        )
    },
    outputs = {
        "wheel": "%{name}-%{version}-py2-none-%{platform}.whl",
        "setup_py": "%{name}/setup.py",
        "manifest": "%{name}/MANIFEST.in"
    },
)
