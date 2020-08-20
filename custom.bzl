def clean_dep(dep):
    return str(Label(dep))

def _is_windows(ctx):
    return ctx.os.name.lower().find("windows") != -1

def _wrap_bash_cmd(ctx, cmd):
    if _is_windows(ctx):
        bazel_sh = _get_env_var(ctx, "BAZEL_SH")
        if not bazel_sh:
            fail("BAZEL_SH environment variable is not set")
        cmd = [bazel_sh, "-l", "-c", " ".join(["\"%s\"" % s for s in cmd])]
    return cmd

def _get_env_var(ctx, name):
    if name in ctx.os.environ:
        return ctx.os.environ[name]
    else:
        return None

# Checks if we should use the system lib instead of the bundled one
def _use_system_lib(ctx, name):
    syslibenv = _get_env_var(ctx, "TF_SYSTEM_LIBS")
    if syslibenv:
        for n in syslibenv.strip().split(","):
            if n.strip() == name:
                return True
    return False

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
    result = repo_ctx.execute(cmd_and_args, timeout = 10)
    if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n" +
              "Stderr: {3}").format(
            " ".join(cmd_and_args),
            result.return_code,
            result.stdout,
            result.stderr,
        ))

def _repos_are_siblings():
    return Label("@foo//bar").workspace_root.startswith("../")

# Apply a patch_file to the repository root directory
# Runs 'patch -p1'
def _apply_patch(ctx, patch_file):
    # Don't check patch on Windows, because patch is only available under bash.
    if not _is_windows(ctx) and not ctx.which("patch"):
        fail("patch command is not found, please install it")
    cmd = _wrap_bash_cmd(
        ctx,
        ["patch", "-p1", "-d", ctx.path("."), "-i", ctx.path(patch_file)],
    )
    _execute_and_check_ret_code(ctx, cmd)

def _apply_delete(ctx, paths):
    for path in paths:
        if path.startswith("/"):
            fail("refusing to rm -rf path starting with '/': " + path)
        if ".." in path:
            fail("refusing to rm -rf path containing '..': " + path)
    cmd = _wrap_bash_cmd(ctx, ["rm", "-rf"] + [ctx.path(path) for path in paths])
    _execute_and_check_ret_code(ctx, cmd)


def _tf_http_archive(ctx):
    if ("mirror.bazel.build" not in ctx.attr.urls[0] and
        (len(ctx.attr.urls) < 2 )):
        fail("tf_http_archive(urls) must have redundant URLs. The " +
             "mirror.bazel.build URL must be present and it must come first. " +
             "Even if you don't have permission to mirror the file, please " +
             "put the correctly formatted mirror URL there anyway, because " +
             "someone will come along shortly thereafter and mirror the file.")

    use_syslib = _use_system_lib(ctx, ctx.attr.name)
    if not use_syslib:
        ctx.download_and_extract(
            ctx.attr.urls,
            "",
            ctx.attr.sha256,
            ctx.attr.type,
            ctx.attr.strip_prefix,
        )
        if ctx.attr.delete:
            _apply_delete(ctx, ctx.attr.delete)
        if ctx.attr.patch_file != None:
            _apply_patch(ctx, ctx.attr.patch_file)

    if use_syslib and ctx.attr.system_build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.system_build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    elif ctx.attr.build_file != None:
        # Use BUILD.bazel to avoid conflict with third party projects with
        # BUILD or build (directory) underneath.
        ctx.template("BUILD.bazel", ctx.attr.build_file, {
            "%prefix%": ".." if _repos_are_siblings() else "external",
        }, False)

    if use_syslib:
        for internal_src, external_dest in ctx.attr.system_link_files.items():
            ctx.symlink(Label(internal_src), ctx.path(external_dest))

ei_http_archive = repository_rule(
    implementation = _tf_http_archive,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True, allow_empty = False),
        "strip_prefix": attr.string(),
        "type": attr.string(),
        "delete": attr.string_list(),
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "system_build_file": attr.label(),
        "system_link_files": attr.string_dict(),
    },
    environ = [
        "TF_SYSTEM_LIBS",
    ],
)

# Rule for simple expansion of template files. This performs a simple
# search over the template file for the keys in substitutions,
# and replaces them with the corresponding values.
#
# Typical usage:
#   load("/tools/build_rules/template_rule", "expand_header_template")
#   template_rule(
#       name = "ExpandMyTemplate",
#       src = "my.template",
#       out = "my.txt",
#       substitutions = {
#         "$VAR1": "foo",
#         "$VAR2": "bar",
#       }
#   )
#
# Args:
#   name: The name of the rule.
#   template: The template file to expand
#   out: The destination of the expanded file
#   substitutions: A dictionary mapping strings to their substitutions

def template_rule_impl(ctx):
  ctx.template_action(
      template = ctx.file.src,
      output = ctx.outputs.out,
      substitutions = ctx.attr.substitutions,
  )

template_rule = rule(
    attrs = {
        "src": attr.label(
            mandatory = True,
            #allow_files = True,
            allow_single_file = True,
        ),
        "substitutions": attr.string_dict(mandatory = True),
        "out": attr.output(mandatory = True),
    },
    # output_to_genfiles is required for header files.
    output_to_genfiles = True,
    implementation = template_rule_impl,
)

def bind_custom():
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # gRPC wants a cares dependency but its contents is not actually
    # important since we have set GRPC_ARES=0 in tools/bazel.rc
    native.bind(
        name = "cares",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@grpc//:grpc++_unsecure",
    )

    # Needed by gRPC
    native.bind(
        name = "libssl",
        #actual = "@openssl//:openssl",
        actual = "@boringssl//:ssl",
    )

    # Needed by gRPC
    native.bind(
        name = "nanopb",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf",
        actual = "@com_google_protobuf//:protobuf",
    )

    # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
    # to point to Protobuf's compiler library.
    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = clean_dep("//third_party/python_runtime:headers"),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

    # Needed by gRPC
    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )