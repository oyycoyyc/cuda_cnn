#!/usr/bin/env python3

import argparse
import io
import os
import re
import shlex
import sys


SOURCE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu")
ARTIFACT_SUFFIXES = (".o", ".obj", ".a", ".lib", ".so", ".dll")
INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*([<"])([^>"]+)[>"]'
)
SHELL_CONTROL_CHARS = ";&|<>"
ARCHIVE_TOOLS = ("ar", "llvm-ar", "gcc-ar")
REQUIRED_GENCODE = (
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=compute_90",
)


class AnalysisError(Exception):
    pass


def display_path(root, path):
    return os.path.relpath(path, root).replace(os.sep, "/")


def is_within(root, path):
    try:
        return os.path.commonpath((root, path)) == root
    except ValueError:
        return False


def resolve_path(root, token, require_file=False):
    lexical = token
    if not os.path.isabs(lexical):
        lexical = os.path.join(root, lexical)
    lexical = os.path.abspath(os.path.normpath(lexical))
    if not is_within(root, lexical):
        raise AnalysisError("path escapes source root: {0}".format(token))
    resolved = os.path.realpath(lexical)
    if not is_within(root, resolved):
        raise AnalysisError("path escapes source root through symlink: {0}".format(token))
    if require_file and not os.path.isfile(resolved):
        raise AnalysisError("source file does not exist: {0}".format(token))
    return lexical, resolved


def source_token(root, token, require_file):
    if token.startswith("-") or not token.lower().endswith(SOURCE_SUFFIXES):
        return None
    lexical, resolved = resolve_path(root, token, require_file=require_file)
    return {
        "lexical": lexical,
        "real": resolved,
        "relative": display_path(root, lexical),
    }


def executable_basename(token):
    basename = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename.endswith(".exe"):
        basename = basename[:-4]
    return basename


def positional_archive_output(tokens):
    if len(tokens) < 4 or executable_basename(tokens[0]) not in ARCHIVE_TOOLS:
        return None
    operation = tokens[1]
    if operation.startswith("-"):
        operation = operation[1:]
    if (not operation or not operation.isalpha() or
            not any(action in operation.lower() for action in ("q", "r"))):
        return None
    excluded_indexes = set((1,))
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            excluded_indexes.add(index)
            index += 1
            break
        if token in ("--plugin", "--target", "--format"):
            if index + 1 >= len(tokens):
                raise AnalysisError(
                    "archive option has no value: {0}".format(token)
                )
            excluded_indexes.update((index, index + 1))
            index += 2
            continue
        if (token == "--thin" or token.startswith("--plugin=") or
                token.startswith("--target=") or
                token.startswith("--format=") or
                token.startswith("--record-libdeps=")):
            excluded_indexes.add(index)
            index += 1
            continue
        break
    if index >= len(tokens):
        return None
    archive = tokens[index]
    if archive.startswith("-") or not archive.lower().endswith((".a", ".lib")):
        return None
    if index + 1 >= len(tokens):
        return None
    return archive, index, excluded_indexes


def output_token(tokens):
    outputs = []
    output_indexes = set()
    if tokens and executable_basename(tokens[0]) in ARCHIVE_TOOLS:
        archive_output = positional_archive_output(tokens)
        if archive_output is not None:
            output, index, excluded_indexes = archive_output
            outputs.append(output)
            output_indexes.add(index)
            output_indexes.update(excluded_indexes)
    else:
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "-o":
                if index + 1 >= len(tokens):
                    raise AnalysisError("output option has no path")
                outputs.append(tokens[index + 1])
                output_indexes.update((index, index + 1))
                index += 2
                continue
            if token.startswith("-o") and len(token) > 2:
                attached = token[2:]
                lower = attached.lower()
                if ("/" in attached or "\\" in attached or
                        lower.endswith(ARTIFACT_SUFFIXES) or
                        os.path.basename(lower) in ("lenet_cuda", "lenet_cuda.exe")):
                    outputs.append(attached)
                    output_indexes.add(index)
            index += 1
    if len(outputs) > 1:
        raise AnalysisError("recipe command has multiple output paths")
    return (outputs[0] if outputs else None), output_indexes


def include_directories(root, tokens):
    quote_dirs = []
    include_dirs = []
    option_indexes = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        destination = None
        value = None
        if token == "-I" or token == "-iquote":
            if index + 1 >= len(tokens):
                raise AnalysisError("include option has no directory: {0}".format(token))
            value = tokens[index + 1]
            destination = quote_dirs if token == "-iquote" else include_dirs
            option_indexes.update((index, index + 1))
            index += 2
        elif token.startswith("-iquote") and len(token) > len("-iquote"):
            value = token[len("-iquote"):]
            destination = quote_dirs
            option_indexes.add(index)
            index += 1
        elif token.startswith("-I") and len(token) > 2:
            value = token[2:]
            destination = include_dirs
            option_indexes.add(index)
            index += 1
        else:
            index += 1
        if destination is None:
            continue
        lexical = value if os.path.isabs(value) else os.path.join(root, value)
        lexical = os.path.abspath(os.path.normpath(lexical))
        resolved = os.path.realpath(lexical)
        if is_within(root, lexical) and not is_within(root, resolved):
            raise AnalysisError(
                "include directory escapes source root through symlink: {0}".format(value)
            )
        destination.append(lexical)
    return quote_dirs, include_dirs, option_indexes


def has_unsupported_shell(line):
    single_quoted = False
    double_quoted = False
    escaped = False
    index = 0
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if character == "\\" and not single_quoted:
            escaped = True
            index += 1
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            index += 1
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            index += 1
            continue
        if single_quoted:
            index += 1
            continue
        if character == "`" or (character == "$" and
                                  index + 1 < len(line) and
                                  line[index + 1] == "("):
            return True
        if not double_quoted and character in SHELL_CONTROL_CHARS:
            return True
        index += 1
    return False


def split_recipe(line):
    lexer = shlex.shlex(
        line, posix=True, punctuation_chars=SHELL_CONTROL_CHARS
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def parse_recipes(root, recipe_path, require_sources=True):
    commands = []
    with io.open(recipe_path, "r", encoding="utf-8") as recipe_file:
        for line_number, raw_line in enumerate(recipe_file, 1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                tokens = split_recipe(line)
            except ValueError as error:
                raise AnalysisError(
                    "cannot parse Make recipe line {0}: {1}".format(line_number, error)
                )
            if not tokens:
                continue
            output, output_indexes = output_token(tokens)
            quote_dirs, include_dirs, include_option_indexes = include_directories(
                root, tokens
            )
            sources = []
            source_indexes = set()
            for index, token in enumerate(tokens):
                if index in output_indexes or index in include_option_indexes:
                    continue
                parsed = source_token(root, token, require_sources)
                if parsed is not None:
                    sources.append(parsed)
                    source_indexes.add(index)
            response = any(token.startswith("@") for token in tokens)
            unsupported_shell = has_unsupported_shell(line)
            tests_owned = any(
                source["relative"].startswith("tests/") for source in sources
            )
            if tests_owned and response:
                raise AnalysisError(
                    "unsupported response-file input in tests-owned source recipe "
                    "on line {0}".format(line_number)
                )
            if tests_owned and unsupported_shell:
                raise AnalysisError(
                    "unsupported shell control in tests-owned source recipe on line "
                    "{0}".format(line_number)
                )
            if sources and output is None:
                raise AnalysisError(
                    "source-bearing recipe has no supported output on line {0}".format(
                        line_number
                    )
                )
            output_path = None
            if output is not None:
                _, output_path = resolve_path(root, output)
            candidates = []
            for index, token in enumerate(tokens):
                if (index in output_indexes or index in source_indexes or
                        index in include_option_indexes):
                    continue
                if token.startswith("-") or token.startswith("@"):
                    continue
                try:
                    _, candidate = resolve_path(root, token)
                except AnalysisError:
                    if token.lower().endswith(ARTIFACT_SUFFIXES):
                        raise
                    continue
                candidates.append(candidate)
            commands.append({
                "line": line,
                "line_number": line_number,
                "tokens": tokens,
                "sources": sources,
                "output": output_path,
                "candidate_inputs": candidates,
                "quote_dirs": quote_dirs,
                "include_dirs": include_dirs,
                "response": response,
                "unsupported_shell": unsupported_shell,
            })
    return commands


def read_manifest(root, manifest_path):
    expected = {}
    with io.open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line_number, raw_line in enumerate(manifest_file, 1):
            line = raw_line.rstrip("\r\n")
            if not line.startswith("test-source="):
                raise AnalysisError(
                    "malformed test-source manifest line {0}: {1}".format(
                        line_number, line
                    )
                )
            token = line[len("test-source="):]
            lexical, resolved = resolve_path(root, token, require_file=True)
            relative = display_path(root, lexical)
            if not relative.startswith("tests/"):
                raise AnalysisError(
                    "test-source manifest entry is not tests-owned: {0}".format(token)
                )
            if not relative.lower().endswith(SOURCE_SUFFIXES):
                raise AnalysisError(
                    "test-source manifest entry is not a translation unit: {0}".format(token)
                )
            if relative in expected:
                raise AnalysisError(
                    "duplicate test-source manifest entry: {0}".format(relative)
                )
            expected[relative] = resolved
    if not expected:
        raise AnalysisError("test-source manifest is empty")
    return expected


def reconcile_test_sources(root, commands, expected):
    actual = {}
    compile_contexts = []
    for command in commands:
        for source in command["sources"]:
            if source["relative"].startswith("tests/"):
                actual[source["relative"]] = source["real"]
                compile_contexts.append((
                    source["lexical"],
                    command["quote_dirs"],
                    command["include_dirs"],
                ))
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        details = []
        if missing:
            details.append("missing from recipes: {0}".format(", ".join(missing)))
        if extra:
            details.append("missing from manifest: {0}".format(", ".join(extra)))
        raise AnalysisError("test source manifest mismatch; {0}".format("; ".join(details)))
    return compile_contexts


def active_visit_key(path, quote_dirs, include_dirs):
    lexical_path = os.path.abspath(os.path.normpath(path))
    return (
        lexical_path,
        tuple(quote_dirs),
        tuple(include_dirs),
    )


def collect_active_inputs(root, compile_contexts):
    active = set()
    visited = set()
    recursion_stack = set()

    def visit(path, quote_dirs, include_dirs):
        lexical_path = os.path.abspath(os.path.normpath(path))
        key = active_visit_key(lexical_path, quote_dirs, include_dirs)
        real_path = os.path.realpath(lexical_path)
        if not is_within(root, real_path):
            raise AnalysisError(
                "active include escapes source root: {0}".format(lexical_path)
            )
        if key in visited:
            return
        # An unguarded recursive include cannot compile. Stopping a repeated
        # canonical file only in this chain models guarded/pragma-once
        # termination without hiding content from later independent routes.
        if real_path in recursion_stack:
            return
        visited.add(key)
        active.add(real_path)
        recursion_stack.add(real_path)
        try:
            try:
                source_file = io.open(real_path, "r", encoding="utf-8")
            except (IOError, OSError) as error:
                raise AnalysisError(
                    "cannot read active source input {0}: {1}".format(
                        display_path(root, real_path), error
                    )
                )
            with source_file:
                for line in source_file:
                    match = INCLUDE_RE.match(line)
                    if match is None:
                        continue
                    delimiter, include_name = match.groups()
                    if delimiter == '"':
                        search_dirs = [os.path.dirname(lexical_path)] + quote_dirs + include_dirs
                    else:
                        search_dirs = include_dirs
                    resolved = None
                    for directory in search_dirs:
                        candidate = os.path.abspath(os.path.normpath(
                            os.path.join(directory, include_name)
                        ))
                        if not os.path.exists(candidate):
                            continue
                        candidate_real = os.path.realpath(candidate)
                        if not is_within(root, candidate_real):
                            raise AnalysisError(
                                "active include escapes source root through symlink: {0}".format(
                                    include_name
                                )
                            )
                        if not os.path.isfile(candidate_real):
                            raise AnalysisError(
                                "active include is not a file: {0}".format(include_name)
                            )
                        resolved = candidate
                        break
                    if resolved is None:
                        if delimiter == '"':
                            raise AnalysisError(
                                "cannot resolve quoted include from {0}: {1}".format(
                                    display_path(root, real_path), include_name
                                )
                            )
                        continue
                    visit(resolved, quote_dirs, include_dirs)
        finally:
            recursion_stack.remove(real_path)

    for source, quote_dirs, include_dirs in compile_contexts:
        visit(source, quote_dirs, include_dirs)
    return active


def build_artifact_graph(root, commands):
    producers = {}
    for command in commands:
        output = command["output"]
        if output is None:
            continue
        if output in producers:
            raise AnalysisError(
                "multiple recipe commands produce artifact: {0}".format(
                    display_path(root, output)
                )
            )
        producers[output] = command

    for command in commands:
        inputs = []
        for candidate in command["candidate_inputs"]:
            if (candidate != command["output"] and
                    (candidate in producers or
                     candidate.lower().endswith(ARTIFACT_SUFFIXES)) and
                    candidate not in inputs):
                inputs.append(candidate)
        command["artifact_inputs"] = inputs

    application_names = set(
        os.path.normcase(path) for path in ("build/lenet_cuda", "build/lenet_cuda.exe")
    )
    applications = [
        output for output in producers
        if os.path.normcase(display_path(root, output)) in application_names
    ]
    if len(applications) != 1:
        raise AnalysisError(
            "Make recipes must expose exactly one build/lenet_cuda[.exe] output; "
            "found {0}".format(len(applications))
        )

    visited = set()

    def walk(artifact, chain):
        if artifact in visited:
            return
        visited.add(artifact)
        command = producers.get(artifact)
        if command is None:
            raise AnalysisError(
                "production graph has an input with no analyzed producer: {0}".format(
                    display_path(root, artifact)
                )
            )
        if command["response"]:
            raise AnalysisError(
                "unsupported response-file input affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        if command["unsupported_shell"]:
            raise AnalysisError(
                "unsupported obfuscated shell recipe affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        for source in command["sources"]:
            if source["relative"].startswith("tests/"):
                rendered = [display_path(root, item) for item in chain]
                raise AnalysisError(
                    "production lenet_cuda graph includes tests-owned source {0} via {1}".format(
                        source["relative"], " -> ".join(rendered)
                    )
                )
        for dependency in command["artifact_inputs"]:
            walk(dependency, chain + [dependency])

    application = applications[0]
    walk(application, [application])


def analyze_source(args):
    root = os.path.realpath(os.path.abspath(args.root))
    commands = parse_recipes(root, args.recipes)
    expected = read_manifest(root, args.manifest)
    contexts = reconcile_test_sources(root, commands, expected)
    active = collect_active_inputs(root, contexts)
    build_artifact_graph(root, commands)
    for path in sorted(active):
        sys.stdout.write(display_path(root, path) + "\n")


def analyze_build_log(args):
    root = os.path.realpath(os.path.abspath(args.root))
    commands = parse_recipes(root, args.log, require_sources=False)
    compile_found = False
    links = []
    for command in commands:
        if "-c" in command["tokens"] and command["sources"] and command["output"]:
            compile_found = True
        output = command["output"]
        if (output is not None and
                os.path.basename(output).lower() in ("lenet_cuda", "lenet_cuda.exe") and
                "-c" not in command["tokens"]):
            links.append(command)
    if not compile_found:
        raise AnalysisError("build log contains no compiler compile command")
    if not links:
        raise AnalysisError("build log contains no lenet_cuda linker command")
    if not any(all(flag in command["tokens"] for flag in REQUIRED_GENCODE)
               for command in links):
        raise AnalysisError(
            "lenet_cuda linker command is missing required compute_90 architecture flags"
        )
    for command in links:
        if command["response"]:
            raise AnalysisError(
                "unsupported response-file input affects lenet_cuda build log"
            )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze Make recipe provenance")
    subparsers = parser.add_subparsers(dest="mode")

    source = subparsers.add_parser("source")
    source.add_argument("--root", required=True)
    source.add_argument("--recipes", required=True)
    source.add_argument("--manifest", required=True)

    build_log = subparsers.add_parser("build-log")
    build_log.add_argument("--root", required=True)
    build_log.add_argument("--log", required=True)

    arguments = parser.parse_args()
    if arguments.mode is None:
        parser.error("a mode is required")
    return arguments


def main():
    arguments = parse_arguments()
    try:
        if arguments.mode == "source":
            analyze_source(arguments)
        else:
            analyze_build_log(arguments)
    except AnalysisError as error:
        sys.stderr.write("build graph analysis failed: {0}\n".format(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
