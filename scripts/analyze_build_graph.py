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
ARCHIVE_VALUE_OPTIONS = ("--plugin", "--target", "--format")
ARCHIVE_ATTACHED_VALUE_OPTIONS = (
    "--plugin=", "--target=", "--format=", "--record-libdeps=",
)
ARCHIVE_OPERATION_MODIFIERS = {
    "q": "cDflPsSTUv",
    "r": "abicDflPsSTUuv",
}
COMMAND_WRAPPERS = ("ccache", "sccache", "distcc")
COMPILER_TOOLS = ("cc", "c++", "gcc", "g++", "clang", "clang++", "nvcc")
DIRECT_LINKER_TOOLS = ("ld", "ld.bfd", "ld.gold", "ld.lld", "lld", "gold")
COMPILER_LANGUAGES = (
    "c", "c-header", "cpp-output", "c++", "c++-header",
    "c++-system-header", "c++-user-header", "c++-cpp-output", "cu", "cuda",
    "cuda-cpp-output", "assembler", "assembler-with-cpp", "none",
)
DISTINCT_O_OPTION_PREFIXES = ("-opt-info", "-openmp", "-objc")
SEPARATED_NON_ARTIFACT_OPTIONS = (
    "-MF", "-MT", "-MQ", "--sysroot",
)
ATTACHED_NON_ARTIFACT_OPTIONS = ("-MF", "-MT", "-MQ")
NVCC_NON_LIBRARY_L_OPTIONS = ("-link", "-lib", "-ltoir", "-lineinfo")
ENVIRONMENT_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
POSITIVE_INTEGER_RE = re.compile(r"^[1-9][0-9]*$")
LINKER_NAME_RE = re.compile(r"^[A-Za-z_.$][A-Za-z0-9_.$@+-]*$")
LINKER_EMULATION_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_+.-]*$")
LINKER_EXPRESSION_INTEGER_RE = re.compile(r"(?:0[xX][0-9A-Fa-f]+|[0-9]+)")
LINKER_EXPRESSION_SYMBOL_RE = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$@]*")
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


def canonical_path_identity(path):
    return os.path.normcase(os.path.abspath(os.path.realpath(path)))


def source_is_tests_owned(root, source):
    tests_root = canonical_path_identity(os.path.join(root, "tests"))
    return is_within(tests_root, canonical_path_identity(source["real"]))


def executable_basename(token):
    basename = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if basename.endswith(".exe"):
        basename = basename[:-4]
    return basename


def command_executable_index(tokens):
    index = 0
    while (index < len(tokens) and
           ENVIRONMENT_ASSIGNMENT_RE.match(tokens[index]) is not None):
        index += 1
    while (index < len(tokens) and
           executable_basename(tokens[index]) in COMMAND_WRAPPERS):
        index += 1
    if (index >= len(tokens) or not tokens[index] or
            tokens[index].startswith("-")):
        raise AnalysisError("recipe command has no executable")
    return index


def non_artifact_option_indexes(tokens):
    option_indexes = set()
    executable_index = command_executable_index(tokens)
    executable = executable_basename(tokens[executable_index])
    compiler_command = executable in COMPILER_TOOLS
    nvcc_command = executable == "nvcc"
    direct_linker_command = executable in DIRECT_LINKER_TOOLS
    index = 0

    def required_value(option_index):
        option = tokens[option_index]
        if (option_index + 1 >= len(tokens) or not tokens[option_index + 1] or
                tokens[option_index + 1].startswith("-")):
            raise AnalysisError("option has no value: {0}".format(option))
        option_indexes.update((option_index, option_index + 1))
        return tokens[option_index + 1], option_index + 2

    def invalid_value(option, value):
        raise AnalysisError(
            "invalid option value for {0}: {1}".format(option, value)
        )

    def path_like(value):
        lower = value.lower()
        return ("/" in value or "\\" in value or
                lower.endswith(SOURCE_SUFFIXES + ARTIFACT_SUFFIXES))

    def validate_symbol(option, value):
        if path_like(value) or LINKER_NAME_RE.match(value) is None:
            invalid_value(option, value)

    def valid_linker_expression(expression):
        position = 0
        parentheses = 0
        expects_operand = True
        while position < len(expression):
            character = expression[position]
            if character.isspace():
                position += 1
                continue
            if expects_operand:
                if character in "+-~":
                    position += 1
                    continue
                if character == "(":
                    parentheses += 1
                    position += 1
                    continue
                match = LINKER_EXPRESSION_INTEGER_RE.match(expression, position)
                if match is None:
                    match = LINKER_EXPRESSION_SYMBOL_RE.match(expression, position)
                if match is None:
                    return False
                position = match.end()
                expects_operand = False
                continue
            if character == ")":
                if parentheses == 0:
                    return False
                parentheses -= 1
                position += 1
                continue
            if expression.startswith(("<<", ">>"), position):
                position += 2
                expects_operand = True
                continue
            if character in "+-*/%&|^":
                position += 1
                expects_operand = True
                continue
            return False
        return not expects_operand and parentheses == 0

    def validate_defsym(option, value):
        if "=" not in value:
            invalid_value(option, value)
        symbol, expression = value.split("=", 1)
        if (LINKER_NAME_RE.match(symbol) is None or
                not valid_linker_expression(expression)):
            invalid_value(option, value)

    def nvcc_option(token):
        return (token in ("--threads", "-t", "-ccbin", "--compiler-bindir") or
                token.startswith("--threads=") or token.startswith("-t=") or
                (token.startswith("-t") and len(token) > 2 and
                 (nvcc_command or token[2:].isdigit())) or
                token.startswith("-ccbin=") or
                token.startswith("--compiler-bindir="))

    def linker_option(token):
        return (token in (
                    "-m", "-e", "-u", "--emulation", "--entry",
                    "--undefined", "--defsym"
                ) or any(token.startswith(prefix) for prefix in (
                    "--emulation=", "--entry=", "--undefined=", "--defsym="
                )))

    while index < len(tokens):
        token = tokens[index]
        if direct_linker_command and token == "-t":
            option_indexes.add(index)
            index += 1
            continue
        if nvcc_option(token) and not nvcc_command:
            raise AnalysisError(
                "unsupported scoped option for {0}: {1}".format(executable, token)
            )
        if linker_option(token) and not direct_linker_command:
            raise AnalysisError(
                "unsupported scoped option for {0}: {1}".format(executable, token)
            )
        if token in SEPARATED_NON_ARTIFACT_OPTIONS:
            _, index = required_value(index)
            continue
        if compiler_command and token == "-x":
            value, index = required_value(index)
            if value not in COMPILER_LANGUAGES:
                invalid_value(token, value)
            continue
        if nvcc_command and token in ("--threads", "-t"):
            value, index = required_value(index)
            if POSITIVE_INTEGER_RE.match(value) is None:
                invalid_value(token, value)
            continue
        if nvcc_command and token in ("-ccbin", "--compiler-bindir"):
            _, index = required_value(index)
            continue
        if direct_linker_command and token in ("-m", "--emulation"):
            value, index = required_value(index)
            if (path_like(value) or
                    LINKER_EMULATION_RE.match(value) is None):
                invalid_value(token, value)
            continue
        if direct_linker_command and token in (
                "-e", "--entry", "-u", "--undefined"):
            value, index = required_value(index)
            validate_symbol(token, value)
            continue
        if direct_linker_command and token == "--defsym":
            value, index = required_value(index)
            validate_defsym(token, value)
            continue
        if token.startswith("--sysroot="):
            option_indexes.add(index)
        elif any(token.startswith(option) and len(token) > len(option)
                  for option in ATTACHED_NON_ARTIFACT_OPTIONS):
            option_indexes.add(index)
        elif nvcc_command and token.startswith("--threads="):
            value = token.split("=", 1)[1]
            if POSITIVE_INTEGER_RE.match(value) is None:
                invalid_value("--threads", value)
            option_indexes.add(index)
        elif nvcc_command and token.startswith("-t") and len(token) > 2:
            value = token[2:]
            if value.startswith("="):
                value = value[1:]
            if POSITIVE_INTEGER_RE.match(value) is None:
                invalid_value("-t", value)
            option_indexes.add(index)
        elif nvcc_command and any(token.startswith(prefix) for prefix in (
                "-ccbin=", "--compiler-bindir=")):
            value = token.split("=", 1)[1]
            if not value or value.startswith("-"):
                raise AnalysisError(
                    "option has no value: {0}".format(token.split("=", 1)[0])
                )
            option_indexes.add(index)
        elif direct_linker_command and token.startswith("--emulation="):
            value = token.split("=", 1)[1]
            if (path_like(value) or
                    LINKER_EMULATION_RE.match(value) is None):
                invalid_value("--emulation", value)
            option_indexes.add(index)
        elif direct_linker_command and any(token.startswith(prefix) for prefix in (
                "--entry=", "--undefined=")):
            option, value = token.split("=", 1)
            validate_symbol(option, value)
            option_indexes.add(index)
        elif direct_linker_command and token.startswith("--defsym="):
            validate_defsym("--defsym", token[len("--defsym="):])
            option_indexes.add(index)
        index += 1
    return option_indexes


def command_prefix_indexes(tokens):
    executable_index = command_executable_index(tokens)
    prefix_indexes = set(range(executable_index))
    if executable_index < len(tokens):
        prefix_indexes.add(executable_index)
    return prefix_indexes


def positional_archive_output(tokens):
    if not tokens:
        return None
    executable_index = command_executable_index(tokens)
    if executable_basename(tokens[executable_index]) not in ARCHIVE_TOOLS:
        return None

    excluded_indexes = set()
    index = executable_index + 1

    def consume_options(current):
        while current < len(tokens):
            token = tokens[current]
            if token in ARCHIVE_VALUE_OPTIONS:
                if current + 1 >= len(tokens):
                    raise AnalysisError(
                        "archive option has no value: {0}".format(token)
                    )
                excluded_indexes.update((current, current + 1))
                current += 2
                continue
            if token == "--thin" or any(
                    token.startswith(prefix) and len(token) > len(prefix)
                    for prefix in ARCHIVE_ATTACHED_VALUE_OPTIONS):
                excluded_indexes.add(current)
                current += 1
                continue
            break
        return current

    index = consume_options(index)
    if index >= len(tokens):
        raise AnalysisError("archive command has no operation")
    operation_index = index
    operation = tokens[index]
    if operation.startswith("-"):
        operation = operation[1:]
    if not operation or not operation.isalpha():
        raise AnalysisError(
            "unsupported archive operation: {0}".format(tokens[operation_index])
        )
    if operation.count("q") + operation.count("r") != 1 or any(
            action in operation for action in "dmptx"):
        raise AnalysisError(
            "ambiguous archive operation: {0}".format(tokens[operation_index])
        )
    producer_operation = "q" if "q" in operation else "r"
    allowed = set(
        producer_operation + ARCHIVE_OPERATION_MODIFIERS[producer_operation]
    )
    if any(character not in allowed for character in operation):
        raise AnalysisError(
            "unsupported archive modifier for {0}: {1}".format(
                producer_operation, tokens[operation_index]
            )
        )
    placement = [modifier for modifier in "abi" if modifier in operation]
    if len(placement) > 1:
        raise AnalysisError(
            "ambiguous archive position modifiers: {0}".format(
                tokens[operation_index]
            )
        )
    if any(operation.count(modifier) > 1 for modifier in allowed):
        raise AnalysisError(
            "duplicate archive operation modifier: {0}".format(
                tokens[operation_index]
            )
        )

    excluded_indexes.add(operation_index)
    index = consume_options(operation_index + 1)
    if index < len(tokens) and tokens[index] == "--":
        excluded_indexes.add(index)
        index += 1
    if placement:
        if index >= len(tokens):
            raise AnalysisError("archive position modifier has no relative member")
        excluded_indexes.add(index)
        index += 1
    if index >= len(tokens):
        raise AnalysisError("archive command has no archive path")
    archive = tokens[index]
    if not archive or archive.startswith("-"):
        raise AnalysisError("archive command has invalid archive path: {0}".format(archive))
    for member in tokens[index + 1:]:
        if member.startswith("-"):
            raise AnalysisError(
                "unsupported archive option after archive path: {0}".format(member)
            )
    return archive, index, excluded_indexes


def output_token(tokens, opaque_indexes):
    outputs = []
    output_indexes = set()
    stateful_archive = False
    executable_index = command_executable_index(tokens)
    if executable_basename(tokens[executable_index]) in ARCHIVE_TOOLS:
        archive_output = positional_archive_output(tokens)
        if archive_output is not None:
            output, index, excluded_indexes = archive_output
            outputs.append(output)
            output_indexes.add(index)
            output_indexes.update(excluded_indexes)
            stateful_archive = True
    else:
        index = 0
        while index < len(tokens):
            if index in opaque_indexes:
                index += 1
                continue
            token = tokens[index]
            if token == "-o":
                if index + 1 >= len(tokens):
                    raise AnalysisError("output option has no path")
                outputs.append(tokens[index + 1])
                output_indexes.update((index, index + 1))
                index += 2
                continue
            if token.startswith("-o") and len(token) > 2:
                lower = token.lower()
                if any(lower.startswith(prefix)
                       for prefix in DISTINCT_O_OPTION_PREFIXES):
                    index += 1
                    continue
                outputs.append(token[2:])
                output_indexes.add(index)
            index += 1
    if len(outputs) > 1:
        raise AnalysisError("recipe command has multiple output paths")
    return (outputs[0] if outputs else None), output_indexes, stateful_archive


def include_directories(root, tokens, ignored_indexes=None):
    quote_dirs = []
    include_dirs = []
    system_dirs = []
    option_indexes = set()
    ignored_indexes = ignored_indexes or set()
    index = 0
    while index < len(tokens):
        if index in ignored_indexes:
            index += 1
            continue
        token = tokens[index]
        destination = None
        value = None
        if token in ("-I", "-iquote", "-isystem"):
            if index + 1 >= len(tokens):
                raise AnalysisError("include option has no directory: {0}".format(token))
            value = tokens[index + 1]
            if token == "-iquote":
                destination = quote_dirs
            elif token == "-isystem":
                destination = system_dirs
            else:
                destination = include_dirs
            option_indexes.update((index, index + 1))
            index += 2
        elif token.startswith("-iquote") and len(token) > len("-iquote"):
            value = token[len("-iquote"):]
            if value == "=":
                value = ""
            destination = quote_dirs
            option_indexes.add(index)
            index += 1
        elif token.startswith("-I") and len(token) > 2:
            value = token[2:]
            if value == "=":
                value = ""
            destination = include_dirs
            option_indexes.add(index)
            index += 1
        elif token.startswith("-isystem") and len(token) > len("-isystem"):
            value = token[len("-isystem"):]
            if value.startswith("="):
                value = value[1:]
            if not value:
                raise AnalysisError("include option has no directory: -isystem")
            destination = system_dirs
            option_indexes.add(index)
            index += 1
        else:
            index += 1
        if destination is None:
            continue
        if not value:
            raise AnalysisError("include option has no directory: {0}".format(token))
        lexical = value if os.path.isabs(value) else os.path.join(root, value)
        lexical = os.path.abspath(os.path.normpath(lexical))
        resolved = os.path.realpath(lexical)
        if is_within(root, lexical) and not is_within(root, resolved):
            raise AnalysisError(
                "include directory escapes source root through symlink: {0}".format(value)
            )
        destination.append(lexical)
    system_identities = set(canonical_path_identity(path) for path in system_dirs)
    include_dirs = [
        path for path in include_dirs
        if canonical_path_identity(path) not in system_identities
    ]
    return quote_dirs, include_dirs, system_dirs, option_indexes


def forced_input_options(tokens, ignored_indexes=None):
    forced_inputs = []
    option_indexes = set()
    ignored_indexes = ignored_indexes or set()
    index = 0
    while index < len(tokens):
        if index in ignored_indexes:
            index += 1
            continue
        token = tokens[index]
        matched = None
        value = None
        if token in ("-include", "-imacros"):
            if index + 1 >= len(tokens):
                raise AnalysisError("forced-input option has no file: {0}".format(token))
            matched = token
            value = tokens[index + 1]
            option_indexes.update((index, index + 1))
            index += 2
        else:
            for option in ("-include", "-imacros"):
                if token.startswith(option) and len(token) > len(option):
                    matched = option
                    value = token[len(option):]
                    if value.startswith("="):
                        value = value[1:]
                    option_indexes.add(index)
                    index += 1
                    break
        if matched is None:
            index += 1
            continue
        if not value:
            raise AnalysisError("forced-input option has no file: {0}".format(matched))
        forced_inputs.append(value)
    return forced_inputs, option_indexes


def active_compiler_tokens(tokens):
    active_tokens = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        payload = None
        if token in ("-Xcompiler", "--compiler-options"):
            if index + 1 >= len(tokens):
                raise AnalysisError(
                    "host compiler forwarding option has no value: {0}".format(token)
                )
            payload = tokens[index + 1]
            index += 2
        elif (token.startswith("-Xcompiler=") or
              token.startswith("--compiler-options=")):
            payload = token.split("=", 1)[1]
            index += 1
        else:
            active_tokens.append(token)
            index += 1
            continue
        if not payload:
            raise AnalysisError(
                "host compiler forwarding option has no value: {0}".format(token)
            )
        fields = payload.split(",")
        if any(not field for field in fields):
            raise AnalysisError(
                "host compiler forwarding option has an empty field: {0}".format(token)
            )
        active_tokens.extend(fields)
    return active_tokens


def command_input_metadata(tokens):
    response_file = False
    direct_linker = False
    library_search = False
    library_name = False
    external_control = False
    opaque_indexes = set()
    executable_index = command_executable_index(tokens)
    nvcc_command = (
        executable_index < len(tokens) and
        executable_basename(tokens[executable_index]) == "nvcc"
    )
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if nvcc_command and token == "-ldir":
            if (index + 1 >= len(tokens) or not tokens[index + 1] or
                    tokens[index + 1].startswith("-")):
                raise AnalysisError("option has no value: -ldir")
            opaque_indexes.update((index, index + 1))
            index += 2
            continue
        if token in ("-T", "--script", "-specs", "--config"):
            if index + 1 >= len(tokens) or not tokens[index + 1]:
                raise AnalysisError(
                    "control option has no file: {0}".format(token)
                )
            external_control = True
            opaque_indexes.update((index, index + 1))
            index += 2
            continue
        control_value = None
        if token.startswith("-T") and len(token) > len("-T"):
            control_value = token[len("-T"):]
            if control_value.startswith("="):
                control_value = control_value[1:]
        else:
            for prefix in ("--script=", "-specs=", "--config="):
                if token.startswith(prefix):
                    control_value = token[len(prefix):]
                    break
        if control_value is not None:
            if not control_value:
                raise AnalysisError(
                    "control option has no file: {0}".format(token)
                )
            external_control = True
            opaque_indexes.add(index)
            index += 1
            continue
        if token.startswith("@"):
            response_file = True
            opaque_indexes.add(index)
        if token.startswith("-Wl,"):
            direct_linker = True
            opaque_indexes.add(index)
            if "@" in token[4:]:
                response_file = True
        if token in ("--options-file", "-optf"):
            response_file = True
            opaque_indexes.add(index)
            if index + 1 < len(tokens):
                opaque_indexes.add(index + 1)
                index += 1
        elif (token.startswith("--options-file=") or
              token.startswith("-optf=") or
              (token.startswith("-optf") and len(token) > len("-optf"))):
            response_file = True
            opaque_indexes.add(index)
        if token in ("-Xcompiler", "--compiler-options"):
            opaque_indexes.add(index)
            if index + 1 < len(tokens):
                opaque_indexes.add(index + 1)
                if "@" in tokens[index + 1]:
                    response_file = True
                index += 1
        elif (token.startswith("-Xcompiler=") or
              token.startswith("--compiler-options=")):
            opaque_indexes.add(index)
            if "@" in token.split("=", 1)[1]:
                response_file = True
        if token in ("-Xlinker", "--linker-options"):
            direct_linker = True
            opaque_indexes.add(index)
            if index + 1 < len(tokens):
                opaque_indexes.add(index + 1)
                if "@" in tokens[index + 1]:
                    response_file = True
                index += 1
        elif (token.startswith("-Xlinker=") or
              token.startswith("--linker-options=")):
            direct_linker = True
            opaque_indexes.add(index)
            if "@" in token.split("=", 1)[1]:
                response_file = True
        if token == "-L":
            library_search = True
        elif token.startswith("-L") and len(token) > 2:
            library_search = True
        if token == "-l":
            library_name = True
            opaque_indexes.add(index)
            if index + 1 < len(tokens):
                opaque_indexes.add(index + 1)
                index += 1
        elif (token.startswith("-l") and len(token) > 2 and not (
                nvcc_command and token in NVCC_NON_LIBRARY_L_OPTIONS)):
            library_name = True
        index += 1
    return {
        "response_file": response_file,
        "linker_input": direct_linker or (library_search and library_name),
        "library_input": library_name,
        "external_control": external_control,
    }, opaque_indexes


def bracket_subexpression_end(line, start):
    delimiter = line[start + 1]
    single_quoted = False
    double_quoted = False
    escaped = False
    index = start + 2
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
        elif character == "\\" and not single_quoted:
            escaped = True
        elif character == "'" and not double_quoted:
            single_quoted = not single_quoted
        elif character == '"' and not single_quoted:
            double_quoted = not double_quoted
        elif not single_quoted and not double_quoted:
            if (character == delimiter and index + 1 < len(line) and
                    line[index + 1] == "]"):
                return index + 2
            if character.isspace():
                return None
        index += 1
    return None


def has_glob_bracket(line, start):
    single_quoted = False
    double_quoted = False
    escaped = False
    fallback_close = False
    index = start + 1
    if index < len(line) and line[index] in "!^":
        index += 1
    if index < len(line) and line[index] == "]":
        index += 1
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
        elif character == "\\" and not single_quoted:
            escaped = True
        elif character == "'" and not double_quoted:
            single_quoted = not single_quoted
        elif character == '"' and not single_quoted:
            double_quoted = not double_quoted
        elif not single_quoted and not double_quoted:
            if (character == "[" and index + 1 < len(line) and
                    line[index + 1] in ".:="):
                subexpression_end = bracket_subexpression_end(line, index)
                if subexpression_end is not None:
                    fallback_close = True
                    index = subexpression_end
                    continue
            if character == "]":
                return True
            if character.isspace():
                return fallback_close
        index += 1
    return fallback_close


def shell_command_portion(line):
    single_quoted = False
    double_quoted = False
    escaped = False
    word_start = True
    index = 0
    while index < len(line):
        character = line[index]
        if escaped:
            escaped = False
            word_start = False
            index += 1
            continue
        if character == "\\" and not single_quoted:
            escaped = True
            word_start = False
            index += 1
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            word_start = False
            index += 1
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            word_start = False
            index += 1
            continue
        if not single_quoted and not double_quoted:
            if character == "#" and word_start:
                return line[:index].rstrip()
            if character.isspace() or character in SHELL_CONTROL_CHARS:
                word_start = True
            else:
                word_start = False
        index += 1
    return line


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
        if character == "`" or character == "$":
            return True
        if not double_quoted and character in SHELL_CONTROL_CHARS:
            return True
        if not double_quoted and character in "*?":
            return True
        if not double_quoted and character == "[":
            if has_glob_bracket(line, index):
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
            line = shell_command_portion(raw_line.rstrip("\r\n"))
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
            try:
                command_executable_index(tokens)
            except AnalysisError:
                if (not require_sources and all(
                        ENVIRONMENT_ASSIGNMENT_RE.match(token) is not None
                        for token in tokens)):
                    continue
                raise
            prefix_indexes = command_prefix_indexes(tokens)
            input_metadata, opaque_input_indexes = command_input_metadata(tokens)
            non_artifact_indexes = non_artifact_option_indexes(tokens)
            output, output_indexes, stateful_archive = output_token(
                tokens, opaque_input_indexes | non_artifact_indexes
            )
            input_metadata["stateful_archive"] = stateful_archive
            compiler_tokens = active_compiler_tokens(tokens)
            quote_dirs, include_dirs, system_dirs, include_option_indexes = \
                include_directories(root, compiler_tokens)
            forced_inputs, _ = forced_input_options(compiler_tokens)
            _, _, _, direct_include_option_indexes = include_directories(
                root, tokens, opaque_input_indexes
            )
            _, forced_option_indexes = forced_input_options(
                tokens, opaque_input_indexes
            )
            include_option_indexes = direct_include_option_indexes
            compiler_option_indexes = include_option_indexes | forced_option_indexes
            sources = []
            source_indexes = set()
            for index, token in enumerate(tokens):
                if (index in output_indexes or index in compiler_option_indexes or
                        index in opaque_input_indexes or
                        index in non_artifact_indexes or
                        index in prefix_indexes):
                    continue
                parsed = source_token(root, token, require_sources)
                if parsed is not None:
                    sources.append(parsed)
                    source_indexes.add(index)
            unsupported_shell = has_unsupported_shell(line)
            input_metadata["unsupported_shell"] = unsupported_shell
            tests_owned = any(source_is_tests_owned(root, source) for source in sources)
            if tests_owned and input_metadata["response_file"]:
                raise AnalysisError(
                    "unsupported response-file input in tests-owned source recipe "
                    "on line {0}".format(line_number)
                )
            if tests_owned and input_metadata["linker_input"]:
                raise AnalysisError(
                    "unsupported linker-encoded input in tests-owned source recipe "
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
                output_path, _ = resolve_path(root, output)
            candidates = []
            invalid_candidates = []
            for index, token in enumerate(tokens):
                if (index in output_indexes or index in source_indexes or
                        index in compiler_option_indexes or
                        index in opaque_input_indexes or
                        index in non_artifact_indexes or
                        index in prefix_indexes):
                    continue
                if token.startswith("-") or token.startswith("@"):
                    continue
                try:
                    candidate, _ = resolve_path(root, token)
                except AnalysisError as error:
                    invalid_candidates.append(str(error))
                    continue
                candidates.append(candidate)
            commands.append({
                "line": line,
                "line_number": line_number,
                "tokens": tokens,
                "sources": sources,
                "output": output_path,
                "candidate_inputs": candidates,
                "invalid_candidate_inputs": invalid_candidates,
                "quote_dirs": quote_dirs,
                "include_dirs": include_dirs,
                "system_dirs": system_dirs,
                "forced_inputs": forced_inputs,
                "input_metadata": input_metadata,
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
            source = {"real": resolved, "relative": relative}
            if not source_is_tests_owned(root, source):
                raise AnalysisError(
                    "test-source manifest entry is not tests-owned: {0}".format(token)
                )
            if not relative.lower().endswith(SOURCE_SUFFIXES):
                raise AnalysisError(
                    "test-source manifest entry is not a translation unit: {0}".format(token)
                )
            identity = canonical_path_identity(resolved)
            if identity in expected:
                raise AnalysisError(
                    "duplicate test-source manifest entry: {0}".format(relative)
                )
            expected[identity] = relative
    if not expected:
        raise AnalysisError("test-source manifest is empty")
    return expected


def reconcile_test_sources(root, commands, expected):
    actual = {}
    test_commands = []
    for command in commands:
        tests_owned_sources = [
            source for source in command["sources"]
            if source_is_tests_owned(root, source)
        ]
        for source in tests_owned_sources:
            identity = canonical_path_identity(source["real"])
            actual[identity] = source["relative"]
        if tests_owned_sources:
            test_commands.append(command)
    missing = sorted(
        expected[identity] for identity in set(expected) - set(actual)
    )
    extra = sorted(
        actual[identity] for identity in set(actual) - set(expected)
    )
    if missing or extra:
        details = []
        if missing:
            details.append("missing from recipes: {0}".format(", ".join(missing)))
        if extra:
            details.append("missing from manifest: {0}".format(", ".join(extra)))
        raise AnalysisError("test source manifest mismatch; {0}".format("; ".join(details)))
    compile_contexts = []
    for command in test_commands:
        for source in command["sources"]:
            compile_contexts.append({
                "source": source["lexical"],
                "quote_dirs": command["quote_dirs"],
                "include_dirs": command["include_dirs"],
                "system_dirs": command["system_dirs"],
                "forced_inputs": command["forced_inputs"],
                "production": False,
            })
    return compile_contexts


def active_visit_key(path, quote_dirs, include_dirs, system_dirs, production):
    lexical_path = os.path.abspath(os.path.normpath(path))
    return (
        lexical_path,
        tuple(quote_dirs),
        tuple(include_dirs),
        tuple(system_dirs),
        production,
    )


def collect_active_inputs(root, compile_contexts):
    active = {}
    visited = set()
    recursion_stack = set()
    tests_root = canonical_path_identity(os.path.join(root, "tests"))

    def visit(path, quote_dirs, include_dirs, system_dirs, production):
        lexical_path = os.path.abspath(os.path.normpath(path))
        key = active_visit_key(
            lexical_path, quote_dirs, include_dirs, system_dirs, production
        )
        real_path = os.path.realpath(lexical_path)
        if not is_within(root, real_path):
            raise AnalysisError(
                "active include escapes source root: {0}".format(lexical_path)
            )
        real_identity = canonical_path_identity(real_path)
        if production and is_within(tests_root, real_identity):
            raise AnalysisError(
                "production uses tests-owned compiler input: {0}".format(
                    display_path(root, lexical_path)
                )
            )
        if key in visited:
            return
        # An unguarded recursive include cannot compile. Stopping a repeated
        # canonical file only in this chain models guarded/pragma-once
        # termination without hiding content from later independent routes.
        if real_identity in recursion_stack:
            return
        visited.add(key)
        active.setdefault(real_identity, real_path)
        recursion_stack.add(real_identity)
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
                        search_dirs = ([os.path.dirname(lexical_path)] + quote_dirs +
                                       include_dirs + system_dirs)
                    else:
                        search_dirs = include_dirs + system_dirs
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
                    visit(
                        resolved, quote_dirs, include_dirs, system_dirs,
                        production
                    )
        finally:
            recursion_stack.remove(real_identity)

    def resolve_forced_input(context, forced_input):
        if os.path.isabs(forced_input):
            candidates = [forced_input]
        else:
            direct = os.path.abspath(os.path.normpath(os.path.join(root, forced_input)))
            if not is_within(root, direct):
                raise AnalysisError(
                    "forced compiler input escapes source root: {0}".format(
                        forced_input
                    )
                )
            candidates = [direct]
            candidates.extend(
                os.path.join(directory, forced_input)
                for directory in (context["quote_dirs"] +
                                  context["include_dirs"] +
                                  context["system_dirs"])
            )
        for candidate in candidates:
            lexical = os.path.abspath(os.path.normpath(candidate))
            if not os.path.exists(lexical):
                continue
            real = os.path.realpath(lexical)
            if not is_within(root, lexical) or not is_within(root, real):
                raise AnalysisError(
                    "forced compiler input escapes source root: {0}".format(
                        forced_input
                    )
                )
            if not os.path.isfile(real):
                raise AnalysisError(
                    "forced compiler input is not a file: {0}".format(forced_input)
                )
            return lexical
        raise AnalysisError(
            "cannot resolve forced compiler input: {0}".format(forced_input)
        )

    for context in compile_contexts:
        for forced_input in context["forced_inputs"]:
            forced_path = resolve_forced_input(context, forced_input)
            visit(
                forced_path, context["quote_dirs"], context["include_dirs"],
                context["system_dirs"], context["production"]
            )
        visit(
            context["source"], context["quote_dirs"], context["include_dirs"],
            context["system_dirs"], context["production"]
        )
    return set(active.values())


def build_artifact_graph(root, commands):
    producers = {}
    for command in commands:
        output = command["output"]
        if output is None:
            continue
        identity = canonical_path_identity(output)
        if identity in producers:
            raise AnalysisError(
                "multiple recipe commands produce artifact: {0}".format(
                    display_path(root, output)
                )
            )
        producers[identity] = command

    for command in commands:
        inputs = []
        input_identities = set()
        output_identity = (
            canonical_path_identity(command["output"])
            if command["output"] is not None else None
        )
        for candidate in command["candidate_inputs"]:
            identity = canonical_path_identity(candidate)
            if identity != output_identity and identity not in input_identities:
                inputs.append({"identity": identity, "path": candidate})
                input_identities.add(identity)
        command["artifact_inputs"] = inputs

    application_names = set(
        canonical_path_identity(os.path.join(root, path))
        for path in ("build/lenet_cuda", "build/lenet_cuda.exe")
    )
    applications = [
        (identity, command["output"]) for identity, command in producers.items()
        if identity in application_names
    ]
    if len(applications) != 1:
        raise AnalysisError(
            "Make recipes must expose exactly one build/lenet_cuda[.exe] output; "
            "found {0}".format(len(applications))
        )

    visited = set()
    reachable_commands = []

    def walk(identity, artifact, chain):
        if identity in visited:
            return
        visited.add(identity)
        command = producers.get(identity)
        if command is None:
            raise AnalysisError(
                "production graph has an input with no analyzed producer: {0}".format(
                    display_path(root, artifact)
                )
            )
        reachable_commands.append(command)
        if command["input_metadata"]["stateful_archive"]:
            raise AnalysisError(
                "unsupported stateful archive provenance affects lenet_cuda: {0}".format(
                    display_path(root, artifact)
                )
            )
        if command["input_metadata"]["response_file"]:
            raise AnalysisError(
                "unsupported response-file input affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        if command["input_metadata"]["external_control"]:
            raise AnalysisError(
                "unsupported external control input affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        if command["input_metadata"]["library_input"]:
            raise AnalysisError(
                "unsupported linker library input affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        if command["input_metadata"]["linker_input"]:
            raise AnalysisError(
                "unsupported linker-encoded input affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        if command["input_metadata"]["unsupported_shell"]:
            raise AnalysisError(
                "unsupported obfuscated shell recipe affects lenet_cuda: {0}".format(
                    command["line"]
                )
            )
        if command["invalid_candidate_inputs"]:
            raise AnalysisError(command["invalid_candidate_inputs"][0])
        for source in command["sources"]:
            if source_is_tests_owned(root, source):
                rendered = [display_path(root, item) for item in chain]
                raise AnalysisError(
                    "production lenet_cuda graph includes tests-owned source {0} via {1}".format(
                        source["relative"], " -> ".join(rendered)
                    )
                )
        for dependency in command["artifact_inputs"]:
            walk(
                dependency["identity"], dependency["path"],
                chain + [dependency["path"]]
            )

    application_identity, application = applications[0]
    walk(application_identity, application, [application])
    return reachable_commands


def analyze_source(args):
    root = os.path.realpath(os.path.abspath(args.root))
    commands = parse_recipes(root, args.recipes)
    expected = read_manifest(root, args.manifest)
    contexts = reconcile_test_sources(root, commands, expected)
    reachable_commands = build_artifact_graph(root, commands)
    for command in reachable_commands:
        for source in command["sources"]:
            contexts.append({
                "source": source["lexical"],
                "quote_dirs": command["quote_dirs"],
                "include_dirs": command["include_dirs"],
                "system_dirs": command["system_dirs"],
                "forced_inputs": command["forced_inputs"],
                "production": True,
            })
    active = collect_active_inputs(root, contexts)
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
    if len(links) != 1:
        raise AnalysisError(
            "build log must contain exactly one lenet_cuda linker command; "
            "found {0}".format(len(links))
        )
    link = links[0]
    if not all(flag in link["tokens"] for flag in REQUIRED_GENCODE):
        raise AnalysisError(
            "lenet_cuda linker command is missing required compute_90 architecture flags"
        )
    if link["input_metadata"]["response_file"]:
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
