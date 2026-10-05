#!/usr/bin/env python3
from __future__ import print_function

import argparse
import os
import re
import sys


# Compiled patterns and keyword sets for the signature inventory.
CHECKLIST_ENTRY = re.compile(r"^\s*- \[([ xX])\] `([^`]+)`\s*$")
TYPE_DECLARATION = re.compile(
    r"(?:^|\s)(class|struct|enum(?:\s+class)?)\s+([A-Za-z_]\w*)\b"
)
ALIAS_DECLARATION = re.compile(r"(?:^|\s)using\s+([A-Za-z_]\w*)\s*=")
NAMESPACE_DECLARATION = re.compile(r"(?:^|\s)namespace\s+([A-Za-z_]\w*)\s*$")
FUNCTION_NAME = re.compile(
    r"(~?[A-Za-z_]\w*|operator\s*(?:\(\)|\[\]|[=!<>+\-*/%&|^~]+))\s*\("
)
QUALIFIED_DEFINITION = re.compile(
    r"[A-Za-z_]\w*(?:\s*<[^>{}]+>)?::(?:~?[A-Za-z_]\w*|operator\s*[^\s(]+)\s*\("
)
IGNORED_FUNCTION_NAMES = set(
    (
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "alignas",
        "noexcept",
        "__launch_bounds__",
    )
)
MANUAL_ID = re.compile(r"^manual:([^:]+):([A-Za-z_]\w*)$")


def normalized_path(path):
    return path.replace(os.sep, "/")


# Blanks C++ comments and string or character literals while keeping line structure.
def sanitized_cpp(text):
    output = []
    index = 0
    state = "code"
    quote = None
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if state == "line-comment":
            if char == "\n":
                output.append(char)
                state = "code"
            else:
                output.append(" ")
        elif state == "block-comment":
            if char == "*" and next_char == "/":
                output.extend((" ", " "))
                index += 1
                state = "code"
            elif char == "\n":
                output.append(char)
            else:
                output.append(" ")
        elif state == "string":
            if char == "\\" and next_char:
                output.extend((" ", " "))
                index += 1
            elif char == quote:
                output.append(" ")
                state = "code"
            elif char == "\n":
                output.append(char)
            else:
                output.append(" ")
        elif char == "/" and next_char == "/":
            output.extend((" ", " "))
            index += 1
            state = "line-comment"
        elif char == "/" and next_char == "*":
            output.extend((" ", " "))
            index += 1
            state = "block-comment"
        elif char in ("'", '"'):
            output.append(" ")
            quote = char
            state = "string"
        else:
            output.append(char)
        index += 1

    cleaned_lines = "".join(output).splitlines(True)
    continuation = False
    for line_index, line in enumerate(cleaned_lines):
        if continuation or line.lstrip().startswith("#"):
            continuation = line.rstrip().endswith("\\")
            cleaned_lines[line_index] = "\n" if line.endswith("\n") else ""
        else:
            continuation = False
    return "".join(cleaned_lines)


# Checks whether a declaration has an adjacent comment or block comment above it.
def has_adjacent_documentation(lines, index):
    index -= 1
    if index < 0 or not lines[index].strip():
        return False
    stripped = lines[index].lstrip()
    if stripped.startswith("//"):
        return True
    if stripped.rstrip().endswith("*/"):
        while index >= 0:
            if "/*" in lines[index]:
                return True
            index -= 1
    return False


# Canonicalizes parameter lists and suffixes into stable signature IDs.
def canonical_parameters(parameters):
    value = re.sub(r"\s+", " ", parameters.strip())
    value = re.sub(r"\s*([,*&<>\[\]=])\s*", r"\1", value)
    return value


def canonical_suffix(suffix):
    value = re.sub(r"=\s*(?:delete|default)\s*$", "", suffix.strip())
    if value.startswith(":"):
        value = ""
    value = re.sub(r"\b(?:override|final)\b", "", value)
    return re.sub(r"\s+", "", value)


def matching_parenthesis(text, opening):
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def function_signature(segment):
    if QUALIFIED_DEFINITION.search(segment):
        return None
    for match in FUNCTION_NAME.finditer(segment):
        name = re.sub(r"\s+", "", match.group(1))
        if name in IGNORED_FUNCTION_NAMES:
            continue
        prefix = segment[: match.start()].rstrip()
        if prefix.endswith("::"):
            continue
        opening = match.end() - 1
        closing = matching_parenthesis(segment, opening)
        if closing is None:
            continue
        parameters = canonical_parameters(segment[opening + 1 : closing])
        suffix = canonical_suffix(segment[closing + 1 :])
        return name + "(" + parameters + ")" + suffix
    return None


def declaration_visible(contexts):
    for context in contexts:
        if context["kind"] == "function" or context["kind"] == "other":
            return False
        if context["kind"] == "type" and (
            not context["visible"] or context["access"] != "public"
        ):
            return False
    return True


def qualified_name(contexts, name):
    scope = [
        context["name"]
        for context in contexts
        if context["kind"] in ("namespace", "type") and context["name"]
    ]
    scope.append(name)
    return "::".join(scope)


def declaration_line(text, segment_offset, segment):
    first = re.search(r"\S", segment)
    offset = segment_offset + (first.start() if first else 0)
    return text.count("\n", 0, offset)


# Scans a header for namespaces, types, aliases, and function declarations.
def scan_header(path, relative):
    with open(path, "r") as input_file:
        original = input_file.read()
    text = sanitized_cpp(original)
    lines = original.splitlines(True)
    occurrences = []
    contexts = []
    segment = []
    segment_offset = 0
    parenthesis_depth = 0

    def record(name, raw_segment, role):
        identifier = "public:{0}:{1}".format(
            relative, qualified_name(contexts, name)
        )
        line = declaration_line(text, segment_offset, raw_segment)
        occurrences.append(
            (identifier, line, role, has_adjacent_documentation(lines, line))
        )

    def process(raw_segment, opens_scope):
        stripped = raw_segment.strip()
        if not stripped:
            return {"kind": "other", "name": "", "visible": False}
        visible = declaration_visible(contexts)
        namespace_match = NAMESPACE_DECLARATION.search(stripped)
        if opens_scope and namespace_match:
            return {
                "kind": "namespace",
                "name": namespace_match.group(1),
                "visible": visible,
            }
        type_match = TYPE_DECLARATION.search(stripped)
        if type_match:
            name = type_match.group(2)
            if visible:
                record(name, raw_segment, "definition" if opens_scope else "forward")
            if opens_scope:
                return {
                    "kind": "type",
                    "name": name,
                    "visible": visible,
                    "access": "public" if type_match.group(1) != "class" else "private",
                }
            return None
        alias_match = ALIAS_DECLARATION.search(stripped)
        if alias_match and visible:
            record(alias_match.group(1), raw_segment, "declaration")
            return None
        signature = function_signature(stripped)
        if signature is not None:
            if visible:
                record(signature, raw_segment, "definition" if opens_scope else "declaration")
            if opens_scope:
                return {
                    "kind": "function",
                    "name": "",
                    "visible": visible,
                }
            return None
        if opens_scope and (
            stripped.startswith('extern "C"') or stripped == "extern" or stripped == ""
        ):
            return {"kind": "namespace", "name": "", "visible": visible}
        if opens_scope:
            return {"kind": "other", "name": "", "visible": False}
        return None

    # Track parenthesis depth, access specifiers, and brace scope while scanning.
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            parenthesis_depth += 1
        elif char == ")" and parenthesis_depth > 0:
            parenthesis_depth -= 1

        if parenthesis_depth == 0 and char == ":":
            access = "".join(segment).strip()
            if access in ("public", "private", "protected"):
                for context in reversed(contexts):
                    if context["kind"] == "type":
                        context["access"] = access
                        break
                segment = []
                segment_offset = index + 1
                index += 1
                continue
        if parenthesis_depth == 0 and char in "{};":
            raw_segment = "".join(segment)
            if char == "{":
                context = process(raw_segment, True)
                contexts.append(
                    context
                    if context is not None
                    else {"kind": "other", "name": "", "visible": False}
                )
            elif char == "}":
                if contexts:
                    contexts.pop()
            else:
                process(raw_segment, False)
            segment = []
            segment_offset = index + 1
        else:
            segment.append(char)
        index += 1
    grouped = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence[0], []).append(occurrence)
    found = set()
    missing_comments = set()
    for base, values in grouped.items():
        role_counts = {}
        for value in values:
            role_counts[value[2]] = role_counts.get(value[2], 0) + 1
        for _, line, role, documented in values:
            identifier = base
            if len(values) > 1:
                identifier += "@" + role
                if role_counts[role] > 1:
                    identifier += ":" + str(line + 1)
            found.add(identifier)
            if not documented:
                missing_comments.add(identifier)
    return found, missing_comments


# Finds global CUDA kernels and checks them for adjacent documentation.
def scan_kernels(path, relative):
    with open(path, "r") as input_file:
        original = input_file.read()
    text = sanitized_cpp(original)
    lines = original.splitlines(True)
    found = set()
    missing_comments = set()
    for marker in re.finditer(r"\b__global__\b", text):
        tail = text[marker.end() :]
        opening_brace = tail.find("{")
        if opening_brace < 0:
            continue
        signature = function_signature(tail[:opening_brace])
        if signature is None:
            continue
        stable_id = "kernel:{0}:{1}".format(relative, signature)
        found.add(stable_id)
        line = text.count("\n", 0, marker.start())
        while line > 0 and lines[line - 1].lstrip().startswith("[["):
            line -= 1
        if not has_adjacent_documentation(lines, line):
            missing_comments.add(stable_id)
    return found, missing_comments


# Walks include and source trees to collect declarations and kernels.
def discover(root):
    found = set()
    missing_comments = set()
    include_root = os.path.join(root, "include")
    if os.path.isdir(include_root):
        for directory, names, files in os.walk(include_root):
            names.sort()
            for filename in sorted(files):
                if not filename.endswith((".h", ".hpp", ".cuh")):
                    continue
                path = os.path.join(directory, filename)
                relative = normalized_path(os.path.relpath(path, root))
                declarations, missing = scan_header(path, relative)
                found.update(declarations)
                missing_comments.update(missing)

    source_root = os.path.join(root, "src")
    if os.path.isdir(source_root):
        for directory, names, files in os.walk(source_root):
            names.sort()
            for filename in sorted(files):
                if not filename.endswith((".cu", ".cuh")):
                    continue
                path = os.path.join(directory, filename)
                relative = normalized_path(os.path.relpath(path, root))
                kernels, missing = scan_kernels(path, relative)
                found.update(kernels)
                missing_comments.update(missing)
    return found, missing_comments


# Parses the checklist into identifiers mapped to their review state.
def read_checklist(path):
    entries = {}
    with open(path, "r") as input_file:
        for line_number, line in enumerate(input_file, 1):
            match = CHECKLIST_ENTRY.match(line.rstrip("\n"))
            if not match:
                continue
            identifier = match.group(2)
            if identifier in entries:
                raise ValueError(
                    "duplicate checklist ID {0} at line {1}".format(
                        identifier, line_number
                    )
                )
            entries[identifier] = match.group(1).lower() == "x"
    return entries


# Counts matching source definitions for a manual checklist identifier.
def manual_definition_count(root, identifier):
    match = MANUAL_ID.match(identifier)
    if not match:
        return 0
    relative, symbol = match.groups()
    path = os.path.abspath(os.path.join(root, relative.replace("/", os.sep)))
    if os.path.commonpath((root, path)) != root or not os.path.isfile(path):
        return 0
    with open(path, "r") as input_file:
        text = input_file.read()
    if path.endswith(".py"):
        pattern = re.compile(r"^\s*def\s+{0}\s*\(".format(re.escape(symbol)), re.M)
    else:
        pattern = re.compile(
            r"\b{0}\s*\([^;{{}}]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{{".format(
                re.escape(symbol)
            ),
            re.S,
        )
    return len(pattern.findall(sanitized_cpp(text)))


# Reconciles discovered IDs, checklist entries, and required reviews.
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check public declarations and CUDA kernels for documentation"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--checklist", required=True)
    parser.add_argument("--require-reviewed", action="store_true")
    arguments = parser.parse_args(argv)

    root = os.path.abspath(arguments.root)
    checklist = os.path.abspath(arguments.checklist)
    errors = []
    found, missing_comments = discover(root)
    try:
        entries = read_checklist(checklist)
    except (IOError, OSError, ValueError) as error:
        print("comment check failed: {0}".format(error), file=sys.stderr)
        return 1

    for identifier in sorted(missing_comments):
        errors.append("missing adjacent documentation: " + identifier)

    discovered_ids = set(found)
    checklist_ids = set(
        identifier
        for identifier in entries
        if identifier.startswith(("public:", "kernel:"))
    )
    for identifier in sorted(found - checklist_ids):
        errors.append("missing checklist ID: " + identifier)
    for identifier in sorted(checklist_ids - found):
        errors.append("stale checklist ID: " + identifier)

    for identifier in sorted(entries):
        if identifier.startswith("manual:"):
            count = manual_definition_count(root, identifier)
            if count == 0:
                errors.append("stale checklist ID: " + identifier)
            elif count != 1:
                errors.append(
                    "manual checklist ID must resolve to exactly one definition: "
                    + identifier
                )
            else:
                discovered_ids.add(identifier)
        elif not identifier.startswith(("public:", "kernel:")):
            errors.append("unsupported checklist ID: " + identifier)

    if arguments.require_reviewed:
        for identifier in sorted(entries):
            if not entries[identifier]:
                errors.append("checklist item not reviewed: " + identifier)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "comment check passed: inventory={0} checklist_items={1}".format(
            len(discovered_ids), len(entries)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
