#!/usr/bin/env python3
from __future__ import print_function

import argparse
import os
import re
import sys


TRACKED_ID = re.compile(r"^(?:public|kernel):")
CHECKLIST_ENTRY = re.compile(r"^\s*- \[([ xX])\] `([^`]+)`\s*$")
TYPE_DECLARATION = re.compile(
    r"^\s*(?:template\s*<[^>]+>\s*)?(?:class|struct|enum(?:\s+class)?)\s+(\w+)\b[^;]*\{"
)
ALIAS_DECLARATION = re.compile(r"^\s*using\s+(\w+)\s*=")
FUNCTION_NAME = re.compile(
    r"(~?[A-Za-z_]\w*|operator\s*(?:=|\(\)|\[\]))\s*\("
)
KERNEL_DEFINITION = re.compile(r"__global__\s+void\s+(\w+)\s*\(")
QUALIFIED_DEFINITION = re.compile(
    r"[A-Za-z_]\w*(?:<[^>]+>)?::(?:~?[A-Za-z_]\w*|operator\s*=)\s*\("
)


def normalized_path(path):
    return path.replace(os.sep, "/")


def code_without_line_comment(line):
    return line.split("//", 1)[0]


def brace_delta(line):
    code = code_without_line_comment(line)
    return code.count("{") - code.count("}")


def declaration_start(lines, index):
    if index > 0 and lines[index - 1].lstrip().startswith("template"):
        return index - 1
    return index


def has_adjacent_documentation(lines, index):
    index = declaration_start(lines, index) - 1
    if index < 0 or not lines[index].strip():
        return False
    stripped = lines[index].lstrip()
    if stripped.startswith("//"):
        return True
    if stripped.endswith("*/"):
        while index >= 0:
            if "/*" in lines[index]:
                return True
            index -= 1
    return False


def function_name(line):
    matches = list(FUNCTION_NAME.finditer(code_without_line_comment(line)))
    if not matches:
        return None
    name = re.sub(r"\s+", "", matches[0].group(1))
    if name in ("if", "for", "while", "switch", "return", "sizeof"):
        return None
    return name


def add_declaration(found, missing_comments, relative, identifier, lines, index):
    stable_id = "public:{0}:{1}".format(relative, identifier)
    found.add(stable_id)
    if not has_adjacent_documentation(lines, index):
        missing_comments.add(stable_id)


def scan_header(path, relative):
    with open(path, "r") as input_file:
        lines = input_file.readlines()

    found = set()
    missing_comments = set()
    depth = 0
    class_name = None
    class_depth = None
    access = None

    for index, line in enumerate(lines):
        stripped_code = code_without_line_comment(line).strip()

        if stripped_code.startswith("#") or stripped_code.startswith(":"):
            depth += brace_delta(line)
            continue

        if class_name is None and depth == 0:
            type_match = TYPE_DECLARATION.match(stripped_code)
            if type_match:
                class_name = type_match.group(1)
                class_depth = depth + 1
                access = "public" if stripped_code.startswith("struct") else "private"
                add_declaration(
                    found, missing_comments, relative, class_name, lines, index
                )
            else:
                alias_match = ALIAS_DECLARATION.match(stripped_code)
                if alias_match:
                    add_declaration(
                        found,
                        missing_comments,
                        relative,
                        alias_match.group(1),
                        lines,
                        index,
                    )
                elif not QUALIFIED_DEFINITION.search(stripped_code):
                    name = function_name(stripped_code)
                    if name is not None:
                        add_declaration(
                            found, missing_comments, relative, name, lines, index
                        )
        elif class_name is not None and depth == class_depth:
            access_match = re.match(r"^(public|private|protected)\s*:\s*$", stripped_code)
            if access_match:
                access = access_match.group(1)
            elif access == "public":
                name = function_name(stripped_code)
                if name is not None:
                    add_declaration(
                        found,
                        missing_comments,
                        relative,
                        class_name + "::" + name,
                        lines,
                        index,
                    )

        depth += brace_delta(line)
        if class_name is not None and depth < class_depth:
            class_name = None
            class_depth = None
            access = None

    return found, missing_comments


def scan_kernels(path, relative):
    with open(path, "r") as input_file:
        text = input_file.read()
    lines = text.splitlines(True)
    found = set()
    missing_comments = set()
    for match in KERNEL_DEFINITION.finditer(text):
        index = text.count("\n", 0, match.start())
        stable_id = "kernel:{0}:{1}".format(relative, match.group(1))
        found.add(stable_id)
        if not has_adjacent_documentation(lines, index):
            missing_comments.add(stable_id)
    return found, missing_comments


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

    checklist_ids = set(identifier for identifier in entries if TRACKED_ID.match(identifier))
    for identifier in sorted(found - checklist_ids):
        errors.append("missing checklist ID: " + identifier)
    for identifier in sorted(checklist_ids - found):
        errors.append("stale checklist ID: " + identifier)

    if arguments.require_reviewed:
        for identifier in sorted(entries):
            if not entries[identifier]:
                errors.append("checklist item not reviewed: " + identifier)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "comment check passed: declarations_and_kernels={0} checklist_items={1}".format(
            len(found), len(entries)
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
