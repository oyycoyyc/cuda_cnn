import os
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "scripts", "check_comments.py")


class CommentCheckerTest(unittest.TestCase):
    # Owns a throwaway project tree with include, src, and docs directories.
    def setUp(self):
        self.temporary = tempfile.mkdtemp(prefix="lenet-comments-")
        os.makedirs(os.path.join(self.temporary, "include"))
        os.makedirs(os.path.join(self.temporary, "src"))
        os.makedirs(os.path.join(self.temporary, "docs"))

    def tearDown(self):
        shutil.rmtree(self.temporary)

    # Writes a fixture file into the owned temporary project tree.
    def write(self, relative_path, content):
        path = os.path.join(self.temporary, relative_path)
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path, "w") as output:
            output.write(content)
        return path

    # Invokes the comment checker against the owned tree and captures output.
    def run_checker(self, checklist="docs/checklist.md", require_reviewed=False):
        command = [
            sys.executable,
            CHECKER,
            "--root",
            self.temporary,
            "--checklist",
            os.path.join(self.temporary, checklist),
        ]
        if require_reviewed:
            command.append("--require-reviewed")
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    # Builds the documented positive fixture that the stable-ID tests reuse.
    def documented_fixture(self):
        self.write(
            "include/api.h",
            "// Owns one documented value.\n"
            "struct Widget {\n"
            "  // Runs the documented operation without retaining input.\n"
            "  void Run(\n"
            "      const float* input,\n"
            "      float* output) const;\n"
            "};\n\n"
            "namespace sample {\n"
            "// Returns a documented namespaced value.\n"
            "int Make(int value);\n"
            "}  // namespace sample\n\n"
            "// Launches the documented device operation on caller storage.\n"
            "void LaunchThing(\n"
            "    const float* input, float* output,\n"
            "    int count);\n",
        )
        self.write(
            "src/kernel.cu",
            "// Computes output[i]=input[i]. Each thread owns linear i; the "
            "layout is contiguous.\n"
            "// There is no accumulation or race. Out-of-range threads return; "
            "no atomics,\n"
            "// synchronization, or special numerical stabilization is needed.\n"
            "[[maybe_unused]] __global__ __launch_bounds__(128) void "
            "ExampleKernel(const float* input, float* output, "
            "int count) {\n"
            "  const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < count) output[i] = input[i];\n"
            "}\n",
        )
        self.write(
            "docs/checklist.md",
            "# Review\n\n"
            "- [ ] `public:include/api.h:Widget`\n"
            "- [ ] `public:include/api.h:Widget::Run(const float*input,float*output)const`\n"
            "- [ ] `public:include/api.h:sample::Make(int value)`\n"
            "- [ ] `public:include/api.h:LaunchThing(const float*input,float*output,int count)`\n"
            "- [ ] `kernel:src/kernel.cu:ExampleKernel(const float*input,float*output,int count)`\n",
        )

    # Documented types, methods, launchers, and kernels all pass.
    def test_multiline_types_methods_launchers_and_kernels_pass(self):
        self.documented_fixture()
        result = self.run_checker()
        self.assertEqual(0, result.returncode, result.stdout)

    # Declarations must carry an adjacent comment to be documented.
    def test_each_declaration_kind_requires_an_adjacent_comment(self):
        cases = (
            ("struct MissingType {};\n", "public:include/api.h:MissingType"),
            (
                "// Type docs.\nstruct Widget {\n  void MissingMethod() const;\n};\n",
                "public:include/api.h:Widget::MissingMethod()const",
            ),
            (
                "void LaunchMissing(\n    const float* input, float* output);\n",
                "public:include/api.h:LaunchMissing(const float*input,float*output)",
            ),
        )
        for source, identifier in cases:
            with self.subTest(identifier=identifier):
                self.write("include/api.h", source)
                self.write("docs/checklist.md", "- [ ] `{0}`\n".format(identifier))
                result = self.run_checker()
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn(identifier, result.stdout)

    # Kernels and other global definitions must also be adjacent-commented.
    def test_every_global_definition_requires_an_adjacent_comment(self):
        self.write(
            "src/kernel.cu",
            "__global__ void MissingKernel(float* values) { values[0] = 0; }\n",
        )
        identifier = "kernel:src/kernel.cu:MissingKernel(float*values)"
        self.write("docs/checklist.md", "- [ ] `{0}`\n".format(identifier))
        result = self.run_checker()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn(identifier, result.stdout)

    def test_detached_comment_does_not_document_a_declaration(self):
        self.write(
            "include/api.h",
            "// This comment is detached.\n\nvoid Detached();\n",
        )
        identifier = "public:include/api.h:Detached()"
        self.write("docs/checklist.md", "- [ ] `{0}`\n".format(identifier))
        result = self.run_checker()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn(identifier, result.stdout)

    # Checklist IDs must match all discovered stable IDs, with no stale entries.
    def test_checklist_must_match_all_discovered_stable_ids(self):
        self.documented_fixture()
        checklist = os.path.join(self.temporary, "docs", "checklist.md")
        with open(checklist, "r") as input_file:
            text = input_file.read()
        with open(checklist, "w") as output:
            output.write(text.replace(
                "- [ ] `public:include/api.h:LaunchThing(const float*input,float*output,int count)`\n", ""
            ))
            output.write("- [ ] `kernel:src/kernel.cu:StaleKernel()`\n")
        result = self.run_checker()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("missing checklist ID", result.stdout)
        self.assertIn("stale checklist ID", result.stdout)

    # Overloads must have distinct signature-based stable IDs.
    def test_overloads_have_distinct_signature_ids(self):
        self.write(
            "include/api.h",
            "// Computes from an integer.\n"
            "int Compute(int value);\n"
            "// Computes from a float.\n"
            "int Compute(float value);\n",
        )
        self.write(
            "docs/checklist.md",
            "- [ ] `public:include/api.h:Compute(int value)`\n"
            "- [ ] `public:include/api.h:Compute(float value)`\n",
        )
        result = self.run_checker()
        self.assertEqual(0, result.returncode, result.stdout)

    # Nested public types and methods are inventoried as qualified IDs.
    def test_nested_public_type_and_method_are_inventoried(self):
        self.write(
            "include/api.h",
            "// Owns nested API declarations.\n"
            "class Outer {\n"
            " public:\n"
            "  // Carries one nested public value.\n"
            "  struct Inner {\n"
            "    // Returns the nested value.\n"
            "    int Value() const;\n"
            "  };\n"
            "};\n",
        )
        self.write(
            "docs/checklist.md",
            "- [ ] `public:include/api.h:Outer`\n"
            "- [ ] `public:include/api.h:Outer::Inner`\n"
            "- [ ] `public:include/api.h:Outer::Inner::Value()const`\n",
        )
        result = self.run_checker()
        self.assertEqual(0, result.returncode, result.stdout)

    # Repeated type declarations get distinct role-qualified IDs.
    def test_repeated_type_declarations_have_distinct_role_ids(self):
        self.write(
            "include/api.h",
            "// Declares Widget before its definition.\n"
            "class Widget;\n"
            "// Defines the public Widget type.\n"
            "class Widget {};\n",
        )
        self.write(
            "docs/checklist.md",
            "- [ ] `public:include/api.h:Widget@forward`\n"
            "- [ ] `public:include/api.h:Widget@definition`\n",
        )
        result = self.run_checker()
        self.assertEqual(0, result.returncode, result.stdout)

    # Manual entries must resolve to exactly one source definition.
    def test_manual_entries_must_resolve_to_exactly_one_source_definition(self):
        self.write(
            "src/parser.cpp",
            "// Parses the fixture.\nvoid ParseThing() {}\n",
        )
        self.write(
            "docs/checklist.md",
            "- [ ] `manual:src/parser.cpp:ParseThing`\n",
        )
        valid = self.run_checker()
        self.assertEqual(0, valid.returncode, valid.stdout)

        self.write(
            "docs/checklist.md",
            "- [ ] `manual:src/parser.cpp:MissingThing`\n",
        )
        stale = self.run_checker()
        self.assertNotEqual(0, stale.returncode, stale.stdout)
        self.assertIn("stale checklist ID", stale.stdout)

        self.write(
            "src/parser.cpp",
            "void ParseThing() {}\nvoid ParseThing(int value) {}\n",
        )
        self.write(
            "docs/checklist.md",
            "- [ ] `manual:src/parser.cpp:ParseThing`\n",
        )
        duplicate = self.run_checker()
        self.assertNotEqual(0, duplicate.returncode, duplicate.stdout)
        self.assertIn("exactly one", duplicate.stdout)

    # The review gate only passes when every entry is checked off.
    def test_manual_review_gate_requires_checked_entries(self):
        self.documented_fixture()
        unchecked = self.run_checker(require_reviewed=True)
        self.assertNotEqual(0, unchecked.returncode, unchecked.stdout)
        self.assertIn("not reviewed", unchecked.stdout)
        checklist = os.path.join(self.temporary, "docs", "checklist.md")
        with open(checklist, "r") as input_file:
            text = input_file.read()
        with open(checklist, "w") as output:
            output.write(text.replace("- [ ]", "- [x]"))
        reviewed = self.run_checker(require_reviewed=True)
        self.assertEqual(0, reviewed.returncode, reviewed.stdout)

    # The real project inventory must stay in sync with its checklist.
    def test_project_inventory_and_checklist_are_in_sync(self):
        result = subprocess.run(
            [
                sys.executable,
                CHECKER,
                "--root",
                ROOT,
                "--checklist",
                os.path.join(ROOT, "docs", "comment-review-checklist.md"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertEqual(0, result.returncode, result.stdout)


if __name__ == "__main__":
    unittest.main()
