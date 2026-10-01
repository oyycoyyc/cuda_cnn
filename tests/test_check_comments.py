import os
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "scripts", "check_comments.py")


class CommentCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.mkdtemp(prefix="lenet-comments-")
        os.makedirs(os.path.join(self.temporary, "include"))
        os.makedirs(os.path.join(self.temporary, "src"))
        os.makedirs(os.path.join(self.temporary, "docs"))

    def tearDown(self):
        shutil.rmtree(self.temporary)

    def write(self, relative_path, content):
        path = os.path.join(self.temporary, relative_path)
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path, "w") as output:
            output.write(content)
        return path

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
            "__global__ void ExampleKernel(const float* input, float* output, "
            "int count) {\n"
            "  const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < count) output[i] = input[i];\n"
            "}\n",
        )
        self.write(
            "docs/checklist.md",
            "# Review\n\n"
            "- [ ] `public:include/api.h:Widget`\n"
            "- [ ] `public:include/api.h:Widget::Run`\n"
            "- [ ] `public:include/api.h:LaunchThing`\n"
            "- [ ] `kernel:src/kernel.cu:ExampleKernel`\n",
        )

    def test_multiline_types_methods_launchers_and_kernels_pass(self):
        self.documented_fixture()
        result = self.run_checker()
        self.assertEqual(0, result.returncode, result.stdout)

    def test_each_declaration_kind_requires_an_adjacent_comment(self):
        cases = (
            ("struct MissingType {};\n", "public:include/api.h:MissingType"),
            (
                "// Type docs.\nstruct Widget {\n  void MissingMethod() const;\n};\n",
                "public:include/api.h:Widget::MissingMethod",
            ),
            (
                "void LaunchMissing(\n    const float* input, float* output);\n",
                "public:include/api.h:LaunchMissing",
            ),
        )
        for source, identifier in cases:
            with self.subTest(identifier=identifier):
                self.write("include/api.h", source)
                self.write("docs/checklist.md", "- [ ] `{0}`\n".format(identifier))
                result = self.run_checker()
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn(identifier, result.stdout)

    def test_every_global_definition_requires_an_adjacent_comment(self):
        self.write(
            "src/kernel.cu",
            "__global__ void MissingKernel(float* values) { values[0] = 0; }\n",
        )
        identifier = "kernel:src/kernel.cu:MissingKernel"
        self.write("docs/checklist.md", "- [ ] `{0}`\n".format(identifier))
        result = self.run_checker()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn(identifier, result.stdout)

    def test_detached_comment_does_not_document_a_declaration(self):
        self.write(
            "include/api.h",
            "// This comment is detached.\n\nvoid Detached();\n",
        )
        identifier = "public:include/api.h:Detached"
        self.write("docs/checklist.md", "- [ ] `{0}`\n".format(identifier))
        result = self.run_checker()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn(identifier, result.stdout)

    def test_checklist_must_match_all_discovered_stable_ids(self):
        self.documented_fixture()
        checklist = os.path.join(self.temporary, "docs", "checklist.md")
        with open(checklist, "r") as input_file:
            text = input_file.read()
        with open(checklist, "w") as output:
            output.write(text.replace(
                "- [ ] `public:include/api.h:LaunchThing`\n", ""
            ))
            output.write("- [ ] `kernel:src/kernel.cu:StaleKernel`\n")
        result = self.run_checker()
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("missing checklist ID", result.stdout)
        self.assertIn("stale checklist ID", result.stdout)

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
