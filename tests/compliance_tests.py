import os
import re
import subprocess
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProjectComplianceTest(unittest.TestCase):
    def run_command(self, command, env=None):
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def test_repository_source_policy_gate_passes(self):
        bash = os.environ.get("BASH", "bash")
        environment = os.environ.copy()
        if os.path.dirname(bash):
            environment["PATH"] = os.path.dirname(bash) + os.pathsep + environment["PATH"]
        result = self.run_command(
            [bash, "scripts/check_prohibited.sh", "source", "."],
            env=environment,
        )
        self.assertEqual(0, result.returncode, result.stdout)

    def test_make_dry_run_emits_exact_h20_code_and_ptx_targets(self):
        make = os.environ.get("MAKE", "make")
        result = self.run_command([make, "-B", "-n", "build/lenet_cuda"])
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn("-gencode=arch=compute_90,code=sm_90", result.stdout)
        self.assertIn("-gencode=arch=compute_90,code=compute_90", result.stdout)

    def test_default_make_goal_builds_the_cuda_application(self):
        make = os.environ.get("MAKE", "make")
        result = self.run_command([make, "-B", "-n", "CUDA_ARCH=sm_90"])
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn("build/lenet_cuda", result.stdout)
        self.assertIn("-gencode=arch=compute_90,code=sm_90", result.stdout)
        self.assertIn("-gencode=arch=compute_90,code=compute_90", result.stdout)

    def test_cuda_test_dry_run_prepares_data_before_real_mnist_workflows(self):
        make = os.environ.get("MAKE", "make")
        result = self.run_command([make, "-B", "-n", "cuda-tests"])
        self.assertEqual(0, result.returncode, result.stdout)
        preparation = result.stdout.index(
            "scripts/prepare_mnist.py --output-dir data"
        )
        match = re.search(
            r"workflow_tests(?:\.exe)? --mnist-train data/train\.bin",
            result.stdout,
        )
        self.assertIsNotNone(match, result.stdout)
        workflow = match.start()
        self.assertLess(preparation, workflow)

    def test_make_exposes_python_static_and_aggregate_gates(self):
        with open(os.path.join(ROOT, "Makefile"), "r") as input_file:
            makefile = input_file.read()
        for target in ("python-tests:", "compliance:", "check:", "test:"):
            with self.subTest(target=target):
                self.assertIn(target, makefile)
        for module in (
            "tests.test_check_prohibited",
            "tests.test_check_comments",
            "tests.test_documentation",
            "tests.output_format_tests",
            "tests.compliance_tests",
        ):
            with self.subTest(module=module):
                self.assertIn(module, makefile)


if __name__ == "__main__":
    unittest.main()
