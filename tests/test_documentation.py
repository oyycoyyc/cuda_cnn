import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Reads one repository document relative to the project root.
def read_document(relative_path):
    with open(os.path.join(ROOT, relative_path), "r") as input_file:
        return input_file.read()


class DocumentationTest(unittest.TestCase):
    # README environment constraints, pinned versions, and the local no-CUDA boundary.
    def test_readme_covers_environment_constraints_and_local_boundary(self):
        readme = read_document("README.md")
        required = (
            "Ubuntu 18.04",
            "Python 3.6",
            "CUDA Toolkit 12.1",
            "GCC 7.5",
            "NVIDIA H20",
            "sm_90",
            "compute_90",
            "C++14",
            "FP32",
            "cuDNN",
            "cuBLAS",
            "TensorRT",
            "Thrust",
            "CUB",
            "cuRAND",
            "no CPU fallback",
            "local Windows development machine has no NVIDIA GPU or CUDA compiler",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)

    # Data provenance: dataset sources, wheel hashes, and preparation commands.
    def test_readme_pins_dataset_sources_hashes_and_preparation(self):
        readme = read_document("README.md")
        required = (
            "77f3279092a1c1579b2250db8eafed0ad422088c",
            "train-00000-of-00001.parquet",
            "test-00000-of-00001.parquet",
            "f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf",
            "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
            "python3.6 -m pip install --require-hashes -r requirements-py36.txt",
            "python3.6 scripts/prepare_mnist.py --output-dir data",
            "60,000",
            "10,000",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)

    # Architecture: layer shapes, memory layouts, and binary file formats.
    def test_readme_documents_architecture_layouts_and_binary_formats(self):
        readme = read_document("README.md")
        required = (
            "N x 1 x 28 x 28",
            "N x 6 x 24 x 24",
            "N x 16 x 4 x 4",
            "256 -> 120",
            "120 -> 84",
            "84 -> 10",
            "NCHW",
            "OIHW",
            "[out][in]",
            "MNISTC1\\0",
            "LNETC01\\0",
            "little-endian",
            "conv1.weight",
            "fc3.bias",
            "Adam state is not stored",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)

    # Commands and hyper-parameter defaults, timing method, and threshold.
    def test_readme_documents_commands_defaults_timing_and_threshold(self):
        readme = read_document("README.md")
        required = (
            "make CUDA_ARCH=sm_90",
            "make test",
            "make compliance",
            "./build/lenet_cuda train",
            "./build/lenet_cuda evaluate",
            "./build/lenet_cuda infer",
            "--min-accuracy 0.99",
            "Batch size: `128`",
            "Epochs: `20`",
            "Learning rate: `0.001`",
            "Beta1: `0.9`",
            "Beta2: `0.999`",
            "Epsilon: `1e-8`",
            "Weight decay: `1e-4`",
            "Seed: `1337`",
            "one unmeasured warm-up batch",
            "CUDA Events",
            "no performance acceptance threshold",
            "exit code `3`",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)

    # Error handling and required troubleshooting guidance in the README.
    def test_readme_documents_error_handling_and_required_troubleshooting(self):
        readme = read_document("README.md")
        required = (
            "CUDA architecture",
            "driver/toolkit mismatch",
            "out of memory",
            "malformed data",
            "non-finite",
            "accuracy below 99%",
            "requested bytes",
            "free/total",
            "cudaGetLastError",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, readme)

    # H20 guide ordering: data preparation precedes real MNIST workflow cases.
    def test_h20_guide_prepares_data_before_real_mnist_workflow_cases(self):
        guide = read_document("docs/h20-acceptance.md")
        prepare = guide.index("python3.6 scripts/prepare_mnist.py --output-dir data")
        overfit = guide.index(
            "./build/workflow_tests --case overfit32 --mnist-train data/train.bin"
        )
        infer = guide.index(
            "./build/workflow_tests --case infer --mnist-train data/train.bin"
        )
        self.assertLess(prepare, overfit)
        self.assertLess(prepare, infer)

    # Evidence: the guide lists every required acceptance command and log.
    def test_h20_guide_contains_complete_evidence_and_acceptance_commands(self):
        guide = read_document("docs/h20-acceptance.md")
        required = (
            "nvcc --version",
            "gcc --version",
            "nvidia-smi",
            "make V=1",
            "acceptance/verbose-build.log",
            "scripts/check_prohibited.sh source .",
            "scripts/check_prohibited.sh build acceptance/verbose-build.log",
            "scripts/check_comments.py --root . --checklist docs/comment-review-checklist.md",
            "compute-sanitizer --tool memcheck",
            "cuobjdump --list-elf",
            "cuobjdump --dump-ptx",
            "ldd build/lenet_cuda",
            "readelf -d build/lenet_cuda",
            "--min-accuracy 0.99",
            "status=pass",
            "accuracy",
            ">= 0.99",
        )
        for text in required:
            with self.subTest(text=text):
                self.assertIn(text, guide)

    # Cleanup must not delete the active acceptance log it is still writing.
    def test_h20_cleanup_cannot_delete_the_active_acceptance_log(self):
        guide = read_document("docs/h20-acceptance.md")
        log_open = guide.index("exec > >(tee")
        self.assertIn('exec > >(tee "acceptance/', guide[log_open:])
        self.assertNotIn('exec > >(tee "build/', guide[log_open:])

    # Scope: actual versus dry-run build evidence must remain distinguishable.
    def test_documentation_distinguishes_actual_and_dry_run_build_evidence(self):
        readme = read_document("README.md")
        guide = read_document("docs/h20-acceptance.md")
        self.assertIn("dry-run", readme)
        self.assertIn("commands were not executed", readme)
        self.assertIn(
            "scripts/check_prohibited.sh build acceptance/verbose-build.log", guide
        )
        self.assertIn("event=build status=pass target=all", guide)

    # The CUDA error claim excludes best-effort nonthrowing cleanup calls.
    def test_cuda_error_claim_excludes_nonthrowing_cleanup(self):
        readme = read_document("README.md")
        self.assertNotIn("Every CUDA Runtime result is checked", readme)
        self.assertIn("Operational CUDA Runtime calls", readme)
        self.assertIn("best-effort", readme)
        self.assertIn("cudaFree", readme)
        self.assertIn("cudaStreamDestroy", readme)
        self.assertIn("cudaEventDestroy", readme)

    # Actual acceptance logs stay outside the build directory.
    def test_h20_guide_keeps_actual_logs_outside_build_directory(self):
        readme = read_document("README.md")
        guide = read_document("docs/h20-acceptance.md")
        self.assertIn("tee acceptance/verbose-build.log", guide)
        self.assertIn(
            "scripts/check_prohibited.sh build acceptance/verbose-build.log", guide
        )
        self.assertIn(
            "scripts/check_prohibited.sh build acceptance/verbose-build.log", readme
        )
        self.assertIn("make -Bn V=1 > acceptance/dry-run.log", readme)
        self.assertNotIn("build/verbose-build.log", guide)
        self.assertNotIn("build/verbose-build.log", readme)
        self.assertNotIn("build/dry-run.log", readme)

    # Ordering: clean and pipefail are set before verbose logging starts.
    def test_h20_guide_cleans_and_sets_pipefail_before_verbose_logging(self):
        guide = read_document("docs/h20-acceptance.md")
        self.assertIn("set -o pipefail", guide)
        pipefail = guide.index("set -o pipefail")
        clean = guide.index("make clean")
        tee = guide.index("tee acceptance/verbose-build.log")
        self.assertLess(pipefail, tee)
        self.assertLess(clean, tee)

    # Scanner scope claims remain limited and local CUDA success is disclaimed.
    def test_documentation_scopes_scanner_and_avoids_unsupported_claims(self):
        readme = read_document("README.md")
        guide = read_document("docs/h20-acceptance.md")
        self.assertIn("decode arbitrary preprocessor or shell obfuscation", readme)
        self.assertIn("shell-obfuscated", readme)
        self.assertIn("Response files or shell-obfuscated", guide)
        self.assertIn("unsupported and fail closed", guide)
        self.assertIn("makes no local CUDA-success", readme)
        self.assertIn(
            "not evidence that CUDA succeeded on the local Windows development machine",
            guide,
        )

    # No placeholder TODO/TBD/FIXME markers may remain in the documents.
    def test_documents_contain_no_placeholder_markers(self):
        for relative_path in (
            "README.md",
            "docs/comment-review-checklist.md",
            "docs/h20-acceptance.md",
        ):
            with self.subTest(path=relative_path):
                text = read_document(relative_path)
                self.assertIsNone(
                    re.search(r"(?im)\b(?:TODO|TBD|FIXME|PLACEHOLDER)\b", text)
                )


if __name__ == "__main__":
    unittest.main()
