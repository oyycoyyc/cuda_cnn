import os
import re
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_document(relative_path):
    with open(os.path.join(ROOT, relative_path), "r") as input_file:
        return input_file.read()


class DocumentationTest(unittest.TestCase):
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

    def test_h20_guide_contains_complete_evidence_and_acceptance_commands(self):
        guide = read_document("docs/h20-acceptance.md")
        required = (
            "nvcc --version",
            "gcc --version",
            "nvidia-smi",
            "make V=1",
            "build/verbose-build.log",
            "scripts/check_prohibited.sh source .",
            "scripts/check_prohibited.sh build build/verbose-build.log",
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
