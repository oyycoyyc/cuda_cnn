import os
import shutil
import subprocess
import tempfile
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "scripts", "check_prohibited.sh")


class ProhibitedCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.mkdtemp(prefix="lenet-prohibited-")
        for directory in ("include", "src", "tests", "scripts", "docs"):
            os.makedirs(os.path.join(self.temporary, directory))
        self.write(
            "Makefile",
            "BUILD_DIR := build\n"
            "TEST_SOURCES := $(wildcard tests/*_tests.cpp tests/*_tests.cu)\n"
            "TEST_OUTPUTS := $(patsubst tests/%,build/%,$(TEST_SOURCES))\n"
            "all: $(BUILD_DIR)/lenet_cuda $(TEST_OUTPUTS)\n"
            "$(BUILD_DIR)/lenet_cuda:\n"
            "\tnvcc -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 "
            "src/model.cu -o $@\n"
            "build/%: tests/%\n"
            "\tnvcc $< -o $@\n",
        )
        self.write("src/model.cu", "void LaunchModel() {}\n")

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

    def run_checker(self, mode, target, extra_environment=None):
        bash = os.environ.get("BASH", "bash")
        environment = os.environ.copy()
        if os.path.dirname(bash):
            environment["PATH"] = os.path.dirname(bash) + os.pathsep + environment["PATH"]
        if extra_environment is not None:
            environment.update(extra_environment)
        return subprocess.run(
            [bash, CHECKER, mode, target],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def assert_rejected(self, relative_path, content, mode="source"):
        path = os.path.join(self.temporary, relative_path)
        original = None
        if os.path.isfile(path):
            with open(path, "r") as input_file:
                original = input_file.read()
        try:
            path = self.write(relative_path, content)
            target = self.temporary if mode == "source" else path
            result = self.run_checker(mode, target)
            self.assertNotEqual(0, result.returncode, result.stdout)
            self.assertIn(os.path.basename(relative_path), result.stdout)
        finally:
            if original is None:
                if os.path.exists(path):
                    os.remove(path)
            else:
                self.write(relative_path, original)

    def test_clean_source_tree_passes_and_nonproduction_text_is_ignored(self):
        self.write("include/model.h", "void LaunchModel();\n")
        decoy = "cudnn cublas nvinfer1 thrust:: cub:: curand -lcudnn\n"
        self.write("README.md", decoy)
        self.write("docs/design.md", decoy)
        self.write("scripts/check_prohibited.sh", decoy)
        self.write("tests/test_negative_fixture.py", decoy)
        result = self.run_checker("source", self.temporary)
        self.assertEqual(0, result.returncode, result.stdout)

    def test_every_prohibited_header_is_rejected(self):
        headers = (
            "cudnn.h",
            "cublas_v2.h",
            "NvInfer.h",
            "thrust/device_vector.h",
            "cub/cub.cuh",
            "curand.h",
        )
        for index, header in enumerate(headers):
            with self.subTest(header=header):
                self.assert_rejected(
                    "include/header_{0}.h".format(index),
                    "#include <{0}>\n".format(header),
                )

    def test_header_macro_forced_include_and_whitespace_variants_are_rejected(self):
        variants = (
            '#  include   < CuDnN.h >\n',
            '#define BACKEND_HEADER "NvOnnxParser.h"\n#include BACKEND_HEADER\n',
            'NVCCFLAGS += -include NvInferRuntime.h\n',
            'NVCCFLAGS += -include=thrust/device_vector.h\n',
        )
        paths = ("include/spaced.h", "include/macro.h", "Makefile", "Makefile")
        for path, content in zip(paths, variants):
            with self.subTest(content=content):
                self.assert_rejected(path, content)

    def test_every_prohibited_namespace_is_rejected(self):
        namespaces = (
            "nvinfer1::Builder",
            "using namespace nvinfer1;",
            "::nvonnxparser::IParser",
            "thrust::device_vector",
            "cub::DeviceReduce",
        )
        for index, namespace in enumerate(namespaces):
            with self.subTest(namespace=namespace):
                self.assert_rejected(
                    "src/namespace_{0}.cu".format(index),
                    "void use() {{ {0}; }}\n".format(namespace),
                )

    def test_every_prohibited_api_family_is_rejected(self):
        api_calls = (
            "cudnnCreate(nullptr);",
            "auto cudnn_pointer = &cudnnCreate;",
            "cublasSgemm();",
            "createInferRuntime(logger);",
            "auto builder = &createInferBuilder;",
            "auto refitter = &createInferRefitter;",
            "auto parser = &nvonnxparser::createParser;",
            "curandCreateGenerator(nullptr, 0);",
        )
        for index, api_call in enumerate(api_calls):
            with self.subTest(api_call=api_call):
                self.assert_rejected(
                    "tests/compiled_{0}_tests.cu".format(index),
                    "void test() {{ {0} }}\n".format(api_call),
                )

    def test_source_scan_rejects_prohibited_linker_flags_in_makefile(self):
        variants = (
            "LDLIBS += -lcudnn\n",
            "LDLIBS += -Wl,-lcublasLt\n",
            "LDLIBS += -l:libnvinfer.so\n",
            "LDLIBS += /opt/cuda/lib64/libnvonnxparser.so.10\n",
            "LDLIBS += C:\\cuda\\lib\\x64\\curand.lib\n",
        )
        for content in variants:
            with self.subTest(content=content):
                self.assert_rejected("Makefile", content)

    def test_all_runtime_only_prohibited_library_families_are_rejected(self):
        tokens = (
            "cudnn.h",
            "cublasLt.h",
            "NvInferPlugin.h",
            "NvOnnxParser.h",
            "NvParsers.h",
            "curand.h",
            "cusolverDn.h",
            "cusparse.h",
            "cufft.h",
            "cutensor.h",
            "cutlass/cutlass.h",
            "cudss.h",
            "nccl.h",
            "nvshmem.h",
            "npp.h",
            "torch/torch.h",
            "tensorflow/core/framework/tensor.h",
            "caffe/caffe.hpp",
            "onnxruntime_cxx_api.h",
            "openvino/openvino.hpp",
            "opencv2/dnn.hpp",
            "cuda_fp16.h",
            "mma.h",
        )
        for index, token in enumerate(tokens):
            with self.subTest(token=token):
                self.assert_rejected(
                    "include/library_{0}.h".format(index),
                    '#include "{0}"\n'.format(token),
                )

    def test_production_cpu_fallback_and_nondeterministic_random_apis_are_rejected(self):
        violations = (
            ("src/fallback.cpp", '#include "cpu_reference.h"\n'),
            ("src/fallback.cpp", "cpu_reference::LinearForward(values);\n"),
            ("src/random.cpp", "std::shuffle(first, last, engine);\n"),
            ("include/random.h", "std::normal_distribution<float> normal;\n"),
            ("include/random.h", "#include <random>\n"),
            ("src/random.cpp", "std::mt19937 engine;\n"),
            ("src/random.cpp", "std::uniform_real_distribution<float> value;\n"),
            ("src/fallback.cpp", "float CpuForward(const float* input);\n"),
            ("src/fallback.cpp", "RunOnCpu(model);\n"),
        )
        for relative_path, content in violations:
            with self.subTest(content=content):
                self.assert_rejected(relative_path, content)

    def test_compiled_cpu_reference_tests_are_not_mistaken_for_a_fallback(self):
        self.write(
            "tests/cpu_reference.cpp",
            '#include "cpu_reference.h"\n'
            "float ReferenceOnlyInTests() { return cpu_reference::Relu(1.0F); }\n",
        )
        result = self.run_checker("source", self.temporary)
        self.assertEqual(0, result.returncode, result.stdout)

    def test_production_link_graph_rejects_cpu_reference_object(self):
        self.write(
            "Makefile",
            "BUILD_DIR := build\n"
            "CPU_REFERENCE_OBJECT := $(BUILD_DIR)/cpu_reference.o\n"
            "$(BUILD_DIR)/lenet_cuda: main.o $(CPU_REFERENCE_OBJECT)\n"
            "\tnvcc $^ -o $@\n"
            "$(BUILD_DIR)/cpu_reference.o: tests/cpu_reference.cpp\n"
            "\tg++ -c $< -o $@\n"
            "main.o: src/model.cu\n"
            "\tnvcc -c $< -o $@\n",
        )
        result = self.run_checker("source", self.temporary)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("cpu_reference", result.stdout.lower())

    def test_dormant_negative_fixture_is_ignored_but_compiled_test_is_scanned(self):
        violation = "void test() { cudnnCreate(nullptr); }\n"
        self.write("tests/dormant_negative_fixture.cu", violation)
        clean = self.run_checker("source", self.temporary)
        self.assertEqual(0, clean.returncode, clean.stdout)
        self.write("tests/active_tests.cu", violation)
        active = self.run_checker("source", self.temporary)
        self.assertNotEqual(0, active.returncode, active.stdout)
        self.assertIn("active_tests.cu", active.stdout)

    def test_build_scan_rejects_every_prohibited_linker_flag(self):
        for library in ("cudnn", "cublas", "nvinfer", "nvonnxparser", "curand"):
            with self.subTest(library=library):
                self.assert_rejected(
                    "verbose-build.log",
                    self.valid_build_log() +
                    "nvcc objects.o -l{0} -o build/other\n".format(library),
                    mode="build",
                )

    def valid_build_log(self, completion=True):
        log = (
            "nvcc -std=c++14 -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 -c src/main.cu "
            "-o build/main.o\n"
            "g++ -std=c++14 -c src/reporting.cpp -o build/reporting.o\n"
            "nvcc -std=c++14 -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 build/main.o "
            "-o build/lenet_cuda\n"
        )
        if completion:
            log += "event=build status=pass target=all\n"
        return log

    def test_actual_build_scan_accepts_complete_runtime_only_log(self):
        log = self.write(
            "verbose-build.log",
            self.valid_build_log(),
        )
        result = self.run_checker("build", log)
        self.assertEqual(0, result.returncode, result.stdout)

    def test_dry_run_scan_is_explicit_and_does_not_require_completion_marker(self):
        log = self.write("verbose-build.log", self.valid_build_log(False))
        result = self.run_checker("dry-run", log)
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn("commands-not-executed", result.stdout)

    def test_build_scan_rejects_empty_truncated_stale_and_nonbuild_logs(self):
        cases = (
            ("", "empty"),
            ("documentation only\n", "compiler"),
            (
                "nvcc -c src/main.cu -o build/main.o\n",
                "lenet_cuda",
            ),
            (
                "nvcc -gencode=arch=compute_90,code=sm_90 -c src/main.cu "
                "-o build/main.o\n"
                "nvcc build/main.o -o build/lenet_cuda\n",
                "compute_90",
            ),
            (
                "nvcc -std=c++14 -gencode=arch=compute_90,code=sm_90 "
                "-gencode=arch=compute_90,code=compute_90 build/main.o "
                "-o build/lenet_cuda\n"
                "event=build status=pass target=all\n",
                "compile",
            ),
            (
                "nvcc -gencode=arch=compute_90,code=sm_90 "
                "-gencode=arch=compute_90,code=compute_90 -c src/main.cu "
                "-o build/main.o\n"
                "nvcc build/main.o -o build/lenet_cuda\n"
                "event=build status=pass target=all\n",
                "linker",
            ),
            (self.valid_build_log(False), "completion"),
        )
        for content, expected in cases:
            with self.subTest(expected=expected):
                log = self.write("verbose-build.log", content)
                result = self.run_checker("build", log)
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn(expected, result.stdout.lower())

        log = self.write("verbose-build.log", self.valid_build_log())
        old = time.time() - 10
        os.utime(log, (old, old))
        self.write("src/newer.cu", "void Newer() {}\n")
        stale = self.run_checker("build", log)
        self.assertNotEqual(0, stale.returncode, stale.stdout)
        self.assertIn("stale", stale.stdout.lower())

        os.remove(os.path.join(self.temporary, "src", "newer.cu"))
        log = self.write("verbose-build.log", self.valid_build_log())
        os.utime(log, (old, old))
        self.write("tests/newer_tests.cpp", "int main() { return 0; }\n")
        stale_test = self.run_checker("build", log)
        self.assertNotEqual(0, stale_test.returncode, stale_test.stdout)
        self.assertIn("stale", stale_test.stdout.lower())

    def test_scanner_fails_closed_when_find_or_grep_fails(self):
        for command in ("find", "grep"):
            with self.subTest(command=command):
                tools = os.path.join(self.temporary, "tools_" + command)
                os.makedirs(tools)
                tool = os.path.join(tools, command)
                self.write(
                    os.path.relpath(tool, self.temporary),
                    "#!/bin/sh\necho forced {0} failure >&2\nexit 2\n".format(
                        command
                    ),
                )
                os.chmod(tool, 0o755)
                bash_directory = os.path.dirname(os.environ.get("BASH", ""))
                environment = {
                    "PATH": tools + os.pathsep + bash_directory + os.pathsep +
                    os.environ["PATH"]
                }
                result = self.run_checker(
                    "source", self.temporary, extra_environment=environment
                )
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn(command, result.stdout.lower())
                self.assertIn("failed", result.stdout.lower())

    def test_invalid_mode_and_missing_target_fail(self):
        invalid = self.run_checker("unknown", self.temporary)
        self.assertNotEqual(0, invalid.returncode)
        self.assertIn("usage", invalid.stdout.lower())
        missing = self.run_checker("build", os.path.join(self.temporary, "missing.log"))
        self.assertNotEqual(0, missing.returncode)
        self.assertIn("missing.log", missing.stdout)


if __name__ == "__main__":
    unittest.main()
