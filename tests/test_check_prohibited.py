import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "scripts", "check_prohibited.sh")


class ProhibitedCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.mkdtemp(prefix="lenet-prohibited-")
        for directory in ("include", "src", "tests", "scripts", "docs"):
            os.makedirs(os.path.join(self.temporary, directory))
        self.write("Makefile", "all:\n\t@echo clean\n")

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

    def run_checker(self, mode, target):
        bash = os.environ.get("BASH", "bash")
        environment = os.environ.copy()
        if os.path.dirname(bash):
            environment["PATH"] = os.path.dirname(bash) + os.pathsep + environment["PATH"]
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
        self.write("src/model.cu", "void LaunchModel() {}\n")
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

    def test_every_prohibited_namespace_is_rejected(self):
        namespaces = ("nvinfer1::Builder", "thrust::device_vector", "cub::DeviceReduce")
        for index, namespace in enumerate(namespaces):
            with self.subTest(namespace=namespace):
                self.assert_rejected(
                    "src/namespace_{0}.cu".format(index),
                    "void use() {{ (void)sizeof({0}); }}\n".format(namespace),
                )

    def test_every_prohibited_api_family_is_rejected(self):
        api_calls = (
            "cudnnCreate(nullptr);",
            "cublasSgemm();",
            "createInferRuntime(logger);",
            "curandCreateGenerator(nullptr, 0);",
        )
        for index, api_call in enumerate(api_calls):
            with self.subTest(api_call=api_call):
                self.assert_rejected(
                    "tests/compiled_{0}_tests.cu".format(index),
                    "void test() {{ {0} }}\n".format(api_call),
                )

    def test_source_scan_rejects_prohibited_linker_flags_in_makefile(self):
        for library in ("cudnn", "cublas", "nvinfer", "nvonnxparser", "curand"):
            with self.subTest(library=library):
                self.assert_rejected(
                    "Makefile", "LDLIBS += -l{0}\n".format(library)
                )

    def test_production_cpu_fallback_and_nondeterministic_random_apis_are_rejected(self):
        violations = (
            ("src/fallback.cpp", '#include "cpu_reference.h"\n'),
            ("src/fallback.cpp", "cpu_reference::LinearForward(values);\n"),
            ("src/random.cpp", "std::shuffle(first, last, engine);\n"),
            ("include/random.h", "std::normal_distribution<float> normal;\n"),
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

    def test_build_scan_rejects_every_prohibited_linker_flag(self):
        for library in ("cudnn", "cublas", "nvinfer", "nvonnxparser", "curand"):
            with self.subTest(library=library):
                self.assert_rejected(
                    "verbose-build.log",
                    "nvcc objects.o -l{0} -o build/lenet_cuda\n".format(library),
                    mode="build",
                )

    def test_build_scan_accepts_runtime_only_verbose_link(self):
        log = self.write(
            "verbose-build.log",
            "nvcc -std=c++14 -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 main.o -o build/lenet_cuda\n",
        )
        result = self.run_checker("build", log)
        self.assertEqual(0, result.returncode, result.stdout)

    def test_invalid_mode_and_missing_target_fail(self):
        invalid = self.run_checker("unknown", self.temporary)
        self.assertNotEqual(0, invalid.returncode)
        self.assertIn("usage", invalid.stdout.lower())
        missing = self.run_checker("build", os.path.join(self.temporary, "missing.log"))
        self.assertNotEqual(0, missing.returncode)
        self.assertIn("missing.log", missing.stdout)


if __name__ == "__main__":
    unittest.main()
