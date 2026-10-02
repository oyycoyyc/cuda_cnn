import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKER = os.path.join(ROOT, "scripts", "check_prohibited.sh")
ANALYZER = os.path.join(ROOT, "scripts", "analyze_build_graph.py")


class ProhibitedCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.mkdtemp(prefix="lenet-prohibited-")
        for directory in ("include", "src", "tests", "scripts", "docs"):
            os.makedirs(os.path.join(self.temporary, directory))
        self.write(
            "Makefile",
            "BUILD_DIR := build\n"
            ".DEFAULT_GOAL := all\n"
            "TEST_SOURCES := $(wildcard tests/*_tests.cpp tests/*_tests.cu)\n"
            "TEST_OUTPUTS := $(patsubst tests/%,build/%,$(TEST_SOURCES))\n"
            "COMPLIANCE_TEST_SOURCES := $(TEST_SOURCES)\n"
            ".PHONY: compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@for source in $(COMPLIANCE_TEST_SOURCES); do "
            "printf 'test-source=%s\\n' \"$$source\"; done\n"
            "all: $(BUILD_DIR)/lenet_cuda $(TEST_OUTPUTS)\n"
            "$(BUILD_DIR)/lenet_cuda:\n"
            "\tnvcc -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 "
            "src/model.cu -o $@\n"
            "build/%: tests/%\n"
            "\tnvcc $< -o $@\n",
        )
        self.write("src/model.cu", "void LaunchModel() {}\n")
        self.write("tests/smoke_tests.cpp", "int main() { return 0; }\n")

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
        environment["PYTHON"] = sys.executable.replace("\\", "/")
        if extra_environment is not None:
            environment.update(extra_environment)
        return subprocess.run(
            [bash, CHECKER, mode, target],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def run_analyzer(self, recipes, manifest):
        recipe_path = self.write("recipes.log", recipes)
        manifest_path = self.write("manifest.log", manifest)
        return subprocess.run(
            [
                sys.executable,
                ANALYZER,
                "source",
                "--root",
                self.temporary,
                "--recipes",
                recipe_path,
                "--manifest",
                manifest_path,
            ],
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
            ".DEFAULT_GOAL := all\n"
            "CPU_REFERENCE_OBJECT := $(BUILD_DIR)/cpu_reference.o\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/cpu_reference.cpp\n"
            "all: $(BUILD_DIR)/lenet_cuda\n"
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
        self.write("tests/dormant_negative_fixture.h", "#include <cublas_v2.h>\n")
        clean = self.run_checker("source", self.temporary)
        self.assertEqual(0, clean.returncode, clean.stdout)
        self.write("tests/active_tests.cu", violation)
        active = self.run_checker("source", self.temporary)
        self.assertNotEqual(0, active.returncode, active.stdout)
        self.assertIn("active_tests.cu", active.stdout)

    def test_active_test_transitive_headers_are_scanned_without_dependency_files(self):
        self.write("tests/smoke_tests.cpp", '#include "test_harness.h"\n')
        self.write("tests/test_harness.h", '#include "cpu_reference.h"\n')
        self.write("tests/cpu_reference.h", "#include <cudnn.h>\n")

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("cpu_reference.h", result.stdout)

    def test_active_test_unresolved_quoted_include_fails_closed(self):
        self.write("tests/smoke_tests.cpp", '#include "missing_project_header.h"\n')

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("cannot resolve", result.stdout.lower())
        self.assertIn("missing_project_header.h", result.stdout)

    def test_source_scan_propagates_source_bearing_recipe_without_output(self):
        self.write(
            "Makefile",
            ".PHONY: compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda:\n"
            "\tnvcc src/model.cu -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\tcompiler-wrapper clang++ -c 'tests/smoke_tests.cpp'\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("source-bearing recipe has no supported output", result.stdout)
        self.assertIn("Make recipe provenance analysis failed", result.stdout)

    def test_source_with_archive_input_still_requires_explicit_output(self):
        result = self.run_analyzer(
            "compiler-wrapper clang++ -c tests/smoke_tests.cpp "
            "build/libsupport.a build/input.o\n"
            "clang++ src/model.cu -o build/lenet_cuda\n",
            "test-source=tests/smoke_tests.cpp\n",
        )

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("source-bearing recipe has no supported output", result.stdout)

    def test_known_archive_tool_path_supports_positional_output(self):
        result = self.run_analyzer(
            "clang++ -c src/model.cu -o build/model.o\n"
            "tools/llvm-ar rcs build/libmodel.a build/model.o\n"
            "clang++ build/libmodel.a -o build/lenet_cuda\n"
            "clang++ -c tests/smoke_tests.cpp -o build/smoke.o\n",
            "test-source=tests/smoke_tests.cpp\n",
        )

        self.assertEqual(0, result.returncode, result.stdout)

    def test_test_source_recipe_rejects_response_file_before_manifest_mismatch(self):
        self.write("tests/other_tests.cpp", "int Other() { return 0; }\n")

        result = self.run_analyzer(
            "compiler-wrapper clang++ -c tests/smoke_tests.cpp "
            "@build/test.rsp -o build/smoke.o\n"
            "clang++ src/model.cu -o build/lenet_cuda\n",
            "test-source=tests/other_tests.cpp\n",
        )

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("response", result.stdout.lower())
        self.assertNotIn("manifest mismatch", result.stdout.lower())

    def test_test_source_recipe_rejects_standalone_and_adjacent_shell_controls(self):
        controls = (
            "; other-command",
            "&&other-command",
            "|other-command",
            "2>/dev/null",
            "trailing;",
            "$(other-command)",
            "`other-command`",
        )
        for control in controls:
            with self.subTest(control=control):
                result = self.run_analyzer(
                    "compiler-wrapper clang++ -c tests/smoke_tests.cpp "
                    "-o build/smoke.o {0}\n".format(control) +
                    "clang++ src/model.cu -o build/lenet_cuda\n",
                    "test-source=tests/smoke_tests.cpp\n",
                )

                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn("shell", result.stdout.lower())

    def test_test_source_recipe_allows_quoted_and_escaped_literal_punctuation(self):
        result = self.run_analyzer(
            "compiler-wrapper clang++ '-DNAME=$(literal)' "
            "'-DPIPE=left|right' -I 'tests/include&support' "
            "-DSEMI=left\\;right -c tests/smoke_tests.cpp "
            "-o build/smoke.o\n"
            "clang++ src/model.cu -o build/lenet_cuda\n",
            "test-source=tests/smoke_tests.cpp\n",
        )

        self.assertEqual(0, result.returncode, result.stdout)

    def test_build_scan_rejects_every_prohibited_linker_flag(self):
        for library in ("cudnn", "cublas", "nvinfer", "nvonnxparser", "curand"):
            with self.subTest(library=library):
                self.assert_rejected(
                    "verbose-build.log",
                    self.valid_build_log() +
                    "nvcc objects.o -l{0} -o build/other\n".format(library),
                    mode="build",
                )

    def test_production_link_rejects_renamed_object_compiled_from_test_source(self):
        self.write("tests/cpu_reference.cpp", "float Oracle() { return 0.0F; }\n")
        self.write(
            "Makefile",
            "CXX ?= g++\n"
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/cpu_reference.cpp\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda: build/oracle.o\n"
            "\tnvcc $^ -o $@\n"
            "build/oracle.o: tests/cpu_reference.cpp\n"
            "\tg++ -c $< -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\tg++ $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("tests-owned", result.stdout.lower())
        self.assertIn("oracle.o", result.stdout.lower())

    def test_production_link_rejects_transitively_renamed_test_object(self):
        self.write("tests/cpu_reference.cpp", "float Oracle() { return 0.0F; }\n")
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/cpu_reference.cpp\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda: build/model_support.o\n"
            "\tnvcc $^ -o $@\n"
            "build/model_support.o: build/oracle.o\n"
            "\tnvcc -r $< -o $@\n"
            "build/oracle.o: tests/cpu_reference.cpp\n"
            "\tg++ -c $< -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\tg++ $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("tests-owned", result.stdout.lower())
        self.assertIn("model_support.o", result.stdout.lower())

    def test_clang_ccache_quoted_paths_and_iquote_order_are_scanned(self):
        self.write(
            "tests/space dir/active test.cpp",
            '#include "selected.h"\nint main() { return 0; }\n',
        )
        self.write("tests/include dir/selected.h", "void Clean();\n")
        self.write("tests/quote dir/selected.h", "#include <cudnn.h>\n")
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf '%s\\n' 'test-source=tests/space dir/active test.cpp'\n"
            "all: build/lenet_cuda build/active\\ test\n"
            "build/lenet_cuda:\n"
            "\tclang++ src/model.cu -o $@\n"
            "build/active\\ test: tests/space\\ dir/active\\ test.cpp\n"
            "\tccache clang++ -I 'tests/include dir' "
            "-iquote 'tests/quote dir' -c 'tests/space dir/active test.cpp' "
            "-o 'build/active test.o'\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("quote dir", result.stdout)
        self.assertIn("selected.h", result.stdout)

    def test_analyzer_accepts_unknown_wrapper_with_quoted_source_and_output(self):
        self.write("tests/space dir/active test.cpp", "int main() { return 0; }\n")
        recipes = self.write(
            "recipes.log",
            "remote-cache unusual-cxx -c 'tests/space dir/active test.cpp' "
            "-o 'build/active test.o'\n"
            "unusual-link src/model.cu -o build/lenet_cuda\n",
        )
        manifest = self.write(
            "manifest.log", "test-source=tests/space dir/active test.cpp\n"
        )

        result = subprocess.run(
            [
                sys.executable,
                ANALYZER,
                "source",
                "--root",
                self.temporary,
                "--recipes",
                recipes,
                "--manifest",
                manifest,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        self.assertEqual(0, result.returncode, result.stdout)
        self.assertEqual("tests/space dir/active test.cpp\n", result.stdout)

    def test_attached_include_flags_resolve_active_header(self):
        self.write("tests/smoke_tests.cpp", '#include "selected.h"\n')
        self.write("tests/include/selected.h", "void Clean();\n")
        self.write("tests/quote/selected.h", "#include <cublas_v2.h>\n")
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda:\n"
            "\t$(CXX) src/model.cu -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\t$(CXX) -Itests/include -iquotetests/quote -c $< -o $@\n",
        )

        result = self.run_checker(
            "source", self.temporary, extra_environment={"CXX": "clang++"}
        )

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("tests/quote", result.stdout.replace("\\", "/"))

    def test_make_manifest_and_recipe_sources_must_match(self):
        self.write("tests/other_tests.cpp", "int Other() { return 0; }\n")
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "\t@printf 'test-source=%s\\n' tests/other_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda:\n"
            "\tclang++ src/model.cu -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\tclang++ -c $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("manifest", result.stdout.lower())
        self.assertIn("other_tests.cpp", result.stdout)

    def test_recipe_test_source_absent_from_manifest_is_rejected(self):
        self.write("tests/other_tests.cpp", "int Other() { return 0; }\n")
        result = self.run_analyzer(
            "clang++ -c tests/smoke_tests.cpp -o build/smoke.o\n"
            "clang++ -c tests/other_tests.cpp -o build/other.o\n"
            "clang++ src/model.cu -o build/lenet_cuda\n",
            "test-source=tests/smoke_tests.cpp\n",
        )

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("missing from manifest: tests/other_tests.cpp", result.stdout)

    def test_active_header_symlink_cannot_escape_repository(self):
        outside = tempfile.mkdtemp(prefix="lenet-prohibited-outside-")
        self.addCleanup(shutil.rmtree, outside)
        outside_header = os.path.join(outside, "outside.h")
        with open(outside_header, "w") as output:
            output.write("void Outside();\n")
        link = os.path.join(self.temporary, "tests", "escape.h")
        try:
            os.symlink(outside_header, link)
        except OSError as error:
            self.skipTest("symlink creation unavailable: {0}".format(error))
        self.write("tests/smoke_tests.cpp", '#include "escape.h"\n')

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("escapes source root", result.stdout.lower())

    def test_clang_ld_and_archive_provenance_reaches_production(self):
        self.write("tests/cpu_reference.cpp", "float Oracle() { return 0.0F; }\n")
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/cpu_reference.cpp\n"
            "all: build/lenet_cuda\n"
            "build/lenet_cuda: build/main.o build/libsupport.a\n"
            "\tclang++ $^ -o $@\n"
            "build/libsupport.a: build/renamed.o\n"
            "\tar rcs $@ $<\n"
            "build/renamed.o: build/oracle.o\n"
            "\tld -r $< -o $@\n"
            "build/oracle.o: tests/cpu_reference.cpp\n"
            "\tccache clang++ -c $< -o $@\n"
            "build/main.o: src/model.cu\n"
            "\tclang++ -c $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("tests-owned", result.stdout.lower())
        self.assertIn("libsupport.a", result.stdout)

    def test_test_only_links_do_not_taint_production(self):
        self.write("tests/cpu_reference.cpp", "float Oracle() { return 0.0F; }\n")
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/cpu_reference.cpp\n"
            "all: build/lenet_cuda build/workflow_tests\n"
            "build/lenet_cuda: build/main.o\n"
            "\tclang++ $^ -o $@\n"
            "build/workflow_tests: build/oracle.o\n"
            "\tclang++ $^ -o $@\n"
            "build/oracle.o: tests/cpu_reference.cpp\n"
            "\tclang++ -c $< -o $@\n"
            "build/main.o: src/model.cu\n"
            "\tclang++ -c $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertEqual(0, result.returncode, result.stdout)

    def test_production_include_option_path_is_not_an_artifact_input(self):
        os.makedirs(os.path.join(self.temporary, "src", "include path"))
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda: build/main.o\n"
            "\tclang++ $^ -o $@\n"
            "build/main.o: src/model.cu\n"
            "\tclang++ -I 'src/include path' -c $< -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\tclang++ -c $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertEqual(0, result.returncode, result.stdout)

    def test_production_response_file_is_rejected_as_ambiguous(self):
        self.write(
            "Makefile",
            ".DEFAULT_GOAL := all\n"
            ".PHONY: all compliance-test-sources\n"
            "compliance-test-sources:\n"
            "\t@printf 'test-source=%s\\n' tests/smoke_tests.cpp\n"
            "all: build/lenet_cuda build/smoke_tests\n"
            "build/lenet_cuda:\n"
            "\tclang++ @build/objects.rsp -o $@\n"
            "build/smoke_tests: tests/smoke_tests.cpp\n"
            "\tclang++ -c $< -o $@\n",
        )

        result = self.run_checker("source", self.temporary)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("response", result.stdout.lower())

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

    def test_actual_build_scan_requires_marker_as_final_nonempty_line(self):
        log = self.write(
            "verbose-build.log",
            self.valid_build_log() + "nvcc --version\n",
        )

        result = self.run_checker("build", log)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("final nonempty line", result.stdout.lower())

    def test_actual_build_scan_rejects_failure_after_success_marker(self):
        log = self.write(
            "verbose-build.log",
            self.valid_build_log() + "make: *** [Makefile:80: all] Error 2\n",
        )

        result = self.run_checker("build", log)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("failure record", result.stdout.lower())

    def test_actual_build_scan_rejects_earlier_compiler_fatal_error(self):
        log = self.write(
            "verbose-build.log",
            "src/main.cu:4:2: fatal error: missing.h: No such file or directory\n" +
            self.valid_build_log(),
        )

        result = self.run_checker("build", log)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("failure record", result.stdout.lower())

    def test_actual_build_scan_accepts_benign_error_zero_and_documentation(self):
        log = self.write(
            "verbose-build.log",
            "make: *** [Makefile:80: all] Error 0\n"
            "compiler documentation: error: means a failed translation\n" +
            self.valid_build_log(),
        )

        result = self.run_checker("build", log)

        self.assertEqual(0, result.returncode, result.stdout)

    def test_actual_build_scan_rejects_compiler_error_diagnostic(self):
        log = self.write(
            "verbose-build.log",
            "src/main.cu:4:2: error: invalid conversion\n" +
            self.valid_build_log(),
        )

        result = self.run_checker("build", log)

        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("failure record", result.stdout.lower())

    def test_actual_build_scan_accepts_clang_recipe_structure(self):
        log = self.write(
            "verbose-build.log",
            "clang++ -std=c++14 -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 -c src/main.cu "
            "-o build/main.o\n"
            "clang++ -std=c++14 -gencode=arch=compute_90,code=sm_90 "
            "-gencode=arch=compute_90,code=compute_90 build/main.o "
            "-o build/lenet_cuda\n"
            "event=build status=pass target=all\n",
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
            (
                self.valid_build_log(False) +
                "event=build status=pass target=\n",
                "final nonempty line",
            ),
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

    def test_source_scan_propagates_malformed_and_nonzero_make_invocations(self):
        makefiles = (
            "all:\n  malformed recipe\n",
            "$(error forced nonzero Make invocation)\nall:\n\t@true\n",
        )
        for makefile in makefiles:
            with self.subTest(makefile=makefile):
                self.write("Makefile", makefile)
                result = self.run_checker("source", self.temporary)

                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn("make dry-run failed", result.stdout.lower())

    def test_invalid_mode_and_missing_target_fail(self):
        invalid = self.run_checker("unknown", self.temporary)
        self.assertNotEqual(0, invalid.returncode)
        self.assertIn("usage", invalid.stdout.lower())
        missing = self.run_checker("build", os.path.join(self.temporary, "missing.log"))
        self.assertNotEqual(0, missing.returncode)
        self.assertIn("missing.log", missing.stdout)


if __name__ == "__main__":
    unittest.main()
