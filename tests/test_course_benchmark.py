from __future__ import print_function

import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest

from scripts import course_benchmark

try:
    from unittest import mock
except ImportError:  # pragma: no cover - Python 3.6 includes unittest.mock
    import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PIXELS = 28 * 28
BENCHMARK_KEYS = {
    "schema",
    "executable",
    "images_path",
    "labels_path",
    "model_dir",
    "model_source",
    "n",
    "batch",
    "input_mode",
    "profiled",
    "runs_requested",
    "sha256",
    "runs",
    "summary",
}
SHA256_KEYS = {"binary", "model", "meta", "images", "labels"}
RUN_KEYS = {"wall_ms", "returncode", "correct", "stdout", "stderr"}
SUMMARY_KEYS = {
    "median_ms",
    "max_ms",
    "correct",
    "accuracy",
    "accuracy_pass",
    "all_wall_below_100ms",
}
REFERENCE_KEYS = {"reference_sha256", "reference_mismatches"}


# A harmless real subprocess used in place of the compiled infer executable.
# It records the exact argv and relevant environment it receives, then emits
# stdout, stderr, and an exit status from a per-invocation configuration. It
# never reads labels or performs any numerical work.
MOCK_SCRIPT = r'''
import json
import os
import sys
import time

directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(directory, "config.json"), "r") as config_file:
    config = json.load(config_file)

record_path = os.path.join(directory, "record.jsonl")
invocation = 0
if os.path.exists(record_path):
    with open(record_path, "r") as record_file:
        invocation = sum(1 for _ in record_file)
record = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "COURSE_MODEL_DIR": os.environ.get("COURSE_MODEL_DIR"),
    "COURSE_INFER_BATCH": os.environ.get("COURSE_INFER_BATCH"),
    "COURSE_PROFILE": os.environ.get("COURSE_PROFILE"),
    "COURSE_INPUT_MODE": os.environ.get("COURSE_INPUT_MODE"),
    "COURSE_EPOCHS": os.environ.get("COURSE_EPOCHS"),
}
with open(record_path, "a") as record_file:
    record_file.write(json.dumps(record) + "\n")

sequence = config.get("sequence")
if sequence:
    step = sequence[invocation % len(sequence)]
else:
    step = config
if step.get("sleep_seconds", 0):
    time.sleep(step["sleep_seconds"])
if step.get("stderr"):
    sys.stderr.buffer.write(step["stderr"].encode("utf-8"))
lines = step.get("stdout_lines", [])
if lines:
    text = "\n".join(str(line) for line in lines)
    if step.get("final_newline", True):
        text += "\n"
    sys.stdout.buffer.write(text.encode("utf-8"))
sys.exit(step.get("exit_code", 0))
'''


# Returns the SHA-256 digest of one file for identity checks.
def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class CourseBenchmarkTest(unittest.TestCase):
    # Every test owns its executable, model, inputs, and output JSON.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def path(self, name):
        return os.path.join(self.directory, name)

    def output_path(self, name="benchmark.json"):
        return self.path(name)

    # Writes the stdlib mock implementation beside its configuration.
    def _write_mock_script(self, path):
        with open(path, "w", newline="\n") as script_file:
            script_file.write(MOCK_SCRIPT)

    # Creates an executable with the same name on both POSIX and Windows. The
    # POSIX executable is an executable Python script; Windows uses a batch
    # wrapper so the same tests can produce a local RED before the Linux run.
    def make_mock(self, name, config):
        mock_directory = self.path("mock-" + name)
        os.makedirs(mock_directory)
        with open(os.path.join(mock_directory, "config.json"), "w") as config_file:
            json.dump(config, config_file)

        if os.name == "nt":
            implementation = os.path.join(mock_directory, name + "-impl.py")
            self._write_mock_script(implementation)
            executable = os.path.join(mock_directory, name + ".cmd")
            with open(executable, "w", newline="") as wrapper:
                wrapper.write("@echo off\r\n")
                wrapper.write('"%s" "%s" %%*\r\n' % (
                    sys.executable, implementation))
            return executable

        executable = os.path.join(mock_directory, name)
        with open(executable, "w", newline="\n") as script_file:
            script_file.write("#!" + sys.executable + "\n")
            script_file.write(MOCK_SCRIPT)
        os.chmod(executable, 0o755)
        return executable

    # Returns every invocation record written by one mock executable.
    def records(self, executable):
        record_path = os.path.join(os.path.dirname(executable),
                                   "record.jsonl")
        if not os.path.exists(record_path):
            return []
        with open(record_path, "r") as record_file:
            return [json.loads(line) for line in record_file if line.strip()]

    # Builds a successful mock configuration with exact stdout lines.
    def correct_config(self, predictions, sleep_seconds=0.0):
        return {
            "stdout_lines": [str(value) for value in predictions],
            "sleep_seconds": sleep_seconds,
            "exit_code": 0,
        }

    # Writes headerless little-endian input arrays for external checking.
    def make_inputs(self, labels):
        images_path = self.path("images.f32")
        with open(images_path, "wb") as images_file:
            images_file.write(b"\x00" * (len(labels) * IMAGE_PIXELS * 4))
        labels_path = self.path("labels.i32")
        with open(labels_path, "wb") as labels_file:
            labels_file.write(struct.pack(
                "<%di" % len(labels), *labels))
        return images_path, labels_path

    # Writes a validated-looking bundle with an explicit persisted input mode.
    def make_model(self, name="model", input_mode="byte"):
        model_dir = self.path(name)
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, "lenet.bin"), "wb") as model_file:
            model_file.write(b"course model weights\n")
        with open(os.path.join(model_dir, "course.meta"), "w") as meta_file:
            meta_file.write(
                "course_lenet_v1\n"
                "input_mode={0}\n"
                "shape=1,28,28,10\n"
                "checkpoint_crc32=00000000\n".format(input_mode))
        return model_dir

    # Invokes the proposed helper API with every parameter named explicitly.
    def benchmark(self, executable, images_path, labels_path, model_dir,
                  output_json, count, batch=1024, runs=5,
                  model_source="legacy", profile=False,
                  reference_predictions=None):
        return course_benchmark.run_benchmark(
            infer_path=executable,
            images_path=images_path,
            labels_path=labels_path,
            model_dir=model_dir,
            output_json=output_json,
            count=count,
            batch=batch,
            runs=runs,
            model_source=model_source,
            profile=profile,
            reference_predictions=reference_predictions,
        )

    def read_json(self, path):
        with open(path, "r") as json_file:
            return json.load(json_file)

    # Runs the documented CLI in a stdlib-only interpreter.
    def run_cli(self, arguments):
        return subprocess.run(
            [sys.executable, "-S", "-m", "scripts.course_benchmark"] +
            list(arguments),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    # The pinned schema, hashes, persisted mode, argv isolation, and summary.
    def test_json_schema_hashes_and_official_argv_only(self):
        labels = [0, 4, 9]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "correct", self.correct_config(labels))
        output = self.output_path()

        result = self.benchmark(
            executable, images_path, labels_path, model_dir, output,
            count=len(labels), batch=17, runs=2, model_source="legacy")

        self.assertEqual(BENCHMARK_KEYS, set(result))
        self.assertEqual("course-benchmark-v1", result["schema"])
        self.assertEqual(executable, result["executable"])
        self.assertEqual(images_path, result["images_path"])
        self.assertEqual(labels_path, result["labels_path"])
        self.assertEqual(model_dir, result["model_dir"])
        self.assertEqual("legacy", result["model_source"])
        self.assertEqual(3, result["n"])
        self.assertEqual(17, result["batch"])
        self.assertEqual("byte", result["input_mode"])
        self.assertIs(False, result["profiled"])
        self.assertEqual(2, result["runs_requested"])

        self.assertEqual(SHA256_KEYS, set(result["sha256"]))
        self.assertEqual(file_digest(executable), result["sha256"]["binary"])
        self.assertEqual(
            file_digest(os.path.join(model_dir, "lenet.bin")),
            result["sha256"]["model"])
        self.assertEqual(
            file_digest(os.path.join(model_dir, "course.meta")),
            result["sha256"]["meta"])
        self.assertEqual(file_digest(images_path),
                         result["sha256"]["images"])
        self.assertEqual(file_digest(labels_path),
                         result["sha256"]["labels"])

        self.assertEqual(2, len(result["runs"]))
        for run in result["runs"]:
            self.assertEqual(RUN_KEYS, set(run))
            self.assertEqual(0, run["returncode"])
            self.assertEqual(3, run["correct"])
            self.assertEqual("0\n4\n9\n", run["stdout"])
            self.assertEqual("", run["stderr"])
            self.assertGreaterEqual(run["wall_ms"], 0.0)

        summary = result["summary"]
        self.assertEqual(SUMMARY_KEYS, set(summary))
        self.assertEqual(3, summary["correct"])
        self.assertEqual(1.0, summary["accuracy"])
        self.assertIs(True, summary["accuracy_pass"])
        walls = [run["wall_ms"] for run in result["runs"]]
        self.assertAlmostEqual(max(walls), summary["max_ms"], places=3)
        self.assertAlmostEqual(statistics.median(walls),
                               summary["median_ms"], places=3)
        self.assertEqual(all(wall < 100.0 for wall in walls),
                         summary["all_wall_below_100ms"])

        self.assertEqual(result, self.read_json(output))

        records = self.records(executable)
        self.assertEqual(2, len(records))
        expected_argv = [images_path, "3", "1", "28", "28", "10"]
        for record in records:
            self.assertEqual(expected_argv, record["argv"])
            self.assertNotIn(labels_path, record["argv"])
            self.assertEqual(model_dir, record["COURSE_MODEL_DIR"])
            self.assertEqual("17", record["COURSE_INFER_BATCH"])

    # Profiling is a separate diagnostic artifact: every requested sample in
    # that call is profiled and retained, with no hidden extra process.
    def test_profile_run_is_separate_and_inherited_profile_is_cleared(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        environment = {
            "COURSE_PROFILE": "1",
            "COURSE_INPUT_MODE": "unit",
            "COURSE_EPOCHS": "2",
        }

        fast_executable = self.make_mock(
            "fast", self.correct_config(labels))
        with mock.patch.dict(os.environ, environment):
            fast_result = self.benchmark(
                fast_executable, images_path, labels_path, model_dir,
                self.output_path("fast.json"),
                count=len(labels), runs=2, model_source="course_train")

        self.assertIs(False, fast_result["profiled"])
        self.assertEqual("course_train", fast_result["model_source"])
        self.assertEqual(2, len(fast_result["runs"]))
        fast_records = self.records(fast_executable)
        self.assertEqual(2, len(fast_records))
        for record in fast_records:
            self.assertIsNone(record["COURSE_PROFILE"])
            self.assertIsNone(record["COURSE_INPUT_MODE"])
            self.assertIsNone(record["COURSE_EPOCHS"])
            self.assertEqual(model_dir, record["COURSE_MODEL_DIR"])
            self.assertEqual("1024", record["COURSE_INFER_BATCH"])

        profiled_executable = self.make_mock(
            "profiled", self.correct_config(labels))
        with mock.patch.dict(os.environ, environment):
            profiled_result = self.benchmark(
                profiled_executable, images_path, labels_path, model_dir,
                self.output_path("profiled.json"),
                count=len(labels), runs=2, model_source="legacy",
                profile=True)

        self.assertIs(True, profiled_result["profiled"])
        self.assertEqual(2, len(profiled_result["runs"]))
        profiled_records = self.records(profiled_executable)
        self.assertEqual(2, len(profiled_records))
        for record in profiled_records:
            self.assertEqual("1", record["COURSE_PROFILE"])
            self.assertEqual(
                [images_path, "3", "1", "28", "28", "10"],
                record["argv"])
            self.assertEqual(model_dir, record["COURSE_MODEL_DIR"])
            self.assertEqual("1024", record["COURSE_INFER_BATCH"])
        self.assertIs(True, self.read_json(
            self.output_path("profiled.json"))["profiled"])

    # Wrong, extra, partial, non-integer, and out-of-range predictions fail.
    def test_malformed_predictions_never_produce_a_passing_summary(self):
        labels = [0, 1, 2]
        cases = [
            ("wrong", ["1", "2", "0"], 3),
            ("extra", ["0", "1", "2", "3"], 4),
            ("partial", ["0", "1"], 2),
            ("noninteger", ["0", "one", "2"], 3),
            ("unicode-digit", ["0", "\u00b2", "2"], 3),
            ("out-of-range", ["0", "10", "2"], 3),
            ("negative", ["0", "-1", "2"], 3),
        ]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()

        for name, lines, line_count in cases:
            with self.subTest(case=name):
                executable = self.make_mock(name, {
                    "stdout_lines": lines,
                    "exit_code": 0,
                })
                output = self.output_path("invalid-" + name + ".json")
                result = self.benchmark(
                    executable, images_path, labels_path, model_dir, output,
                    count=len(labels), runs=1, model_source="legacy")

                run = result["runs"][0]
                self.assertEqual(0, run["returncode"])
                self.assertEqual(0, run["correct"])
                self.assertEqual(line_count, len(run["stdout"].splitlines()))
                self.assertIs(False, result["summary"]["accuracy_pass"])
                self.assertEqual(0, result["summary"]["correct"])
                self.assertEqual(0.0, result["summary"]["accuracy"])
                self.assertIs(False, self.read_json(output)["summary"][
                    "accuracy_pass"])

    # Repeated successful runs must produce identical predictions.
    def test_repeated_runs_with_different_predictions_do_not_pass(self):
        labels = [0] * 100
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        almost = [0] * 99 + [1]
        executable = self.make_mock("repeated", {
            "sequence": [
                self.correct_config(labels),
                self.correct_config(almost),
            ],
        })
        output = self.output_path("repeated.json")

        result = self.benchmark(
            executable, images_path, labels_path, model_dir, output,
            count=len(labels), runs=2, model_source="legacy")

        self.assertEqual(2, len(result["runs"]))
        self.assertEqual(100, result["runs"][0]["correct"])
        self.assertEqual(99, result["runs"][1]["correct"])
        self.assertNotEqual(result["runs"][0]["stdout"],
                            result["runs"][1]["stdout"])
        self.assertIs(False, result["summary"]["accuracy_pass"])
        self.assertIs(False, self.read_json(output)["summary"][
            "accuracy_pass"])

    # A nonzero process exit is retained and can never become a pass.
    def test_nonzero_exit_is_retained_and_not_a_pass(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock("nonzero", {
            "sequence": [
                self.correct_config(labels),
                {
                    "stdout_lines": ["0", "1", "2"],
                    "stderr": "mock failure \u00e9\n",
                    "exit_code": 3,
                },
            ],
        })
        output = self.output_path("nonzero.json")

        result = self.benchmark(
            executable, images_path, labels_path, model_dir, output,
            count=len(labels), runs=2, model_source="legacy")

        self.assertEqual(2, len(result["runs"]))
        failed = result["runs"][1]
        self.assertEqual(3, failed["returncode"])
        self.assertEqual("0\n1\n2\n", failed["stdout"])
        self.assertEqual("mock failure \u00e9\n", failed["stderr"])
        self.assertEqual(0, failed["correct"])
        self.assertIs(False, result["summary"]["accuracy_pass"])
        self.assertIs(False, self.read_json(output)["summary"][
            "accuracy_pass"])

    # Every slow sample is retained, including an outlier above 100 ms.
    def test_all_slow_samples_are_retained_with_max_and_median(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        sequence = [self.correct_config(labels) for _ in range(5)]
        sequence[2] = self.correct_config(labels, sleep_seconds=0.12)
        executable = self.make_mock("slow", {"sequence": sequence})
        output = self.output_path("slow.json")

        result = self.benchmark(
            executable, images_path, labels_path, model_dir, output,
            count=len(labels), runs=5, model_source="legacy")

        walls = [run["wall_ms"] for run in result["runs"]]
        self.assertEqual(5, len(walls))
        self.assertGreaterEqual(walls[2], 100.0)
        self.assertAlmostEqual(max(walls), result["summary"]["max_ms"],
                               places=3)
        self.assertAlmostEqual(statistics.median(walls),
                               result["summary"]["median_ms"], places=3)
        self.assertIs(False, result["summary"]["all_wall_below_100ms"])
        for run in result["runs"]:
            self.assertEqual("0\n1\n2\n", run["stdout"])

    # The optional reference is exactly N uint8 predictions, never labels.
    def test_reference_uint8_validation_and_mismatch_count(self):
        labels = [0, 1, 2, 3]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "reference", self.correct_config(labels))

        reference = self.path("reference.u8")
        with open(reference, "wb") as reference_file:
            reference_file.write(bytes(bytearray([0, 1, 2, 9])))
        output = self.output_path("reference.json")

        result = self.benchmark(
            executable, images_path, labels_path, model_dir, output,
            count=len(labels), runs=1, model_source="legacy",
            reference_predictions=reference)

        self.assertEqual(BENCHMARK_KEYS | REFERENCE_KEYS, set(result))
        self.assertEqual(file_digest(reference),
                         result["reference_sha256"])
        self.assertEqual(1, result["reference_mismatches"])
        self.assertIs(True, result["summary"]["accuracy_pass"])

        exact = self.path("reference-exact.u8")
        with open(exact, "wb") as reference_file:
            reference_file.write(bytes(bytearray(labels)))
        exact_result = self.benchmark(
            executable, images_path, labels_path, model_dir,
            self.output_path("reference-exact.json"),
            count=len(labels), runs=1, model_source="legacy",
            reference_predictions=exact)
        self.assertEqual(0, exact_result["reference_mismatches"])

        invalid_references = [
            ("short", bytes(bytearray([0, 1, 2]))),
            ("long", bytes(bytearray([0, 1, 2, 3, 0]))),
            ("labels-file", struct.pack("<4i", 0, 1, 2, 3)),
            ("out-of-range", bytes(bytearray([0, 1, 2, 10]))),
        ]
        for name, content in invalid_references:
            with self.subTest(reference=name):
                path = self.path("invalid-" + name + ".u8")
                with open(path, "wb") as reference_file:
                    reference_file.write(content)
                before = len(self.records(executable))
                output = self.output_path("invalid-" + name + ".json")
                with self.assertRaises(ValueError):
                    self.benchmark(
                        executable, images_path, labels_path, model_dir,
                        output, count=len(labels), runs=1,
                        model_source="legacy",
                        reference_predictions=path)
                self.assertFalse(os.path.exists(output))
                self.assertEqual(before, len(self.records(executable)))

    # Existing output JSON is immutable and is never truncated or replaced.
    def test_existing_output_json_is_never_overwritten(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "immutable", self.correct_config(labels))
        output = self.output_path("immutable.json")
        sentinel = b'{"schema": "must-not-change"}\n'
        with open(output, "wb") as existing:
            existing.write(sentinel)

        with self.assertRaises(FileExistsError):
            self.benchmark(
                executable, images_path, labels_path, model_dir, output,
                count=len(labels), runs=1, model_source="legacy")

        with open(output, "rb") as existing:
            self.assertEqual(sentinel, existing.read())
        self.assertEqual([], self.records(executable))

    # A symlink alias of the output JSON must not write through to its target.
    def test_symlinked_output_json_target_is_never_overwritten(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "immutable-alias", self.correct_config(labels))
        sentinel = b'{"schema": "must-not-change"}\n'
        target = self.output_path("alias-target.json")
        with open(target, "wb") as target_file:
            target_file.write(sentinel)
        alias = self.output_path("alias-output.json")
        try:
            os.symlink(target, alias)
        except (OSError, NotImplementedError):
            self.skipTest("file symlinks are unavailable on this host")

        with self.assertRaises(FileExistsError):
            self.benchmark(
                executable, images_path, labels_path, model_dir, alias,
                count=len(labels), runs=1, model_source="legacy")

        self.assertTrue(os.path.islink(alias))
        with open(target, "rb") as target_file:
            self.assertEqual(sentinel, target_file.read())
        self.assertEqual([], self.records(executable))

    # Labels are the external checker's int32 array and must be exactly N.
    def test_invalid_label_files_are_rejected_before_any_run(self):
        labels = [0, 1, 2]
        images_path, _ = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "bad-labels", self.correct_config(labels))
        cases = [
            ("short", struct.pack("<2i", 0, 1)),
            ("long", struct.pack("<4i", 0, 1, 2, 3)),
            ("out-of-range", struct.pack("<3i", 0, 10, 2)),
        ]
        for name, content in cases:
            with self.subTest(case=name):
                labels_path = self.path("bad-labels-" + name + ".i32")
                with open(labels_path, "wb") as labels_file:
                    labels_file.write(content)
                output = self.output_path("bad-labels-" + name + ".json")
                with self.assertRaises(ValueError):
                    self.benchmark(
                        executable, images_path, labels_path, model_dir,
                        output, count=len(labels), runs=1,
                        model_source="legacy")
                self.assertFalse(os.path.exists(output))
                self.assertEqual([], self.records(executable))

    # CLI defaults are R5/B1024; a successful run exits zero and writes JSON.
    def test_cli_defaults_to_five_runs_and_batch_1024(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "cli-success", self.correct_config(labels))
        output = self.output_path("cli-success.json")

        result = self.run_cli([
            executable, images_path, labels_path, model_dir, output,
            "--count", "3", "--model-source", "legacy",
        ])

        self.assertEqual(0, result.returncode, result.stderr)
        benchmark = self.read_json(output)
        self.assertEqual(1024, benchmark["batch"])
        self.assertEqual(5, benchmark["runs_requested"])
        self.assertEqual(5, len(benchmark["runs"]))
        self.assertIs(True, benchmark["summary"]["accuracy_pass"])

    # A slow but correct CLI measurement is truthful, not a pass/fail mismatch:
    # the wall flag is false while the correctness acceptance exit is zero.
    def test_cli_slow_correct_run_keeps_wall_flag_separate(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "cli-slow", self.correct_config(labels, sleep_seconds=0.12))
        output = self.output_path("cli-slow.json")

        result = self.run_cli([
            executable, images_path, labels_path, model_dir, output,
            "--count", "3", "--runs", "1", "--model-source", "legacy",
        ])

        self.assertEqual(0, result.returncode, result.stderr)
        benchmark = self.read_json(output)
        self.assertIs(True, benchmark["summary"]["accuracy_pass"])
        self.assertIs(False, benchmark["summary"]["all_wall_below_100ms"])

    # CLI failures exit nonzero but preserve truthful failure JSON.
    def test_cli_failure_exit_codes_and_reference_mismatch(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        wrong_executable = self.make_mock(
            "cli-wrong", self.correct_config([1, 2, 0]))
        wrong_output = self.output_path("cli-wrong.json")

        result = self.run_cli([
            wrong_executable, images_path, labels_path, model_dir,
            wrong_output, "--count", "3", "--runs", "1",
            "--model-source", "legacy",
        ])
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIs(False, self.read_json(wrong_output)["summary"][
            "accuracy_pass"])

        executable = self.make_mock(
            "cli-reference", self.correct_config(labels))
        reference = self.path("cli-reference.u8")
        with open(reference, "wb") as reference_file:
            reference_file.write(bytes(bytearray([0, 1, 9])))
        reference_output = self.output_path("cli-reference.json")
        result = self.run_cli([
            executable, images_path, labels_path, model_dir,
            reference_output, "--count", "3", "--runs", "1",
            "--model-source", "legacy", "--reference-predictions", reference,
        ])
        self.assertNotEqual(0, result.returncode, result.stdout)
        benchmark = self.read_json(reference_output)
        self.assertEqual(1, benchmark["reference_mismatches"])
        self.assertIs(True, benchmark["summary"]["accuracy_pass"])

        existing = self.output_path("cli-existing.json")
        sentinel = b"do not overwrite\n"
        with open(existing, "wb") as existing_file:
            existing_file.write(sentinel)
        result = self.run_cli([
            executable, images_path, labels_path, model_dir, existing,
            "--count", "3", "--runs", "1", "--model-source", "legacy",
        ])
        self.assertNotEqual(0, result.returncode, result.stdout)
        with open(existing, "rb") as existing_file:
            self.assertEqual(sentinel, existing_file.read())

    # Missing required options and an unknown model source are usage errors.
    def test_cli_usage_errors_exit_two_without_output(self):
        labels = [0, 1, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "cli-usage", self.correct_config(labels))
        cases = [
            ("missing-count", [
                executable, images_path, labels_path, model_dir,
                self.output_path("usage-missing-count.json"),
                "--model-source", "legacy"]),
            ("missing-model-source", [
                executable, images_path, labels_path, model_dir,
                self.output_path("usage-missing-source.json"),
                "--count", "3"]),
            ("unknown-model-source", [
                executable, images_path, labels_path, model_dir,
                self.output_path("usage-unknown-source.json"),
                "--count", "3", "--model-source", "baseline"]),
        ]
        for name, arguments in cases:
            with self.subTest(case=name):
                result = self.run_cli(arguments)
                self.assertEqual(2, result.returncode, result.stderr)
                self.assertTrue(result.stderr)
                self.assertFalse(os.path.exists(arguments[4]))


if __name__ == "__main__":
    unittest.main()
