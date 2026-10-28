from __future__ import print_function

import hashlib
import inspect
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

from contextlib import redirect_stderr

from scripts import course_warm_benchmark as warm

try:
    from unittest import mock
except ImportError:  # pragma: no cover - Python 3.6 includes unittest.mock
    import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PIXELS = 28 * 28
CLASS_COUNT = 10

RESULT_KEYS = {
    "schema",
    "timing_scope",
    "executable",
    "official_infer",
    "images_path",
    "labels_path",
    "model_dir",
    "model_source",
    "n",
    "batch",
    "input_mode",
    "warmups",
    "runs",
    "sha256",
    "sha256_after",
    "hashes_stable",
    "returncode",
    "process_wall_ms",
    "stdout",
    "stderr",
    "warmup_records",
    "measured_records",
    "predictions_identical_all",
    "status",
    "errors",
    "summary",
}
HASH_KEYS = {"binary", "official_infer", "model", "meta", "images", "labels"}
SUMMARY_KEYS = {
    "mean_ms",
    "median_ms",
    "min_ms",
    "max_ms",
    "correct",
    "accuracy",
    "accuracy_pass",
    "mean_below_100ms",
    "median_below_100ms",
}
ROUND_KEYS = {
    "iteration",
    "samples",
    "wall_ms",
    "predictions_identical",
}
PARSED_KEYS = {
    "warmup_records",
    "measured_records",
    "predictions",
    "completion",
}
COMPLETION_KEYS = {
    "status",
    "warmups",
    "runs",
    "samples",
    "batch",
    "input_mode",
}


# A harmless real subprocess used in place of the compiled warm tool. It records
# the exact argv and COURSE_* environment it receives, then emits the configured
# output and exit status. It performs no numerical work and never invents
# per-round stdout vectors: the configured stdout is emitted once, exactly like
# the reviewed tool's single final N-line vector.
MOCK_SCRIPT = r'''
import json
import os
import sys
import time

directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(directory, "config.json"), "r") as config_file:
    config = json.load(config_file)

record_path = os.path.join(directory, "record.jsonl")
record = {
    "argv": sys.argv[1:],
    "course_env": dict(
        (key, value) for key, value in os.environ.items()
        if key.startswith("COURSE_")),
    "cwd": os.getcwd(),
}
with open(record_path, "a") as record_file:
    record_file.write(json.dumps(record) + "\n")

if config.get("stderr"):
    sys.stderr.buffer.write(config["stderr"].encode("utf-8"))
    sys.stderr.buffer.flush()
if config.get("stdout"):
    sys.stdout.buffer.write(config["stdout"].encode("utf-8"))
    sys.stdout.buffer.flush()

mutation = config.get("mutate")
if mutation:
    with open(mutation["path"], mutation.get("mode", "a")) as mutated:
        mutated.write(mutation.get("data", ""))

if config.get("close_streams"):
    os.close(1)
    os.close(2)

if config.get("sleep_seconds"):
    time.sleep(config["sleep_seconds"])

sys.exit(config.get("exit_code", 0))
'''


# Returns the exact reviewed-tool round record line.
def iteration_line(phase, iteration, samples, wall_ms, identical=1):
    return (
        "event=course_warm_iteration phase={0} iteration={1} samples={2} "
        "wall_ms={3} predictions_identical={4}\n"
    ).format(phase, iteration, samples, wall_ms, identical)


# Returns the exact reviewed-tool completion record line.
def completion_line(warmups, runs, samples, batch, input_mode, status="pass"):
    return (
        "event=course_warm_complete status={0} warmups={1} runs={2} "
        "samples={3} batch={4} input_mode={5}\n"
    ).format(status, warmups, runs, samples, batch, input_mode)


# Builds one complete protocol from literal synthetic durations.
def protocol_text(warmups, runs, samples, batch, input_mode,
                  warm_walls=None, measure_walls=None):
    if warm_walls is None:
        warm_walls = tuple(1000 + index for index in range(warmups))
    if measure_walls is None:
        measure_walls = tuple(10 * (index + 1) for index in range(runs))
    lines = [
        iteration_line("warmup", index + 1, samples, warm_walls[index])
        for index in range(warmups)
    ]
    lines.extend(
        iteration_line("measure", index + 1, samples, measure_walls[index])
        for index in range(runs)
    )
    lines.append(completion_line(warmups, runs, samples, batch, input_mode))
    return "".join(lines)


# Renders one final N-line prediction vector.
def prediction_text(values):
    return "".join("{0}\n".format(value) for value in values)


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


class WarmBenchmarkTest(unittest.TestCase):
    # Every test owns its executable, model, inputs, and output JSON.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def path(self, name):
        return os.path.join(self.directory, name)

    def output_path(self, name="warm.json"):
        return self.path(name)

    # Writes the stdlib mock implementation and returns a native executable
    # path for both POSIX and Windows.
    def make_mock(self, name, config):
        mock_directory = self.path("mock-" + name)
        os.makedirs(mock_directory)
        with open(os.path.join(mock_directory, "config.json"), "w") as config_file:
            json.dump(config, config_file)

        if os.name == "nt":
            implementation = os.path.join(mock_directory, name + "-impl.py")
            with open(implementation, "w", newline="\n") as script_file:
                script_file.write(MOCK_SCRIPT)
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
    def invocations(self, executable):
        record_path = os.path.join(os.path.dirname(executable),
                                   "record.jsonl")
        if not os.path.exists(record_path):
            return []
        with open(record_path, "r") as record_file:
            return [json.loads(line) for line in record_file if line.strip()]

    # Builds a successful mock configuration with one final stdout vector.
    def success_config(self, predictions, warmups, runs, samples, batch,
                       input_mode, warm_walls=None, measure_walls=None):
        return {
            "stdout": prediction_text(predictions),
            "stderr": protocol_text(
                warmups, runs, samples, batch, input_mode,
                warm_walls=warm_walls, measure_walls=measure_walls),
            "exit_code": 0,
        }

    # Writes headerless little-endian input arrays for external checking.
    def make_inputs(self, labels, image_bytes=None):
        if image_bytes is None:
            image_bytes = len(labels) * IMAGE_PIXELS * 4
        images_path = self.path("images.f32")
        with open(images_path, "wb") as images_file:
            images_file.write(b"\x00" * image_bytes)
        labels_path = self.path("labels.i32")
        with open(labels_path, "wb") as labels_file:
            labels_file.write(struct.pack("<%di" % len(labels), *labels))
        return images_path, labels_path

    # Writes a validated-looking bundle with an explicit persisted input mode.
    def make_model(self, name="model", input_mode="normalized"):
        model_dir = self.path(name)
        os.makedirs(model_dir)
        with open(os.path.join(model_dir, "lenet.bin"), "wb") as model_file:
            model_file.write(b"course model weights\n")
        with open(os.path.join(model_dir, "course.meta"), "w",
                  newline="\n") as meta_file:
            meta_file.write(
                "course_lenet_v1\n"
                "input_mode={0}\n"
                "shape=1,28,28,10\n"
                "checkpoint_crc32=00000000\n".format(input_mode))
        return model_dir

    # Writes the official-infer identity input; it is hashed, never executed.
    def make_official_infer(self):
        official = self.path("infer")
        with open(official, "wb") as infer_file:
            infer_file.write(b"official inference executable placeholder\n")
        return official

    # Invokes the proposed reusable parser API positionally.
    def parse(self, stdout, stderr, returncode=0, count=3, batch=2,
              warmups=2, runs=3, input_mode="normalized"):
        return warm.parse_tool_output(
            stdout, stderr, returncode, count, batch, warmups, runs,
            input_mode)

    # Invokes the proposed collector API with every parameter named.
    def benchmark(self, warm_exe, infer_exe, images_path, labels_path,
                  model_dir, output_json, count, batch=1024, warmups=5,
                  runs=101, model_source="course_train"):
        return warm.run_benchmark(
            warm_exe=warm_exe,
            infer_exe=infer_exe,
            images=images_path,
            labels=labels_path,
            model_dir=model_dir,
            output_json=output_json,
            count=count,
            batch=batch,
            warmups=warmups,
            runs=runs,
            model_source=model_source,
        )

    def read_json(self, path):
        with open(path, "r") as json_file:
            return json.load(json_file)

    # Runs the documented CLI in a stdlib-only interpreter.
    def run_cli(self, arguments):
        return subprocess.run(
            [sys.executable, "-S", "-m",
             "scripts.course_warm_benchmark"] + list(arguments),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    # Shared strict assertions for every collection-failure JSON.
    def assert_collection_failure(self, result, output):
        self.assertEqual(RESULT_KEYS, set(result))
        self.assertEqual("fail", result["status"])
        self.assertTrue(result["errors"])
        self.assertFalse(result["predictions_identical_all"])
        self.assertEqual(SUMMARY_KEYS, set(result["summary"]))
        self.assertIsNone(result["summary"]["mean_ms"])
        self.assertIsNone(result["summary"]["median_ms"])
        self.assertIsNone(result["summary"]["min_ms"])
        self.assertIsNone(result["summary"]["max_ms"])
        self.assertEqual(0, result["summary"]["correct"])
        self.assertEqual(0.0, result["summary"]["accuracy"])
        self.assertFalse(result["summary"]["accuracy_pass"])
        self.assertFalse(result["summary"]["mean_below_100ms"])
        self.assertFalse(result["summary"]["median_below_100ms"])
        self.assertTrue(os.path.exists(output))
        self.assertEqual(result, self.read_json(output))


class ParseToolOutputTest(WarmBenchmarkTest):
    # The reusable parser name, order, and schema are pinned for Task 3.
    def test_parser_signature_and_constants(self):
        parameters = list(inspect.signature(warm.parse_tool_output).parameters)
        self.assertEqual(
            ["stdout", "stderr", "returncode", "count", "batch",
             "warmups", "runs", "input_mode"],
            parameters)
        self.assertEqual("course-warm-benchmark-v1",
                         warm.WARM_BENCHMARK_SCHEMA)
        self.assertEqual(
            "warm_full_request_h2d_prepare_forward_argmax_d2h_validation",
            warm.TIMING_SCOPE)
        self.assertEqual(0.96, warm.ACCURACY_THRESHOLD)
        self.assertEqual(600, warm.PROCESS_TIMEOUT_SECONDS)
        self.assertEqual(1024, warm.DEFAULT_BATCH)
        self.assertEqual(5, warm.DEFAULT_WARMUPS)
        self.assertEqual(101, warm.DEFAULT_RUNS)
        self.assertEqual(100.0, warm.FULL_REQUEST_LIMIT_MS)
        self.assertEqual(("legacy", "course_train"), warm.MODEL_SOURCES)
        self.assertEqual(("unit", "byte", "normalized"), warm.INPUT_MODES)
        self.assertEqual((1, 28, 28, 10), warm.SHAPE)

    # The complete W2/R3 protocol is accepted with every raw record retained.
    def test_accepts_exact_protocol_and_retains_all_records(self):
        stdout = prediction_text([1, 0, 2])
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        parsed = self.parse(stdout, stderr)

        self.assertEqual(PARSED_KEYS, set(parsed))
        self.assertEqual([1, 0, 2], parsed["predictions"])
        self.assertEqual(
            [
                {"iteration": 1, "samples": 3, "wall_ms": 1000.0,
                 "predictions_identical": True},
                {"iteration": 2, "samples": 3, "wall_ms": 2000.0,
                 "predictions_identical": True},
            ],
            parsed["warmup_records"])
        self.assertEqual(
            [
                {"iteration": 1, "samples": 3, "wall_ms": 10.0,
                 "predictions_identical": True},
                {"iteration": 2, "samples": 3, "wall_ms": 20.0,
                 "predictions_identical": True},
                {"iteration": 3, "samples": 3, "wall_ms": 40.0,
                 "predictions_identical": True},
            ],
            parsed["measured_records"])
        self.assertEqual(COMPLETION_KEYS, set(parsed["completion"]))
        self.assertEqual(
            {
                "status": "pass",
                "warmups": 2,
                "runs": 3,
                "samples": 3,
                "batch": 2,
                "input_mode": "normalized",
            },
            parsed["completion"])
        for record in (parsed["warmup_records"] +
                       parsed["measured_records"]):
            self.assertEqual(ROUND_KEYS, set(record))

    # Bounded W1/R1/N1 counts are parser-valid; the >=101 gate belongs to the
    # later packager, not this reusable parser.
    def test_accepts_bounded_minimum_counts(self):
        stdout = prediction_text([9])
        stderr = protocol_text(
            1, 1, 1, 1, "unit", warm_walls=(3,), measure_walls=(7,))
        parsed = self.parse(
            stdout, stderr, count=1, batch=1, warmups=1, runs=1,
            input_mode="unit")
        self.assertEqual([9], parsed["predictions"])
        self.assertEqual(1, len(parsed["warmup_records"]))
        self.assertEqual(1, len(parsed["measured_records"]))

    def test_rejects_nonzero_returncode(self):
        stdout = prediction_text([1, 0, 2])
        stderr = protocol_text(2, 3, 3, 2, "normalized")
        with self.assertRaises(ValueError):
            self.parse(stdout, stderr, returncode=1)

    # Missing, extra, duplicated, out-of-order, or wrongly-phased records fail.
    def test_rejects_missing_extra_duplicate_or_misordered_rounds(self):
        stdout = prediction_text([1, 0, 2])
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        lines = stderr.splitlines(True)
        self.assertEqual(6, len(lines))

        variants = {}
        variants["missing_warmup"] = lines[1:]
        variants["missing_measure"] = lines[:3] + lines[4:]
        variants["extra_warmup"] = lines[:2] + [lines[1]] + lines[2:]
        variants["extra_measure"] = lines[:5] + [lines[4]] + lines[5:]
        variants["duplicate_measure"] = lines[:4] + [lines[3]] + lines[4:]
        variants["zero_based_measure"] = (
            lines[:2] + [lines[2].replace("iteration=1", "iteration=0")] +
            lines[3:])
        swapped = list(lines)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        variants["reversed_warmups"] = swapped
        variants["measure_before_warmup"] = (
            [lines[2]] + lines[:2] + lines[3:])
        variants["unknown_phase"] = (
            [lines[0].replace("phase=warmup", "phase=warm")] + lines[1:])
        variants["wrong_samples"] = (
            [lines[0].replace("samples=3", "samples=4")] + lines[1:])

        for name, variant_lines in variants.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.parse(stdout, "".join(variant_lines))

    # Every round field is strict: exact count/order/keys, positive finite time,
    # and an explicit true invariance flag.
    def test_rejects_bad_round_fields_and_times(self):
        stdout = prediction_text([1, 0, 2])
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        lines = stderr.splitlines(True)

        variants = {}
        for name, bad_time in (
                ("zero", "0"),
                ("negative", "-1"),
                ("nan", "nan"),
                ("inf", "inf"),
                ("negative_inf", "-inf"),
                ("text", "abc"),
                ("empty", "")):
            variants["time_" + name] = (
                [lines[0].replace("wall_ms=1000", "wall_ms=" + bad_time)] +
                lines[1:])
        variants["identical_zero"] = (
            [lines[0].replace("predictions_identical=1",
                              "predictions_identical=0")] + lines[1:])
        variants["identical_word"] = (
            [lines[0].replace("predictions_identical=1",
                              "predictions_identical=true")] + lines[1:])
        variants["missing_field"] = (
            [lines[0].replace(" predictions_identical=1", "")] + lines[1:])
        variants["extra_field"] = (
            [lines[0].replace("\n", " extra=1\n")] + lines[1:])
        variants["duplicate_field"] = (
            [lines[0].replace("\n", " phase=warmup\n")] + lines[1:])
        variants["reordered_fields"] = (
            [lines[0].replace(
                "samples=3 wall_ms=1000",
                "wall_ms=1000 samples=3")] + lines[1:])
        variants["wrong_event"] = (
            [lines[0].replace("event=course_warm_iteration",
                              "event=course_cold_iteration")] + lines[1:])

        for name, variant_lines in variants.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.parse(stdout, "".join(variant_lines))

    # Completion is required exactly once, last, and must bind every requested
    # count, the batch, the persisted mode, and the passing status.
    def test_rejects_bad_completion(self):
        stdout = prediction_text([1, 0, 2])
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        lines = stderr.splitlines(True)
        completion = lines[-1]
        rounds = lines[:-1]

        variants = {}
        variants["missing"] = rounds
        variants["trailing_junk"] = lines + ["junk\n"]
        variants["blank_trailing"] = lines + ["\n"]
        variants["status_fail"] = rounds + [
            completion.replace("status=pass", "status=fail")]
        variants["warmups_mismatch"] = rounds + [
            completion.replace("warmups=2", "warmups=1")]
        variants["runs_mismatch"] = rounds + [
            completion.replace("runs=3", "runs=2")]
        variants["samples_mismatch"] = rounds + [
            completion.replace("samples=3", "samples=4")]
        variants["batch_mismatch"] = rounds + [
            completion.replace("batch=2", "batch=3")]
        variants["mode_mismatch"] = rounds + [
            completion.replace("input_mode=normalized", "input_mode=byte")]
        variants["mode_unknown"] = rounds + [
            completion.replace("input_mode=normalized", "input_mode=float")]
        variants["extra_field"] = rounds + [
            completion.replace("\n", " extra=1\n")]
        variants["missing_field"] = rounds + [
            completion.replace(" batch=2", "")]
        variants["wrong_event"] = rounds + [
            completion.replace("event=course_warm_complete",
                               "event=course_warm_iteration")]
        variants["duplicate"] = lines + [completion]

        for name, variant_lines in variants.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.parse(stdout, "".join(variant_lines))

    # Stdout is exactly N canonical decimal class lines; it is not a per-round
    # stream and it is never guessed or padded.
    def test_rejects_bad_predictions_stdout(self):
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        variants = {
            "missing_line": "1\n0\n",
            "extra_line": "1\n0\n2\n3\n",
            "empty": "",
            "blank_line": "1\n\n2\n",
            "leading_zero": "01\n0\n2\n",
            "plus_sign": "+1\n0\n2\n",
            "negative": "-1\n0\n2\n",
            "out_of_range": "1\n0\n10\n",
            "fraction": "1\n0\n2.0\n",
            "leading_space": " 1\n0\n2\n",
            "trailing_space": "1 \n0\n2\n",
        }
        for name, stdout in variants.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.parse(stdout, stderr)

    # Request counts and the expected mode are validated before parsing.
    def test_rejects_bad_request_arguments(self):
        stdout = prediction_text([1, 0, 2])
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        arguments = {
            "zero_count": {"count": 0},
            "negative_count": {"count": -1},
            "bool_count": {"count": True},
            "zero_batch": {"batch": 0},
            "zero_warmups": {"warmups": 0},
            "zero_runs": {"runs": 0},
            "unknown_mode": {"input_mode": "float"},
        }
        for name, overrides in arguments.items():
            with self.subTest(name=name):
                kwargs = dict(
                    count=3, batch=2, warmups=2, runs=3,
                    input_mode="normalized")
                kwargs.update(overrides)
                with self.assertRaises(ValueError):
                    self.parse(stdout, stderr, **kwargs)


class WarmCollectorTest(WarmBenchmarkTest):
    # The collector API exactly matches the brief's six paths and five options.
    def test_collector_signature_defaults(self):
        parameters = inspect.signature(warm.run_benchmark).parameters
        self.assertEqual(
            ["warm_exe", "infer_exe", "images", "labels", "model_dir",
             "output_json", "count", "batch", "warmups", "runs",
             "model_source"],
            list(parameters))
        self.assertEqual(1024, parameters["batch"].default)
        self.assertEqual(5, parameters["warmups"].default)
        self.assertEqual(101, parameters["runs"].default)
        self.assertEqual("course_train", parameters["model_source"].default)

    # One real process, all W/R records, exact one-vector stdout, stable hashes,
    # and mean/median/range computed only from the measured rounds.
    def test_success_schema_records_summary_and_argv_isolation(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model(input_mode="normalized")
        official = self.make_official_infer()
        executable = self.make_mock(
            "correct",
            self.success_config(
                labels, 2, 3, 3, 2, "normalized",
                warm_walls=(1000, 2000), measure_walls=(10, 20, 40)))
        expected_stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        output = self.output_path()

        with mock.patch.dict(os.environ, {
                "COURSE_PROFILE": "1",
                "COURSE_INPUT_MODE": "byte",
                "COURSE_EPOCHS": "20",
                "COURSE_SEED": "7",
                "COURSE_LABELS": labels_path}):
            result = self.benchmark(
                executable, official, images_path, labels_path, model_dir,
                output, count=3, batch=2, warmups=2, runs=3,
                model_source="course_train")

        self.assertEqual(RESULT_KEYS, set(result))
        self.assertEqual("course-warm-benchmark-v1", result["schema"])
        self.assertEqual(warm.TIMING_SCOPE, result["timing_scope"])
        self.assertEqual(executable, result["executable"])
        self.assertEqual(official, result["official_infer"])
        self.assertEqual(images_path, result["images_path"])
        self.assertEqual(labels_path, result["labels_path"])
        self.assertEqual(model_dir, result["model_dir"])
        self.assertEqual("course_train", result["model_source"])
        self.assertEqual(3, result["n"])
        self.assertEqual(2, result["batch"])
        self.assertEqual("normalized", result["input_mode"])
        self.assertEqual(2, result["warmups"])
        self.assertEqual(3, result["runs"])
        self.assertEqual(0, result["returncode"])
        self.assertEqual("pass", result["status"])
        self.assertEqual([], result["errors"])
        self.assertTrue(result["hashes_stable"])
        self.assertTrue(result["predictions_identical_all"])
        self.assertIsInstance(result["process_wall_ms"], float)
        self.assertTrue(math.isfinite(result["process_wall_ms"]))
        self.assertGreater(result["process_wall_ms"], 0.0)

        self.assertEqual(HASH_KEYS, set(result["sha256"]))
        self.assertEqual(HASH_KEYS, set(result["sha256_after"]))
        self.assertEqual(result["sha256"], result["sha256_after"])
        self.assertEqual(file_digest(executable), result["sha256"]["binary"])
        self.assertEqual(file_digest(official),
                         result["sha256"]["official_infer"])
        self.assertEqual(
            file_digest(os.path.join(model_dir, "lenet.bin")),
            result["sha256"]["model"])
        self.assertEqual(
            file_digest(os.path.join(model_dir, "course.meta")),
            result["sha256"]["meta"])
        self.assertEqual(file_digest(images_path), result["sha256"]["images"])
        self.assertEqual(file_digest(labels_path), result["sha256"]["labels"])

        self.assertEqual(
            [
                {"iteration": 1, "samples": 3, "wall_ms": 1000.0,
                 "predictions_identical": True},
                {"iteration": 2, "samples": 3, "wall_ms": 2000.0,
                 "predictions_identical": True},
            ],
            result["warmup_records"])
        self.assertEqual(
            [
                {"iteration": 1, "samples": 3, "wall_ms": 10.0,
                 "predictions_identical": True},
                {"iteration": 2, "samples": 3, "wall_ms": 20.0,
                 "predictions_identical": True},
                {"iteration": 3, "samples": 3, "wall_ms": 40.0,
                 "predictions_identical": True},
            ],
            result["measured_records"])

        self.assertEqual(SUMMARY_KEYS, set(result["summary"]))
        self.assertEqual(70.0 / 3.0, result["summary"]["mean_ms"])
        self.assertEqual(20.0, result["summary"]["median_ms"])
        self.assertEqual(10.0, result["summary"]["min_ms"])
        self.assertEqual(40.0, result["summary"]["max_ms"])
        self.assertEqual(3, result["summary"]["correct"])
        self.assertEqual(1.0, result["summary"]["accuracy"])
        self.assertTrue(result["summary"]["accuracy_pass"])
        self.assertTrue(result["summary"]["mean_below_100ms"])
        self.assertTrue(result["summary"]["median_below_100ms"])

        # One final N-line vector only; no per-round stdout copies.
        self.assertEqual(prediction_text(labels), result["stdout"])
        self.assertEqual(3, len(result["stdout"].splitlines()))
        self.assertEqual(expected_stderr, result["stderr"])
        self.assertEqual(6, len(result["stderr"].splitlines()))
        for record in (result["warmup_records"] +
                       result["measured_records"]):
            self.assertEqual(ROUND_KEYS, set(record))
            self.assertNotIn("stdout", record)

        self.assertTrue(os.path.exists(output))
        self.assertEqual(result, self.read_json(output))

        # Exactly one tool process; exact eight-position argv and explicit env.
        records = self.invocations(executable)
        self.assertEqual(1, len(records))
        self.assertEqual(
            [images_path, "3", "1", "28", "28", "10", "2", "3"],
            records[0]["argv"])
        self.assertEqual(
            {"COURSE_MODEL_DIR": model_dir,
             "COURSE_INFER_BATCH": "2"},
            records[0]["course_env"])
        self.assertNotIn(labels_path, records[0]["argv"])
        self.assertNotIn("--profile", records[0]["argv"])
        self.assertNotIn("--labels", records[0]["argv"])

    # A canonical but label-mismatched vector is still a valid measurement; the
    # accuracy gate is reported separately and rejects the result.
    def test_accuracy_mismatch_is_measured_but_not_accepted(self):
        labels = [1, 0, 3]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "mismatch",
            self.success_config(
                [1, 0, 2], 2, 3, 3, 2, "normalized",
                warm_walls=(1000, 2000), measure_walls=(10, 20, 40)))
        output = self.output_path()
        result = self.benchmark(
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output, count=3, batch=2, warmups=2, runs=3)

        self.assertEqual("pass", result["status"])
        self.assertEqual([], result["errors"])
        self.assertEqual(2, result["summary"]["correct"])
        self.assertEqual(2.0 / 3.0, result["summary"]["accuracy"])
        self.assertFalse(result["summary"]["accuracy_pass"])
        self.assertTrue(result["summary"]["mean_below_100ms"])
        self.assertTrue(result["summary"]["median_below_100ms"])
        self.assertEqual([10.0, 20.0, 40.0],
                         [record["wall_ms"]
                          for record in result["measured_records"]])
        # The CLI returns one for a valid measurement whose accuracy gate fails;
        # the JSON still retains every measured time.
        official = self.make_official_infer()
        self.assertEqual(1, warm.main([
            executable, official, images_path, labels_path,
            model_dir, self.output_path("cli.json"),
            "--count", "3", "--batch", "2", "--warmups", "2", "--runs", "3",
            "--model-source", "course_train"]))
        cli_result = self.read_json(self.output_path("cli.json"))
        self.assertEqual("pass", cli_result["status"])
        self.assertFalse(cli_result["summary"]["accuracy_pass"])
        self.assertEqual([10.0, 20.0, 40.0],
                         [record["wall_ms"]
                          for record in cli_result["measured_records"]])

    # Warmups are excluded and every measured outlier is retained; the <100ms
    # gate requires both statistics while range stays visible.
    def test_slow_outlier_retained_and_warmups_excluded(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "slow",
            self.success_config(
                labels, 2, 3, 3, 2, "normalized",
                warm_walls=(1000, 2000), measure_walls=(10, 20, 300)))
        output = self.output_path()

        status = warm.main([
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output,
            "--count", "3", "--batch", "2", "--warmups", "2", "--runs", "3",
            "--model-source", "course_train"])

        self.assertEqual(1, status)
        result = self.read_json(output)
        self.assertEqual("pass", result["status"])
        self.assertTrue(result["summary"]["accuracy_pass"])
        self.assertEqual(110.0, result["summary"]["mean_ms"])
        self.assertEqual(20.0, result["summary"]["median_ms"])
        self.assertEqual(10.0, result["summary"]["min_ms"])
        self.assertEqual(300.0, result["summary"]["max_ms"])
        self.assertFalse(result["summary"]["mean_below_100ms"])
        self.assertTrue(result["summary"]["median_below_100ms"])
        self.assertEqual([10.0, 20.0, 300.0],
                         [record["wall_ms"]
                          for record in result["measured_records"]])
        self.assertEqual([1000.0, 2000.0],
                         [record["wall_ms"]
                          for record in result["warmup_records"]])

    # A successful CLI run returns zero and publishes the exact JSON schema.
    def test_cli_success_returns_zero(self):
        labels = [1, 0]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model(input_mode="unit")
        executable = self.make_mock(
            "cli-ok",
            self.success_config(
                labels, 1, 1, 2, 2, "unit",
                warm_walls=(5,), measure_walls=(6,)))
        output = self.output_path()
        status = warm.main([
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output,
            "--count", "2", "--batch", "2", "--warmups", "1", "--runs", "1",
            "--model-source", "course_train"])
        self.assertEqual(0, status)
        result = self.read_json(output)
        self.assertEqual("pass", result["status"])
        self.assertEqual(6.0, result["summary"]["mean_ms"])
        self.assertEqual(6.0, result["summary"]["median_ms"])

    # Non-canonical or out-of-range stdout is a collection failure even when the
    # process exits zero; raw stdout is retained exclusively.
    def test_invalid_prediction_stdout_fails_collection(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        official = self.make_official_infer()
        for name, stdout in (
                ("out_of_range", "1\n0\n10\n"),
                ("noncanonical", "01\n0\n2\n")):
            with self.subTest(name=name):
                executable = self.make_mock(
                    name,
                    {
                        "stdout": stdout,
                        "stderr": protocol_text(
                            2, 3, 3, 2, "normalized",
                            warm_walls=(1000, 2000),
                            measure_walls=(10, 20, 40)),
                        "exit_code": 0,
                    })
                output = self.output_path(name + ".json")
                result = self.benchmark(
                    executable, official, images_path, labels_path, model_dir,
                    output, count=3, batch=2, warmups=2, runs=3)
                self.assert_collection_failure(result, output)
                self.assertEqual(stdout, result["stdout"])
                self.assertEqual(0, result["returncode"])
                self.assertTrue(result["hashes_stable"])

    # Extra protocol records are rejected, and the raw partial/full process
    # output plus exit status are retained in the exclusive failure JSON.
    def test_extra_round_record_fails_collection(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        lines = stderr.splitlines(True)
        stderr = "".join(lines[:5] + [lines[4]] + lines[5:])
        executable = self.make_mock(
            "extra",
            {"stdout": prediction_text(labels), "stderr": stderr,
             "exit_code": 0})
        output = self.output_path()
        result = self.benchmark(
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output, count=3, batch=2, warmups=2, runs=3)

        self.assert_collection_failure(result, output)
        self.assertEqual(stderr, result["stderr"])
        self.assertEqual(prediction_text(labels), result["stdout"])
        self.assertEqual(0, result["returncode"])

    # A closed zero-exit process without the completion record is incomplete and
    # must not become a successful series.
    def test_missing_completion_is_partial_failure(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        stderr = protocol_text(
            2, 3, 3, 2, "normalized",
            warm_walls=(1000, 2000), measure_walls=(10, 20, 40))
        stderr = "".join(stderr.splitlines(True)[:-1])
        executable = self.make_mock(
            "partial",
            {"stdout": prediction_text(labels), "stderr": stderr,
             "exit_code": 0})
        output = self.output_path()
        result = self.benchmark(
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output, count=3, batch=2, warmups=2, runs=3)

        self.assert_collection_failure(result, output)
        self.assertEqual(stderr, result["stderr"])
        self.assertEqual(0, result["returncode"])

    # A nonzero tool exit is a failure and the CLI returns one while preserving
    # the raw stdout, stderr, exit code, and actual process wall.
    def test_process_failure_preserves_raw_output_and_exit(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        stderr = (
            iteration_line("warmup", 1, 3, 1000) +
            "error: injected tool failure\n")
        executable = self.make_mock(
            "failure",
            {"stdout": prediction_text(labels), "stderr": stderr,
             "exit_code": 3})
        output = self.output_path()
        status = warm.main([
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output,
            "--count", "3", "--batch", "2", "--warmups", "2", "--runs", "3",
            "--model-source", "course_train"])

        self.assertEqual(1, status)
        result = self.read_json(output)
        self.assert_collection_failure(result, output)
        self.assertEqual(3, result["returncode"])
        self.assertEqual(prediction_text(labels), result["stdout"])
        self.assertEqual(stderr, result["stderr"])
        self.assertIsInstance(result["process_wall_ms"], float)
        self.assertGreater(result["process_wall_ms"], 0.0)
        self.assertEqual(1, len(self.invocations(executable)))

    # The real timeout branch kills a harmless sleeping mock, retains its partial
    # output, writes failure JSON, and never reports accuracy/timing success.
    def test_timeout_preserves_partial_output_and_fails(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        stderr = iteration_line("warmup", 1, 3, 1000)
        executable = self.make_mock(
            "timeout",
            {"stdout": "", "stderr": stderr, "close_streams": True,
             "sleep_seconds": 3.0, "exit_code": 0})
        output = self.output_path()

        with mock.patch.object(warm, "PROCESS_TIMEOUT_SECONDS", 1.0):
            result = self.benchmark(
                executable, self.make_official_infer(), images_path,
                labels_path, model_dir, output,
                count=3, batch=2, warmups=2, runs=3)

        self.assert_collection_failure(result, output)
        self.assertTrue(any("timed out" in error.lower()
                            for error in result["errors"]))
        self.assertIn("event=course_warm_iteration", result["stderr"])
        self.assertTrue(result["returncode"] is None or
                        isinstance(result["returncode"], int))
        self.assertGreater(result["process_wall_ms"], 0.0)
        self.assertEqual(1, len(self.invocations(executable)))

    # Hashes are bound before and after execution; any mutation is a failure even
    # when the protocol itself is complete.
    def test_stale_mutated_hash_fails_collection(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model()
        executable = self.make_mock(
            "mutate",
            {
                "stdout": prediction_text(labels),
                "stderr": protocol_text(
                    2, 3, 3, 2, "normalized",
                    warm_walls=(1000, 2000), measure_walls=(10, 20, 40)),
                "exit_code": 0,
                "mutate": {"path": images_path, "mode": "a", "data": "x"},
            })
        output = self.output_path()
        status = warm.main([
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output,
            "--count", "3", "--batch", "2", "--warmups", "2", "--runs", "3",
            "--model-source", "course_train"])

        self.assertEqual(1, status)
        result = self.read_json(output)
        self.assert_collection_failure(result, output)
        self.assertFalse(result["hashes_stable"])
        self.assertNotEqual(result["sha256"]["images"],
                            result["sha256_after"]["images"])
        self.assertEqual(result["sha256"]["model"],
                         result["sha256_after"]["model"])
        self.assertTrue(any("sha256" in error.lower()
                            for error in result["errors"]))
        self.assertEqual(prediction_text(labels), result["stdout"])
        self.assertEqual(0, result["returncode"])

    # The persisted model mode is authoritative; a conflicting completion mode
    # is rejected rather than normalized away.
    def test_completion_mode_mismatch_fails_collection(self):
        labels = [1, 0, 2]
        images_path, labels_path = self.make_inputs(labels)
        model_dir = self.make_model(input_mode="normalized")
        executable = self.make_mock(
            "mode",
            {
                "stdout": prediction_text(labels),
                "stderr": protocol_text(
                    2, 3, 3, 2, "byte",
                    warm_walls=(1000, 2000), measure_walls=(10, 20, 40)),
                "exit_code": 0,
            })
        output = self.output_path()
        result = self.benchmark(
            executable, self.make_official_infer(), images_path, labels_path,
            model_dir, output, count=3, batch=2, warmups=2, runs=3)

        self.assert_collection_failure(result, output)
        self.assertEqual("normalized", result["input_mode"])
        self.assertEqual(0, result["returncode"])


class WarmPreflightTest(WarmBenchmarkTest):
    # Existing output is refused before the tool is launched and is not replaced.
    def test_existing_output_refused_before_launch(self):
        images_path, labels_path = self.make_inputs([1, 0, 2])
        model_dir = self.make_model()
        executable = self.make_mock(
            "existing",
            self.success_config(
                [1, 0, 2], 1, 1, 3, 2, "normalized",
                warm_walls=(5,), measure_walls=(6,)))
        output = self.output_path()
        with open(output, "w") as output_file:
            output_file.write("sentinel\n")

        with self.assertRaises(FileExistsError):
            self.benchmark(
                executable, self.make_official_infer(), images_path,
                labels_path, model_dir, output,
                count=3, batch=2, warmups=1, runs=1)

        self.assertEqual([], self.invocations(executable))
        with open(output, "r") as output_file:
            self.assertEqual("sentinel\n", output_file.read())

    # Output aliases of any protected input are also refused before launch.
    def test_output_aliases_refused_before_launch(self):
        images_path, labels_path = self.make_inputs([1, 0, 2])
        model_dir = self.make_model()
        official = self.make_official_infer()
        executable = self.make_mock(
            "alias",
            self.success_config(
                [1, 0, 2], 1, 1, 3, 2, "normalized",
                warm_walls=(5,), measure_walls=(6,)))
        aliases = [
            images_path,
            labels_path,
            os.path.join(model_dir, "course.meta"),
            os.path.join(model_dir, "lenet.bin"),
            official,
            executable,
        ]
        for alias in aliases:
            with self.subTest(alias=alias):
                original = file_digest(alias)
                with self.assertRaises((FileExistsError, OSError)):
                    self.benchmark(
                        executable, official, images_path, labels_path,
                        model_dir, alias,
                        count=3, batch=2, warmups=1, runs=1)
                self.assertEqual(original, file_digest(alias))
        self.assertEqual([], self.invocations(executable))

    # Bad input sizes and labels are rejected before launch with no result JSON.
    def test_bad_input_size_or_labels_refused_before_launch(self):
        model_dir = self.make_model()
        official = self.make_official_infer()
        executable = self.make_mock(
            "inputs",
            self.success_config(
                [1, 0, 2], 1, 1, 3, 2, "normalized",
                warm_walls=(5,), measure_walls=(6,)))

        short_images = self.path("short-images.f32")
        with open(short_images, "wb") as images_file:
            images_file.write(b"\x00" * 8)
        _, labels_path = self.make_inputs([1, 0, 2])

        with self.subTest(name="short_images"):
            output = self.output_path("short.json")
            with self.assertRaises(ValueError):
                self.benchmark(
                    executable, official, short_images, labels_path, model_dir,
                    output, count=3, batch=2, warmups=1, runs=1)
            self.assertFalse(os.path.lexists(output))

        images_path, _ = self.make_inputs([1, 0, 2])
        for name, labels in (
                ("short_labels", [1, 0]),
                ("bad_label_value", [1, 0, 10]),
                ("negative_label", [1, 0, -1])):
            with self.subTest(name=name):
                wrong_labels = self.path(name + ".i32")
                with open(wrong_labels, "wb") as labels_file:
                    labels_file.write(struct.pack(
                        "<%di" % len(labels), *labels))
                output = self.output_path(name + ".json")
                with self.assertRaises(ValueError):
                    self.benchmark(
                        executable, official, images_path, wrong_labels,
                        model_dir, output, count=3, batch=2, warmups=1,
                        runs=1)
                self.assertFalse(os.path.lexists(output))

        self.assertEqual([], self.invocations(executable))

    # Missing inputs and a bad persisted mode fail preflight without launching.
    def test_missing_inputs_and_bad_mode_refused_before_launch(self):
        images_path, labels_path = self.make_inputs([1, 0, 2])
        model_dir = self.make_model()
        official = self.make_official_infer()
        executable = self.make_mock(
            "preflight",
            self.success_config(
                [1, 0, 2], 1, 1, 3, 2, "normalized",
                warm_walls=(5,), measure_walls=(6,)))

        with self.subTest(name="missing_images"):
            output = self.output_path("missing-images.json")
            with self.assertRaises((ValueError, OSError)):
                self.benchmark(
                    executable, official, self.path("absent.f32"),
                    labels_path, model_dir, output,
                    count=3, batch=2, warmups=1, runs=1)
            self.assertFalse(os.path.lexists(output))

        with self.subTest(name="missing_labels"):
            output = self.output_path("missing-labels.json")
            with self.assertRaises((ValueError, OSError)):
                self.benchmark(
                    executable, official, images_path,
                    self.path("absent.i32"), model_dir, output,
                    count=3, batch=2, warmups=1, runs=1)
            self.assertFalse(os.path.lexists(output))

        bad_model = self.make_model(name="bad-model", input_mode="float")
        with self.subTest(name="bad_mode"):
            output = self.output_path("bad-mode.json")
            with self.assertRaises(ValueError):
                self.benchmark(
                    executable, official, images_path, labels_path, bad_model,
                    output, count=3, batch=2, warmups=1, runs=1)
            self.assertFalse(os.path.lexists(output))

        self.assertEqual([], self.invocations(executable))

    # Invalid counts/source are rejected before launch and create no result.
    def test_bad_arguments_refused_before_launch(self):
        images_path, labels_path = self.make_inputs([1, 0, 2])
        model_dir = self.make_model()
        official = self.make_official_infer()
        executable = self.make_mock(
            "arguments",
            self.success_config(
                [1, 0, 2], 1, 1, 3, 2, "normalized",
                warm_walls=(5,), measure_walls=(6,)))
        overrides = [
            {"count": 0},
            {"count": -1},
            {"count": True},
            {"batch": 0},
            {"warmups": 0},
            {"runs": 0},
            {"model_source": "legacy-cold"},
        ]
        for index, override in enumerate(overrides):
            with self.subTest(override=override):
                arguments = {
                    "count": 3, "batch": 2, "warmups": 1, "runs": 1,
                    "model_source": "course_train"}
                arguments.update(override)
                output = self.output_path("arguments-{0}.json".format(index))
                with self.assertRaises(ValueError):
                    self.benchmark(
                        executable, official, images_path, labels_path,
                        model_dir, output, **arguments)
                self.assertFalse(os.path.lexists(output))
        self.assertEqual([], self.invocations(executable))

    # CLI preflight errors return one, print a diagnostic, and never launch.
    def test_cli_preflight_error_returns_one_without_launch(self):
        images_path, labels_path = self.make_inputs([1, 0, 2])
        model_dir = self.make_model()
        official = self.make_official_infer()
        executable = self.make_mock(
            "cli-preflight",
            self.success_config(
                [1, 0, 2], 1, 1, 3, 2, "normalized",
                warm_walls=(5,), measure_walls=(6,)))
        output = self.output_path()
        with open(output, "w") as output_file:
            output_file.write("sentinel\n")

        errors = io.StringIO()
        with redirect_stderr(errors):
            status = warm.main([
                executable, official, images_path, labels_path, model_dir,
                output, "--count", "3", "--batch", "2", "--warmups", "1",
                "--runs", "1", "--model-source", "course_train"])

        self.assertEqual(1, status)
        self.assertIn("exists", errors.getvalue().lower())
        self.assertEqual([], self.invocations(executable))


if __name__ == "__main__":
    unittest.main()
