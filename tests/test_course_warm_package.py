from __future__ import print_function

import contextlib
import hashlib
import io
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib

from scripts import course_package, course_warm_benchmark

# Named missing module: this import is the intended tests-first RED until the
# warm packager exists. Main owns that Linux RED before the GREEN resume.
from scripts import course_warm_package

try:
    from unittest import mock
except ImportError:  # pragma: no cover - Python 3.6 includes unittest.mock
    import mock

try:
    import PIL
    from PIL import Image
    PILLOW_AVAILABLE = True
    PILLOW_VERSION = PIL.__version__
except ImportError:
    Image = None
    PILLOW_AVAILABLE = False
    PILLOW_VERSION = "not importable"


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PIXELS = 28 * 28
WARM_SCHEMA = "course-warm-benchmark-v1"
TIMING_SCOPE = "warm_full_request_h2d_prepare_forward_argmax_d2h_validation"
EXPECTED_EVALUATION = {
    "mode": "image",
    "image_origin": "archive",
    "result_image": "result.png",
}
EXPECTED_EVALUATION_TEXT = (
    '{"mode":"image","image_origin":"archive","result_image":"result.png"}\n')
EXPECTED_WHITELIST = tuple(course_package.WHITELIST_FILES)
GENERATED_MEMBERS = ("evaluation.json", "report", "result.png")
EXPECTED_MEMBERS = tuple(sorted(EXPECTED_WHITELIST + GENERATED_MEMBERS))
HASH_FIELDS = ("binary", "official_infer", "model", "meta", "images", "labels")
SUMMARY_FIELDS = (
    "mean_ms", "median_ms", "min_ms", "max_ms", "correct", "accuracy",
    "accuracy_pass", "mean_below_100ms", "median_below_100ms")
TOP_LEVEL_FIELDS = (
    "schema", "timing_scope", "executable", "official_infer", "images_path",
    "labels_path", "model_dir", "model_source", "n", "batch", "input_mode",
    "warmups", "runs", "sha256", "sha256_after", "hashes_stable",
    "returncode", "process_wall_ms", "stdout", "stderr", "warmup_records",
    "measured_records", "predictions_identical_all", "status", "errors",
    "summary")

# Synthetic arithmetic only. These values are never an H20/GPU performance
# claim and are not the actual Task 2 measurement.
DEFAULT_WALLS = [10.0 + (index % 10) * 0.5 for index in range(101)]
DEFAULT_WARMUP_WALLS = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
OUTLIER_WALLS = [20.0] * 100 + [121.0]
MEAN_SLOW_WALLS = [20.0] * 51 + [200.0] * 50
MEDIAN_BOUNDARY_WALLS = [20.0] * 50 + [100.0] * 51

# Rendering tests are visible skips when the development-only Pillow install is
# absent; no test or helper installs packages automatically.
requires_pillow = unittest.skipUnless(
    PILLOW_AVAILABLE,
    "development-only Pillow renderer unavailable (Pillow {0}); "
    "warm packaging is not exercised".format(PILLOW_VERSION))


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def prediction_lines(predictions, final_newline=True):
    text = "\n".join(str(int(value)) for value in predictions)
    if final_newline:
        text += "\n"
    return text


def wall_text(value):
    return repr(float(value))


def round_records(values, samples):
    return [{
        "iteration": index + 1,
        "samples": samples,
        "wall_ms": float(value),
        "predictions_identical": True,
    } for index, value in enumerate(values)]


def warm_stderr(warmups, runs, samples, batch, input_mode):
    lines = []
    for index, value in enumerate(warmups):
        lines.append(
            "event=course_warm_iteration phase=warmup iteration={0} "
            "samples={1} wall_ms={2} predictions_identical=1".format(
                index + 1, samples, wall_text(value)))
    for index, value in enumerate(runs):
        lines.append(
            "event=course_warm_iteration phase=measure iteration={0} "
            "samples={1} wall_ms={2} predictions_identical=1".format(
                index + 1, samples, wall_text(value)))
    lines.append(
        "event=course_warm_complete status=pass warmups={0} runs={1} "
        "samples={2} batch={3} input_mode={4}".format(
            len(warmups), len(runs), samples, batch, input_mode))
    return "\n".join(lines) + "\n"


def record_image_text(drawn, boxes):
    """Patch the real ImageDraw.text, recording strings and pixel bounds."""
    from PIL import ImageDraw

    original = ImageDraw.ImageDraw.text

    def record(draw, xy, text, *args, **kwargs):
        drawn.append(text)
        bound_options = {}
        for key in ("font", "anchor", "stroke_width", "spacing", "align",
                    "direction", "features", "language"):
            if key in kwargs:
                bound_options[key] = kwargs[key]
        if hasattr(draw, "textbbox"):
            boxes.append(draw.textbbox(xy, text, **bound_options))
        return original(draw, xy, text, *args, **kwargs)

    return mock.patch.object(ImageDraw.ImageDraw, "text", record)


class WarmPackageFixture(unittest.TestCase):
    # Every test owns its repository tree, model, binary, inputs, warm
    # benchmark, training log, and output path. All durations are synthetic
    # arithmetic inputs, never actual measurements.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = self.temporary.name
        self.root = os.path.join(self.directory, "repository")
        self.fixture_count = 0
        self._create_repository()

    def tearDown(self):
        self.temporary.cleanup()

    def _create_repository(self):
        for relative in EXPECTED_WHITELIST:
            destination = os.path.join(self.root, *relative.split("/"))
            parent = os.path.dirname(destination)
            if not os.path.isdir(parent):
                os.makedirs(parent)
            with open(destination, "w", encoding="utf-8", newline="\n") as output:
                output.write("synthetic source fixture: {0}\n".format(relative))

    def _make_fixture(self, correct=9642, walls=None, warmup_walls=None,
                      warmups=5, runs=101, n=10000, batch=1024, epochs=20,
                      status="pass", model_source="course_train",
                      input_mode="unit", training_mode=None,
                      completion_crc=None, write_completion=True,
                      duplicate_completion=False, process_wall_ms=9004.5):
        self.fixture_count += 1
        base = os.path.join(self.directory, "fixture{0}".format(
            self.fixture_count))
        model_dir = os.path.join(base, "model")
        data_dir = os.path.join(base, "data")
        bin_dir = os.path.join(base, "bin")
        output_dir = os.path.join(base, "output")
        for directory in (model_dir, data_dir, bin_dir, output_dir):
            os.makedirs(directory)

        checkpoint = b"synthetic course warm checkpoint bytes; CRC only\n"
        checkpoint_path = os.path.join(model_dir, "lenet.bin")
        with open(checkpoint_path, "wb") as output:
            output.write(checkpoint)
        checkpoint_crc = zlib.crc32(checkpoint) & 0xFFFFFFFF
        crc_text = completion_crc or "{0:08x}".format(checkpoint_crc)
        meta_text = (
            "course_lenet_v1\n"
            "input_mode={0}\n"
            "shape=1,28,28,10\n"
            "checkpoint_crc32={1}\n".format(
                input_mode, "{0:08x}".format(checkpoint_crc)))
        meta_path = os.path.join(model_dir, "course.meta")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as output:
            output.write(meta_text)

        binary_path = os.path.join(bin_dir, "course_warm_benchmark")
        with open(binary_path, "wb") as output:
            output.write(b"synthetic current warm benchmark binary\n")
        infer_path = os.path.join(bin_dir, "infer")
        with open(infer_path, "wb") as output:
            output.write(b"synthetic current official infer binary\n")

        images_path = os.path.join(data_dir, "images.f32")
        with open(images_path, "wb") as output:
            output.truncate(n * IMAGE_PIXELS * 4)
        labels = [index % 10 for index in range(n)]
        labels_path = os.path.join(data_dir, "labels.i32")
        with open(labels_path, "wb") as output:
            output.write(struct.pack("<{0}i".format(n), *labels))

        predictions = list(labels)
        for index in range(correct, n):
            predictions[index] = (predictions[index] + 1) % 10
        stdout = prediction_lines(predictions)

        if walls is None:
            walls = list(DEFAULT_WALLS)
        walls = list(walls)
        if warmup_walls is None:
            warmup_walls = list(DEFAULT_WARMUP_WALLS)
        warmup_walls = list(warmup_walls)
        stderr = warm_stderr(warmup_walls, walls, n, batch, input_mode)

        accuracy = float(correct) / float(n)
        mean_ms = statistics.mean(walls)
        median_ms = statistics.median(walls)
        summary = {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min(walls),
            "max_ms": max(walls),
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_pass": accuracy >= 0.96,
            "mean_below_100ms": mean_ms < 100.0,
            "median_below_100ms": median_ms < 100.0,
        }
        sha256 = {
            "binary": file_digest(binary_path),
            "official_infer": file_digest(infer_path),
            "model": file_digest(checkpoint_path),
            "meta": file_digest(meta_path),
            "images": file_digest(images_path),
            "labels": file_digest(labels_path),
        }
        benchmark = {
            "schema": WARM_SCHEMA,
            "timing_scope": TIMING_SCOPE,
            "executable": binary_path,
            "official_infer": infer_path,
            "images_path": images_path,
            "labels_path": labels_path,
            "model_dir": model_dir,
            "model_source": model_source,
            "n": n,
            "batch": batch,
            "input_mode": input_mode,
            "warmups": warmups,
            "runs": runs,
            "sha256": sha256,
            "sha256_after": dict(sha256),
            "hashes_stable": True,
            "returncode": 0,
            "process_wall_ms": process_wall_ms,
            "stdout": stdout,
            "stderr": stderr,
            "warmup_records": round_records(warmup_walls, n),
            "measured_records": round_records(walls, n),
            "predictions_identical_all": True,
            "status": "pass",
            "errors": [],
            "summary": summary,
        }
        benchmark_path = os.path.join(base, "benchmark.json")
        write_json(benchmark_path, benchmark)

        log_lines = []
        for epoch in range(1, epochs + 1):
            log_lines.append(
                "event=course_train_epoch epoch={0} weighted_loss=0.5 "
                "val_accuracy=0.97 elapsed_ms=1000.000".format(epoch))
        completion_mode = training_mode or input_mode
        if write_completion:
            completion = (
                "event=course_train status={0} epochs={1} best_epoch={1} "
                "validation_accuracy=0.97 input_mode={2} "
                "checkpoint_crc32={3}".format(
                    status, epochs, completion_mode, crc_text))
            log_lines.append(completion)
            if duplicate_completion:
                log_lines.append(completion)
        training_log = os.path.join(base, "training.log")
        with open(training_log, "w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(log_lines) + "\n")

        return {
            "base": base,
            "root": self.root,
            "model_dir": model_dir,
            "model_path": checkpoint_path,
            "meta_path": meta_path,
            "binary": binary_path,
            "official_infer": infer_path,
            "images": images_path,
            "labels": labels_path,
            "benchmark_json": benchmark_path,
            "training_log": training_log,
            "output_zip": os.path.join(
                output_dir, "course-warm-submission-{0}.zip".format(
                    self.fixture_count)),
            "predictions": predictions,
            "correct": correct,
            "n": n,
            "walls": walls,
            "warmup_walls": warmup_walls,
            "sha256": dict(sha256),
            "benchmark": benchmark,
        }

    def _package(self, fixture):
        return course_warm_package.run_warm_package(
            fixture["benchmark_json"], fixture["training_log"],
            fixture["output_zip"], root=fixture["root"])

    def _rewrite_benchmark(self, fixture, value):
        write_json(fixture["benchmark_json"], value)
        fixture["benchmark"] = value

    def _mutate_benchmark(self, fixture, mutate):
        benchmark = read_json(fixture["benchmark_json"])
        mutate(benchmark)
        self._rewrite_benchmark(fixture, benchmark)

    def _assert_no_artifacts(self, fixture):
        for name in GENERATED_MEMBERS:
            path = os.path.join(fixture["root"], name)
            self.assertFalse(
                os.path.lexists(path),
                "unexpected root artifact was created: {0}".format(path))
        self.assertFalse(
            os.path.lexists(fixture["output_zip"]),
            "unexpected archive was created: {0}".format(
                fixture["output_zip"]))

    def _assert_rejected(self, fixture, expected_text=None):
        with self.assertRaises(course_package.PackageError) as context:
            self._package(fixture)
        if expected_text is not None:
            self.assertIn(expected_text.lower(), str(context.exception).lower())
        self._assert_no_artifacts(fixture)
        return context.exception

    def _report(self, fixture):
        with open(os.path.join(fixture["root"], "report"), "r",
                  encoding="utf-8") as input_file:
            return input_file.read()


class WarmPackageApiTest(WarmPackageFixture):
    def test_constants_and_shared_helper_identities(self):
        self.assertEqual(
            course_warm_package.WARM_BENCHMARK_SCHEMA, WARM_SCHEMA)
        self.assertEqual(
            course_warm_package.WARM_BENCHMARK_SCHEMA,
            course_warm_benchmark.WARM_BENCHMARK_SCHEMA)
        self.assertEqual(
            course_warm_package.TIMING_SCOPE,
            course_warm_benchmark.TIMING_SCOPE)
        self.assertEqual(course_warm_package.REQUIRED_N, 10000)
        self.assertEqual(course_warm_package.MIN_RUNS, 101)
        self.assertEqual(course_warm_package.MIN_WARMUPS, 1)
        self.assertEqual(course_warm_package.REQUIRED_EPOCHS, 20)
        self.assertEqual(course_warm_package.ACCURACY_THRESHOLD, 0.96)
        self.assertEqual(course_warm_package.WARM_LIMIT_MS, 100.0)
        self.assertEqual(tuple(course_warm_package.HASH_FIELDS), HASH_FIELDS)
        # The warm helper reuses the reviewed collector parser and the cold
        # publication/error helpers; it must not fork them.
        self.assertIs(
            course_warm_package.parse_tool_output,
            course_warm_benchmark.parse_tool_output)
        self.assertIs(
            course_warm_package.PackageError, course_package.PackageError)
        self.assertEqual(
            tuple(course_warm_package.WHITELIST_FILES), EXPECTED_WHITELIST)
        # The historical cold schema/API stays untouched.
        self.assertEqual(course_package.BENCHMARK_SCHEMA,
                         "course-benchmark-v1")
        self.assertEqual(course_package.REQUIRED_RUNS, 5)
        self.assertEqual(len(TOP_LEVEL_FIELDS), 26)
        self.assertEqual(len(set(TOP_LEVEL_FIELDS)), 26)
        self.assertEqual(len(SUMMARY_FIELDS), 9)
        for name in ("run_warm_package", "metric_lines", "render_result_png",
                     "_repository_root"):
            self.assertTrue(callable(getattr(course_warm_package, name)))

    def test_metric_lines_are_one_accuracy_group_and_one_warm_time_group(self):
        lines = course_warm_package.metric_lines(0.9642, 21.0, 20.0)
        self.assertEqual(list(lines), [
            "Test Accuracy: 96.42%",
            "Test Time: mean 21.00 ms; median 20.00 ms",
        ])
        self.assertEqual(
            sum(1 for line in lines if "Test Accuracy:" in line), 1)
        self.assertEqual(sum(1 for line in lines if "Test Time:" in line), 1)
        slow_lines = course_warm_package.metric_lines(0.971, 150.0, 90.0)
        self.assertEqual(
            slow_lines[1], "Test Time: mean 150.00 ms; median 90.00 ms")
        self.assertEqual(
            sum(1 for line in slow_lines if "Test Time:" in line), 1)

    def test_default_root_is_repository_from_script_location(self):
        self.assertEqual(
            os.path.abspath(course_warm_package._repository_root()),
            os.path.abspath(ROOT))
        self.assertEqual(
            os.path.abspath(course_warm_package._repository_root()),
            os.path.dirname(os.path.dirname(
                os.path.abspath(course_warm_package.__file__))))


class WarmPackageSuccessTest(WarmPackageFixture):
    @requires_pillow
    def test_successful_warm_package_has_37_members_and_exact_evaluation(self):
        fixture = self._make_fixture()
        result = self._package(fixture)

        self.assertEqual(
            set(result), {
                "members", "correct", "accuracy", "mean_ms", "median_ms",
                "min_ms", "max_ms", "performance_target"})
        self.assertEqual(result["performance_target"], "achieved")
        self.assertEqual(result["correct"], 9642)
        self.assertAlmostEqual(result["accuracy"], 0.9642)
        self.assertAlmostEqual(
            result["mean_ms"], statistics.mean(fixture["walls"]))
        self.assertAlmostEqual(
            result["median_ms"], statistics.median(fixture["walls"]))
        self.assertAlmostEqual(result["min_ms"], min(fixture["walls"]))
        self.assertAlmostEqual(result["max_ms"], max(fixture["walls"]))
        self.assertEqual(tuple(sorted(result["members"])), EXPECTED_MEMBERS)
        self.assertEqual(len(result["members"]), 37)

        with zipfile.ZipFile(fixture["output_zip"], "r") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(tuple(sorted(archive.namelist())), EXPECTED_MEMBERS)
            self.assertEqual(len(archive.namelist()), 37)
            unpacked = 0
            for info in archive.infolist():
                name = info.filename
                self.assertFalse(name.startswith("/"))
                self.assertNotIn("\\", name)
                self.assertNotIn("..", name.split("/"))
                self.assertFalse(name.endswith("/"))
                unpacked += info.file_size
            self.assertLess(unpacked, 300000000)
            self.assertLess(os.path.getsize(fixture["output_zip"]), 100000000)
            self.assertEqual(
                archive.read("evaluation.json").decode("utf-8"),
                EXPECTED_EVALUATION_TEXT)
            for relative in EXPECTED_WHITELIST + GENERATED_MEMBERS:
                with open(os.path.join(fixture["root"], relative),
                          "rb") as input_file:
                    self.assertEqual(archive.read(relative), input_file.read())

        for name in GENERATED_MEMBERS:
            self.assertTrue(
                os.path.isfile(os.path.join(fixture["root"], name)))
        self.assertEqual(
            read_json(os.path.join(fixture["root"], "evaluation.json")),
            EXPECTED_EVALUATION)

    @requires_pillow
    def test_summary_uses_measured_rounds_and_excludes_warmups(self):
        fixture = self._make_fixture(
            warmup_walls=[5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
            walls=[20.0] * 101)
        result = self._package(fixture)
        self.assertEqual(result["performance_target"], "achieved")
        self.assertEqual(result["mean_ms"], 20.0)
        self.assertEqual(result["median_ms"], 20.0)
        self.assertEqual(result["min_ms"], 20.0)
        self.assertEqual(result["max_ms"], 20.0)
        report = self._report(fixture)
        self.assertIn("Warmups: 5", report)
        self.assertIn("Warmup 1: wall_ms=5000.000 ms", report)
        self.assertIn("Warmup 5: wall_ms=9000.000 ms", report)
        self.assertIn("Mean measured: 20.000 ms", report)
        self.assertIn("Median measured: 20.000 ms", report)

    @requires_pillow
    def test_report_lists_all_101_measured_times_in_order(self):
        fixture = self._make_fixture()
        self._package(fixture)
        report = self._report(fixture)
        self.assertEqual(report.count("Measured "), 101)
        self.assertEqual(report.count("Warmup "), 5)
        positions = []
        for index, wall_ms in enumerate(fixture["walls"], start=1):
            line = "Measured {0}: wall_ms={1:.3f} ms".format(index, wall_ms)
            self.assertIn(line, report)
            positions.append(report.index(line))
        self.assertEqual(positions, sorted(positions))
        for index, wall_ms in enumerate(fixture["warmup_walls"], start=1):
            self.assertIn(
                "Warmup {0}: wall_ms={1:.3f} ms".format(index, wall_ms),
                report)
        self.assertIn(
            "Mean measured: {0:.3f} ms".format(
                statistics.mean(fixture["walls"])), report)
        self.assertIn(
            "Median measured: {0:.3f} ms".format(
                statistics.median(fixture["walls"])), report)
        self.assertIn(
            "Min measured: {0:.3f} ms".format(min(fixture["walls"])), report)
        self.assertIn(
            "Max measured: {0:.3f} ms".format(max(fixture["walls"])), report)

    @requires_pillow
    def test_report_metric_groups_and_warm_scope(self):
        fixture = self._make_fixture()
        self._package(fixture)
        report = self._report(fixture)
        lowered = report.lower()
        self.assertIn("Test Accuracy: 96.42% (9642/10000)", report)
        self.assertIn(
            "Test Time: mean {0:.2f} ms; median {1:.2f} ms".format(
                statistics.mean(fixture["walls"]),
                statistics.median(fixture["walls"])), report)
        for token in ("warm full-request", "h2d", "input preparation",
                      "forward", "argmax", "d2h", "synchron", "validation"):
            self.assertIn(token, lowered)
        self.assertIn("warmups: 5", lowered)
        self.assertIn("runs=101", lowered)
        self.assertIn("n=10000", lowered)
        self.assertIn("batch=1024", lowered)
        self.assertIn("mean and median <100 ms", lowered)
        self.assertNotIn("startup through exit", lowered)
        self.assertNotIn("median of 5 full-process runs", lowered)

    @requires_pillow
    def test_report_discloses_excluded_stages_and_cold_scope_unchanged(self):
        fixture = self._make_fixture()
        self._package(fixture)
        report = self._report(fixture)
        lowered = report.lower()
        for token in ("excluded from the measured interval",
                      "model/image disk loading", "cuda initialization",
                      "allocation", "weight upload", "stdout formatting"):
            self.assertIn(token, lowered)
        self.assertIn("cold-process", lowered)
        self.assertIn("unchanged", lowered)
        self.assertNotIn("startup through exit", lowered)
        self.assertNotIn("median of 5 full-process runs", lowered)

    @requires_pillow
    def test_report_training_identities_preprocessing_and_leaderboard(self):
        fixture = self._make_fixture()
        self._package(fixture)
        report = self._report(fixture)
        self.assertIn(
            "Training completion: event=course_train status=pass epochs=20",
            report)
        self.assertIn("Training epochs: 20", report)
        self.assertIn("Model source: course_train", report)
        self.assertIn("Preprocessing assumption:", report)
        self.assertIn(
            "official course preprocessing remains unconfirmed",
            report.lower())
        self.assertIn("Leaderboard timing: unresolved", report)
        with open(fixture["model_path"], "rb") as input_file:
            checkpoint = input_file.read()
        self.assertIn(
            "checkpoint_crc32={0:08x}".format(
                zlib.crc32(checkpoint) & 0xFFFFFFFF), report)
        for key in ("binary_sha256", "official_infer_sha256", "model_sha256",
                    "meta_sha256", "images_sha256", "labels_sha256"):
            self.assertIn(key + "=", report)
        self.assertIn(
            "binary_sha256={0}".format(fixture["sha256"]["binary"]), report)
        self.assertIn(
            "official_infer_sha256={0}".format(
                fixture["sha256"]["official_infer"]), report)
        self.assertIn(
            "model_sha256={0}".format(fixture["sha256"]["model"]), report)
        self.assertIn(
            "meta_sha256={0}".format(fixture["sha256"]["meta"]), report)
        self.assertIn(
            "images_sha256={0}".format(fixture["sha256"]["images"]), report)
        self.assertIn(
            "labels_sha256={0}".format(fixture["sha256"]["labels"]), report)

    @requires_pillow
    def test_report_contains_no_personal_or_fixture_absolute_paths(self):
        fixture = self._make_fixture()
        self._package(fixture)
        report = self._report(fixture)
        for absolute_path in (fixture["base"], fixture["root"],
                              fixture["model_dir"], fixture["images"],
                              fixture["labels"], fixture["binary"],
                              fixture["official_infer"],
                              fixture["benchmark_json"]):
            self.assertNotIn(absolute_path, report)
        self.assertNotIn("C:\\Users\\", report)
        self.assertNotIn("/home/", report)

    @requires_pillow
    def test_png_is_within_size_and_pixel_caps(self):
        fixture = self._make_fixture()
        self._package(fixture)
        path = os.path.join(fixture["root"], "result.png")
        self.assertTrue(os.path.isfile(path))
        self.assertLessEqual(os.path.getsize(path), 4000000)
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertIn(image.mode, ("RGB", "RGBA"))
            self.assertLessEqual(image.width * image.height, 16000000)
            image.verify()
        with Image.open(path) as image:
            image.load()

    @requires_pillow
    def test_png_has_single_warm_metric_group_range_and_scope(self):
        fixture = self._make_fixture()
        drawn = []
        boxes = []
        with record_image_text(drawn, boxes):
            self._package(fixture)
        joined = "\n".join(drawn)
        self.assertEqual(joined.count("Test Accuracy:"), 1)
        self.assertEqual(joined.count("Test Time:"), 1)
        self.assertIn("Test Accuracy: 96.42%", joined)
        self.assertIn(
            "Test Time: mean {0:.2f} ms; median {1:.2f} ms".format(
                statistics.mean(fixture["walls"]),
                statistics.median(fixture["walls"])), joined)
        self.assertIn(
            "Warm full-request range: {0:.2f} - {1:.2f} ms".format(
                min(fixture["walls"]), max(fixture["walls"])), joined)
        self.assertIn(
            "5 warmups; 101 measured rounds; 10000 images per round", joined)
        for token in ("H2D", "input preparation", "forward", "argmax", "D2H",
                      "synchronization", "host validation"):
            self.assertIn(token, joined)
        self.assertIn("ACHIEVED", joined)
        self.assertNotIn("FAILED", joined)
        self.assertNotIn("startup through exit", joined)
        self.assertNotIn("median of 5 full-process runs", joined)
        self.assertTrue(boxes)
        with Image.open(os.path.join(fixture["root"], "result.png")) as image:
            for box in boxes:
                self.assertGreaterEqual(box[0], 0)
                self.assertGreaterEqual(box[1], 0)
                self.assertLessEqual(box[2], image.width)
                self.assertLessEqual(box[3], image.height)

    @requires_pillow
    def test_slow_warm_image_reports_range_and_failed_verdict(self):
        fixture = self._make_fixture(walls=MEAN_SLOW_WALLS)
        drawn = []
        boxes = []
        with record_image_text(drawn, boxes):
            result = self._package(fixture)
        self.assertEqual(result["performance_target"], "failed")
        joined = "\n".join(drawn)
        self.assertEqual(joined.count("Test Accuracy:"), 1)
        self.assertEqual(joined.count("Test Time:"), 1)
        self.assertIn(
            "Warm full-request range: 20.00 - 200.00 ms", joined)
        self.assertIn("FAILED", joined)
        self.assertNotIn("ACHIEVED", joined)
        self.assertTrue(boxes)
        with Image.open(os.path.join(fixture["root"], "result.png")) as image:
            for box in boxes:
                self.assertLessEqual(box[2], image.width)
                self.assertLessEqual(box[3], image.height)

    @requires_pillow
    def test_outlier_above_100_with_mean_and_median_below_is_accepted(self):
        fixture = self._make_fixture(walls=OUTLIER_WALLS)
        result = self._package(fixture)
        self.assertEqual(result["performance_target"], "achieved")
        self.assertAlmostEqual(result["mean_ms"], 21.0)
        self.assertAlmostEqual(result["median_ms"], 20.0)
        self.assertAlmostEqual(result["min_ms"], 20.0)
        self.assertAlmostEqual(result["max_ms"], 121.0)
        report = self._report(fixture)
        self.assertIn("Test Time: mean 21.00 ms; median 20.00 ms", report)
        self.assertIn("Mean measured: 21.000 ms", report)
        self.assertIn("Median measured: 20.000 ms", report)
        self.assertIn("Max measured: 121.000 ms", report)
        self.assertIn("Measured 101: wall_ms=121.000 ms", report)
        self.assertIn("Performance target (warm full-request mean and "
                      "median <100 ms): ACHIEVED", report)

    @requires_pillow
    def test_mean_at_or_above_100_with_median_below_is_packaged_failed(self):
        fixture = self._make_fixture(walls=MEAN_SLOW_WALLS)
        result = self._package(fixture)
        mean_ms = statistics.mean(MEAN_SLOW_WALLS)
        median_ms = statistics.median(MEAN_SLOW_WALLS)
        self.assertGreaterEqual(mean_ms, 100.0)
        self.assertLess(median_ms, 100.0)
        self.assertAlmostEqual(result["mean_ms"], mean_ms)
        self.assertAlmostEqual(result["median_ms"], median_ms)
        self.assertEqual(result["performance_target"], "failed")
        report = self._report(fixture)
        self.assertIn("Accuracy target (>=96%): PASS", report)
        self.assertIn(
            "Performance target (warm full-request mean and median <100 ms): "
            "FAILED", report)
        self.assertIn("Measured 101: wall_ms=200.000 ms", report)

    @requires_pillow
    def test_median_exactly_100_with_mean_below_is_failed(self):
        fixture = self._make_fixture(walls=MEDIAN_BOUNDARY_WALLS)
        result = self._package(fixture)
        self.assertLess(statistics.mean(MEDIAN_BOUNDARY_WALLS), 100.0)
        self.assertEqual(statistics.median(MEDIAN_BOUNDARY_WALLS), 100.0)
        self.assertEqual(result["performance_target"], "failed")
        report = self._report(fixture)
        self.assertIn("Median measured: 100.000 ms", report)
        self.assertIn(
            "Performance target (warm full-request mean and median <100 ms): "
            "FAILED", report)

    @requires_pillow
    def test_one_warmup_and_102_runs_are_accepted_as_minimums(self):
        fixture = self._make_fixture(
            warmups=1, warmup_walls=[500.0], runs=102,
            walls=list(DEFAULT_WALLS) + [15.0])
        result = self._package(fixture)
        self.assertEqual(result["performance_target"], "achieved")
        report = self._report(fixture)
        self.assertEqual(report.count("Measured "), 102)
        self.assertIn("Warmup 1: wall_ms=500.000 ms", report)
        self.assertIn("runs=102", report)

    @requires_pillow
    def test_accuracy_exactly_9600_passes(self):
        fixture = self._make_fixture(correct=9600)
        result = self._package(fixture)
        self.assertEqual(result["correct"], 9600)
        self.assertEqual(result["accuracy"], 0.96)
        self.assertIn("Test Accuracy: 96.00% (9600/10000)",
                      self._report(fixture))

    @requires_pillow
    def test_outlier_cannot_be_discarded_from_recomputed_summary(self):
        fixture = self._make_fixture(walls=OUTLIER_WALLS)
        self._mutate_benchmark(fixture, lambda benchmark: benchmark[
            "summary"].update(
                mean_ms=20.0, median_ms=20.0, min_ms=20.0, max_ms=20.0))
        self._assert_rejected(fixture, "summary")

    @requires_pillow
    def test_summary_including_warmups_is_rejected(self):
        fixture = self._make_fixture(
            warmup_walls=[5000.0, 6000.0, 7000.0, 8000.0, 9000.0],
            walls=[20.0] * 101)
        combined_mean = statistics.mean(
            fixture["warmup_walls"] + fixture["walls"])
        self._mutate_benchmark(fixture, lambda benchmark: benchmark[
            "summary"].update(mean_ms=combined_mean))
        self._assert_rejected(fixture, "summary")


class WarmPackageRejectionTest(WarmPackageFixture):
    def test_cold_schema_is_rejected(self):
        fixture = self._make_fixture()
        self._mutate_benchmark(
            fixture, lambda benchmark: benchmark.update(
                schema="course-benchmark-v1"))
        self._assert_rejected(fixture, "schema")

    def test_cold_five_process_series_is_not_normalized_as_warm(self):
        fixture = self._make_fixture()
        self._mutate_benchmark(fixture, lambda benchmark: benchmark.update(
            runs=[{"wall_ms": 100.0, "returncode": 0}] * 5))
        self._assert_rejected(fixture)

    def test_timing_scope_and_model_source_are_pinned(self):
        for name, mutate in (
                ("timing_scope", lambda benchmark: benchmark.update(
                    timing_scope="full_process_startup_through_exit")),
                ("model_source", lambda benchmark: benchmark.update(
                    model_source="legacy"))):
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_required_n_runs_and_warmup_counts_are_enforced(self):
        cases = (
            ("small_n", lambda benchmark: benchmark.update(n=9999)),
            ("short_runs", lambda benchmark: benchmark.update(runs=100)),
            ("zero_warmups", lambda benchmark: benchmark.update(warmups=0)),
            ("bool_n", lambda benchmark: benchmark.update(n=True)),
            ("bool_runs", lambda benchmark: benchmark.update(runs=True)),
            ("bool_warmups", lambda benchmark: benchmark.update(warmups=True)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_partial_or_extra_record_sets_are_rejected(self):
        cases = (
            ("missing_measured", lambda benchmark: benchmark[
                "measured_records"].pop()),
            ("extra_measured", lambda benchmark: benchmark[
                "measured_records"].append(dict(
                    benchmark["measured_records"][-1]))),
            ("missing_warmup", lambda benchmark: benchmark[
                "warmup_records"].pop()),
            ("extra_warmup", lambda benchmark: benchmark[
                "warmup_records"].append(dict(
                    benchmark["warmup_records"][-1]))),
            ("empty_warmups", lambda benchmark: benchmark.update(
                warmup_records=[])),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_stored_records_must_equal_parsed_raw_records(self):
        cases = (
            ("warmup_wall", lambda benchmark: benchmark["warmup_records"][0]
                .update(wall_ms=1.25)),
            ("measured_wall", lambda benchmark: benchmark[
                "measured_records"][7].update(wall_ms=1.25)),
            ("iteration", lambda benchmark: benchmark["measured_records"][2]
                .update(iteration=99)),
            ("samples", lambda benchmark: benchmark["measured_records"][3]
                .update(samples=9999)),
            ("flag", lambda benchmark: benchmark["measured_records"][4]
                .update(predictions_identical=False)),
            ("extra_key", lambda benchmark: benchmark["measured_records"][5]
                .update(extra="value")),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_raw_protocol_must_be_exact(self):
        def replace_stderr(fixture, mutate):
            benchmark = read_json(fixture["benchmark_json"])
            benchmark["stderr"] = mutate(benchmark["stderr"])
            self._rewrite_benchmark(fixture, benchmark)

        mutations = (
            ("missing_completion", lambda text: "\n".join(
                text.splitlines()[:-1]) + "\n"),
            ("extra_record", lambda text: text.replace(
                "event=course_warm_complete",
                "event=course_warm_iteration phase=measure iteration=102 "
                "samples=10000 wall_ms=10.0 predictions_identical=1\n"
                "event=course_warm_complete")),
            ("false_invariance", lambda text: text.replace(
                "predictions_identical=1", "predictions_identical=0", 1)),
            ("wrong_completion_runs", lambda text: text.replace(
                "runs=101", "runs=102")),
            ("wrong_completion_mode", lambda text: text.replace(
                "input_mode=unit", "input_mode=byte")),
            ("failed_completion", lambda text: text.replace(
                "status=pass", "status=fail")),
            ("blank_line", lambda text: text.replace(
                "event=course_warm_complete", "\nevent=course_warm_complete")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                replace_stderr(fixture, mutate)
                self._assert_rejected(fixture)

    def test_status_errors_and_returncode_are_required(self):
        cases = (
            ("status", lambda benchmark: benchmark.update(status="fail")),
            ("errors", lambda benchmark: benchmark.update(
                errors=["partial collection"])),
            ("returncode", lambda benchmark: benchmark.update(returncode=1)),
            ("null_returncode", lambda benchmark: benchmark.update(
                returncode=None)),
            ("bool_returncode", lambda benchmark: benchmark.update(
                returncode=False)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_incomplete_or_duplicate_training_is_rejected(self):
        cases = (
            ("missing", dict(write_completion=False)),
            ("status", dict(status="fail")),
            ("epochs", dict(epochs=19)),
            ("duplicate", dict(duplicate_completion=True)),
        )
        for name, options in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture(**options)
                self._assert_rejected(fixture)

    def test_training_crc_and_mode_bindings_are_rejected(self):
        checkpoint = b"synthetic course warm checkpoint bytes; CRC only\n"
        wrong_crc = "{0:08x}".format(
            (zlib.crc32(checkpoint) & 0xFFFFFFFF) ^ 0xFFFFFFFF)
        cases = (
            ("crc", dict(completion_crc=wrong_crc), None),
            ("training_mode", dict(training_mode="byte"), None),
            ("benchmark_mode", {}, lambda benchmark: benchmark.update(
                input_mode="byte")),
            ("unsupported_mode", {}, lambda benchmark: benchmark.update(
                input_mode="rgb")),
        )
        for name, options, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture(**options)
                if mutate is not None:
                    self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_stale_identity_files_are_rejected(self):
        cases = ("binary", "official_infer", "model", "meta", "images",
                 "labels")
        for case in cases:
            with self.subTest(case=case):
                fixture = self._make_fixture()
                if case == "binary":
                    with open(fixture["binary"], "wb") as output:
                        output.write(b"rebuilt warm benchmark\n")
                elif case == "official_infer":
                    with open(fixture["official_infer"], "wb") as output:
                        output.write(b"rebuilt official infer\n")
                elif case == "model":
                    with open(fixture["model_path"], "wb") as output:
                        output.write(b"replaced checkpoint bytes\n")
                elif case == "meta":
                    with open(fixture["meta_path"], "w", encoding="utf-8",
                              newline="\n") as output:
                        output.write(
                            "course_lenet_v2\ninput_mode=unit\n"
                            "shape=1,28,28,10\ncheckpoint_crc32=00000000\n")
                elif case == "images":
                    with open(fixture["images"], "r+b") as output:
                        output.seek(0)
                        output.write(b"\x01\x00\x00\x00")
                else:
                    with open(fixture["labels"], "r+b") as output:
                        output.seek(0)
                        output.write(struct.pack("<i", 9))
                self._assert_rejected(fixture)

    def test_before_and_after_hashes_must_be_current_stable_and_complete(self):
        cases = (
            ("before_mismatch", lambda benchmark: benchmark[
                "sha256"].update(images="0" * 64)),
            ("after_mismatch", lambda benchmark: benchmark[
                "sha256_after"].update(model="0" * 64)),
            ("unstable", lambda benchmark: benchmark.update(
                hashes_stable=False)),
            ("unstable_string", lambda benchmark: benchmark.update(
                hashes_stable="true")),
            ("missing_field", lambda benchmark: benchmark[
                "sha256"].pop("official_infer")),
            ("extra_field", lambda benchmark: benchmark[
                "sha256_after"].update(extra="0" * 64)),
            ("bad_hex", lambda benchmark: benchmark[
                "sha256"].update(binary="A" * 64)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_predictions_are_recounted_against_int32_labels(self):
        cases = (
            ("forged_correct", lambda benchmark: benchmark["summary"].update(
                correct=10000)),
            ("short_stdout", lambda benchmark: benchmark.update(
                stdout="\n".join(benchmark["stdout"].splitlines()[:-1])
                + "\n")),
            ("noncanonical", lambda benchmark: benchmark.update(
                stdout="01\n" + "\n".join(
                    benchmark["stdout"].splitlines()[1:]) + "\n")),
            ("out_of_range", lambda benchmark: benchmark.update(
                stdout="10\n" + "\n".join(
                    benchmark["stdout"].splitlines()[1:]) + "\n")),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_prediction_invariance_evidence_is_required(self):
        cases = (
            ("global", lambda benchmark: benchmark.update(
                predictions_identical_all=False)),
            ("record", lambda benchmark: benchmark["measured_records"][9]
                .update(predictions_identical=False)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_forged_summary_fields_are_rejected(self):
        mutations = (
            ("mean", lambda summary: summary.update(mean_ms=1.0)),
            ("median", lambda summary: summary.update(median_ms=1.0)),
            ("min", lambda summary: summary.update(min_ms=1.0)),
            ("max", lambda summary: summary.update(max_ms=1.0)),
            ("correct", lambda summary: summary.update(correct=10000)),
            ("accuracy", lambda summary: summary.update(accuracy=1.0)),
            ("accuracy_pass", lambda summary: summary.update(
                accuracy_pass=False)),
            ("mean_gate", lambda summary: summary.update(
                mean_below_100ms=False)),
            ("median_gate", lambda summary: summary.update(
                median_below_100ms=False)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(
                    fixture, lambda benchmark, mutate=mutate: mutate(
                        benchmark["summary"]))
                self._assert_rejected(fixture)

    def test_accuracy_9599_is_rejected(self):
        fixture = self._make_fixture(correct=9599)
        self._assert_rejected(fixture)

    def test_missing_or_wrong_sized_inputs_are_rejected(self):
        cases = ("missing_images", "short_images", "short_labels",
                 "missing_labels", "bad_label")
        for case in cases:
            with self.subTest(case=case):
                fixture = self._make_fixture()
                if case == "missing_images":
                    os.remove(fixture["images"])
                elif case == "short_images":
                    with open(fixture["images"], "r+b") as output:
                        output.truncate(fixture["n"] * IMAGE_PIXELS * 4 - 4)
                elif case == "short_labels":
                    with open(fixture["labels"], "r+b") as output:
                        output.truncate(fixture["n"] * 4 - 4)
                elif case == "missing_labels":
                    os.remove(fixture["labels"])
                else:
                    with open(fixture["labels"], "r+b") as output:
                        output.seek(0)
                        output.write(struct.pack("<i", 10))
                self._assert_rejected(fixture)

    def test_missing_tool_or_official_infer_is_rejected(self):
        for key in ("binary", "official_infer"):
            with self.subTest(key=key):
                fixture = self._make_fixture()
                os.remove(fixture[key])
                self._assert_rejected(fixture)

    def test_extra_or_missing_top_level_keys_are_rejected(self):
        cases = (
            ("extra", lambda benchmark: benchmark.update(extra="value")),
            ("missing", lambda benchmark: benchmark.pop("timing_scope")),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)

    def test_nonpositive_process_wall_and_bad_batch_are_rejected(self):
        cases = (
            ("process_wall", lambda benchmark: benchmark.update(
                process_wall_ms=0.0)),
            ("batch", lambda benchmark: benchmark.update(batch=0)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                self._mutate_benchmark(fixture, mutate)
                self._assert_rejected(fixture)


class WarmPackagePublicationTest(WarmPackageFixture):
    def test_existing_archive_is_never_overwritten(self):
        fixture = self._make_fixture()
        prior = b"prior owner archive bytes"
        with open(fixture["output_zip"], "wb") as output:
            output.write(prior)
        with self.assertRaises(FileExistsError):
            self._package(fixture)
        with open(fixture["output_zip"], "rb") as input_file:
            self.assertEqual(input_file.read(), prior)
        for name in GENERATED_MEMBERS:
            self.assertFalse(
                os.path.lexists(os.path.join(fixture["root"], name)))

    def test_preexisting_root_artifacts_are_rejected(self):
        for name in GENERATED_MEMBERS:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                path = os.path.join(fixture["root"], name)
                with open(path, "wb") as output:
                    output.write(b"preexisting owner data\n")
                with self.assertRaises(FileExistsError):
                    self._package(fixture)
                with open(path, "rb") as input_file:
                    self.assertEqual(
                        input_file.read(), b"preexisting owner data\n")
                self.assertFalse(os.path.lexists(fixture["output_zip"]))
                for other in GENERATED_MEMBERS:
                    if other != name:
                        self.assertFalse(os.path.lexists(
                            os.path.join(fixture["root"], other)))
                os.remove(path)

    @requires_pillow
    def test_publication_rollback_removes_partial_root_artifacts(self):
        fixture = self._make_fixture()
        real = course_package._write_exclusive
        calls = []

        def failing(path, content):
            if len(calls) == 2:
                raise OSError("synthetic publication failure")
            calls.append(path)
            return real(path, content)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                course_package, "_write_exclusive", failing))
            if hasattr(course_warm_package, "_write_exclusive"):
                stack.enter_context(mock.patch.object(
                    course_warm_package, "_write_exclusive", failing))
            with self.assertRaises(OSError):
                self._package(fixture)
        self._assert_no_artifacts(fixture)

    @requires_pillow
    def test_packaging_leaves_benchmark_training_and_inputs_unchanged(self):
        fixture = self._make_fixture()
        protected = {
            "benchmark": file_digest(fixture["benchmark_json"]),
            "training": file_digest(fixture["training_log"]),
            "binary": file_digest(fixture["binary"]),
            "official_infer": file_digest(fixture["official_infer"]),
            "model": file_digest(fixture["model_path"]),
            "meta": file_digest(fixture["meta_path"]),
            "images": file_digest(fixture["images"]),
            "labels": file_digest(fixture["labels"]),
        }
        self._package(fixture)
        self.assertEqual(file_digest(fixture["benchmark_json"]),
                         protected["benchmark"])
        self.assertEqual(file_digest(fixture["training_log"]),
                         protected["training"])
        self.assertEqual(file_digest(fixture["binary"]),
                         protected["binary"])
        self.assertEqual(file_digest(fixture["official_infer"]),
                         protected["official_infer"])
        self.assertEqual(file_digest(fixture["model_path"]),
                         protected["model"])
        self.assertEqual(file_digest(fixture["meta_path"]),
                         protected["meta"])
        self.assertEqual(file_digest(fixture["images"]),
                         protected["images"])
        self.assertEqual(file_digest(fixture["labels"]),
                         protected["labels"])


class WarmPackageDirectCliTest(WarmPackageFixture):
    def test_direct_script_bad_warm_json_is_a_clean_diagnostic(self):
        # A real direct-script CLI run in an owned scratch root, with the three
        # helper scripts copied in and PYTHONPATH removed. Direct execution
        # gives __main__ and the warm module's fallback sibling import separate
        # course_package modules, so this test covers the public error-class
        # boundary that in-process/help tests cannot see.
        scripts_dir = os.path.join(self.root, "scripts")
        os.makedirs(scripts_dir)
        for name in ("course_package.py", "course_warm_package.py",
                     "course_warm_benchmark.py"):
            shutil.copyfile(
                os.path.join(ROOT, "scripts", name),
                os.path.join(scripts_dir, name))

        benchmark_json = os.path.join(self.root, "bad-warm.json")
        write_json(benchmark_json, {"schema": WARM_SCHEMA})
        training_log = os.path.join(self.root, "training.log")
        log_lines = [
            "event=course_train_epoch epoch={0} weighted_loss=0.5 "
            "val_accuracy=0.97 elapsed_ms=1000.000".format(epoch)
            for epoch in range(1, 21)
        ]
        log_lines.append(
            "event=course_train status=pass epochs=20 best_epoch=20 "
            "validation_accuracy=0.97 input_mode=unit "
            "checkpoint_crc32=00000000")
        with open(training_log, "w", encoding="utf-8",
                  newline="\n") as output:
            output.write("\n".join(log_lines) + "\n")
        output_zip = os.path.join(self.root, "scratch-output.zip")

        environment = {
            name: value for name, value in os.environ.items()
            if name != "PYTHONPATH"
        }
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-S", os.path.join(
                scripts_dir, "course_package.py"),
             "--benchmark", benchmark_json,
             "--training-log", training_log,
             "--output", output_zip],
            cwd=self.root, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        stderr = result.stderr.decode("utf-8", "replace")
        self.assertIn("course_package:", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertFalse(os.path.lexists(output_zip))
        for name in GENERATED_MEMBERS:
            self.assertFalse(
                os.path.lexists(os.path.join(self.root, name)))


class WarmPackageDispatchTest(WarmPackageFixture):
    def test_warm_schema_dispatches_to_run_warm_package(self):
        fixture = self._make_fixture()
        expected = {"members": []}
        with mock.patch.object(
                course_warm_package, "run_warm_package",
                return_value=expected) as runner:
            result = course_package.run_package(
                fixture["benchmark_json"], fixture["training_log"],
                fixture["output_zip"], root=fixture["root"])
        self.assertEqual(result, expected)
        runner.assert_called_once_with(
            fixture["benchmark_json"], fixture["training_log"],
            fixture["output_zip"], root=fixture["root"])

    def test_cold_schema_never_uses_the_warm_runner(self):
        fixture = self._make_fixture()
        write_json(fixture["benchmark_json"],
                   {"schema": "course-benchmark-v1"})
        with mock.patch.object(
                course_package, "_validate_benchmark",
                side_effect=course_package.PackageError(
                    "cold validation reached")) as cold, mock.patch.object(
                    course_warm_package, "run_warm_package") as warm:
            with self.assertRaises(course_package.PackageError) as context:
                course_package.run_package(
                    fixture["benchmark_json"], fixture["training_log"],
                    fixture["output_zip"], root=fixture["root"])
        self.assertIn("cold validation reached", str(context.exception))
        cold.assert_called_once()
        warm.assert_not_called()

    def test_preexisting_output_refusal_precedes_dispatch_and_validation(self):
        fixture = self._make_fixture()
        os.remove(fixture["benchmark_json"])
        with open(os.path.join(fixture["root"], "evaluation.json"),
                  "wb") as output:
            output.write(b"preexisting owner data\n")
        with mock.patch.object(
                course_warm_package, "run_warm_package") as warm:
            with self.assertRaises(FileExistsError):
                course_package.run_package(
                    fixture["benchmark_json"], fixture["training_log"],
                    fixture["output_zip"], root=fixture["root"])
        warm.assert_not_called()

    @requires_pillow
    def test_cli_packages_warm_success_with_zero_status(self):
        fixture = self._make_fixture()
        with mock.patch.object(
                course_package, "_repository_root",
                return_value=fixture["root"]):
            status = course_package.main([
                "--benchmark", fixture["benchmark_json"],
                "--training-log", fixture["training_log"],
                "--output", fixture["output_zip"],
            ])
        self.assertEqual(status, 0)
        for name in GENERATED_MEMBERS:
            self.assertTrue(
                os.path.isfile(os.path.join(fixture["root"], name)))
        self.assertTrue(os.path.isfile(fixture["output_zip"]))

    @requires_pillow
    def test_cli_packages_truthful_warm_failure_with_zero_status(self):
        fixture = self._make_fixture(walls=MEAN_SLOW_WALLS)
        with mock.patch.object(
                course_package, "_repository_root",
                return_value=fixture["root"]):
            status = course_package.main([
                "--benchmark", fixture["benchmark_json"],
                "--training-log", fixture["training_log"],
                "--output", fixture["output_zip"],
            ])
        self.assertEqual(status, 0)
        report = self._report(fixture)
        self.assertIn("Accuracy target (>=96%): PASS", report)
        self.assertIn(
            "Performance target (warm full-request mean and median <100 ms): "
            "FAILED", report)

    def test_cli_maps_warm_validation_failure_to_one(self):
        fixture = self._make_fixture()
        self._mutate_benchmark(fixture, lambda benchmark: benchmark[
            "summary"].update(mean_ms=1.0))
        stderr = io.StringIO()
        with mock.patch.object(
                course_package, "_repository_root",
                return_value=fixture["root"]), mock.patch.object(
                    sys, "stderr", stderr):
            status = course_package.main([
                "--benchmark", fixture["benchmark_json"],
                "--training-log", fixture["training_log"],
                "--output", fixture["output_zip"],
            ])
        self.assertEqual(status, 1)
        self.assertIn("summary", stderr.getvalue().lower())
        self._assert_no_artifacts(fixture)


if __name__ == "__main__":
    unittest.main()
