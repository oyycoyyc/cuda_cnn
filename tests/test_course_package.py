from __future__ import print_function

import hashlib
import io
import json
import os
import statistics
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib

from scripts import course_package

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
BENCHMARK_SCHEMA = "course-benchmark-v1"
EXPECTED_EVALUATION = {
    "mode": "image",
    "image_origin": "archive",
    "result_image": "result.png",
}
EXPECTED_EVALUATION_TEXT = (
    '{"mode":"image","image_origin":"archive","result_image":"result.png"}\n')

# The explicit submission whitelist, kept sorted. It is only the actual default
# CMake runtime source graph: no tests, tools, Python helpers, data, build,
# diagnostics, evidence, Dockerfile, credentials, or symlinks.
EXPECTED_WHITELIST = (
    "CMakeLists.txt",
    "course/gpu_ops.cu",
    "course/gpu_ops.h",
    "course/inference.cu",
    "course/inference.h",
    "course/interface.h",
    "course/io.cpp",
    "course/model_io.cpp",
    "course/model_io.h",
    "course/runtime.cu",
    "course/runtime.h",
    "course/training.cu",
    "course/training.h",
    "include/checkpoint.h",
    "include/cuda_check.h",
    "include/layers.h",
    "include/lenet.h",
    "include/parameters.h",
    "include/random.h",
    "include/tensor.h",
    "main.cu",
    "src/binary_io.h",
    "src/checkpoint.cpp",
    "src/kernels/activation.cu",
    "src/kernels/adam.cu",
    "src/kernels/convolution.cu",
    "src/kernels/input.cu",
    "src/kernels/linear.cu",
    "src/kernels/loss.cu",
    "src/kernels/metrics.cu",
    "src/kernels/pooling.cu",
    "src/lenet.cu",
    "src/parameters.cpp",
    "src/random.cpp",
)
GENERATED_MEMBERS = ("evaluation.json", "report", "result.png")
EXPECTED_MEMBERS = tuple(sorted(EXPECTED_WHITELIST + GENERATED_MEMBERS))
FAST_WALLS = [88.1, 90.2, 85.3, 87.4, 91.5]
SLOW_WALLS = [148.0, 150.0, 151.0, 149.0, 152.0]

# Rendering tests are visible skips when the development-only Pillow install is
# absent; the availability is printed by unittest's skip reason. No test or
# helper may install packages automatically.
requires_pillow = unittest.skipUnless(
    PILLOW_AVAILABLE,
    "development-only Pillow renderer unavailable (Pillow {0}); "
    "packaging is not exercised".format(PILLOW_VERSION))


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


class CoursePackageTest(unittest.TestCase):
    # Every test owns its repository tree, model, binary, inputs, benchmark,
    # training log, and output path. All numerical values are synthetic test
    # data and are never actual measurements.
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

    def _make_fixture(self, correct=9642, walls=None, epochs=20, status="pass",
                      profiled=False, model_source="course_train", runs=5,
                      n=10000, input_mode="unit", completion_crc=None,
                      write_completion=True, duplicate_completion=False):
        self.fixture_count += 1
        base = os.path.join(self.directory, "fixture{0}".format(
            self.fixture_count))
        model_dir = os.path.join(base, "model")
        data_dir = os.path.join(base, "data")
        bin_dir = os.path.join(base, "bin")
        output_dir = os.path.join(base, "output")
        for directory in (model_dir, data_dir, bin_dir, output_dir):
            os.makedirs(directory)

        checkpoint = b"synthetic course checkpoint bytes; CRC only\n"
        checkpoint_path = os.path.join(model_dir, "lenet.bin")
        with open(checkpoint_path, "wb") as output:
            output.write(checkpoint)
        checkpoint_crc = zlib.crc32(checkpoint) & 0xFFFFFFFF
        crc_text = completion_crc or "{0:08x}".format(checkpoint_crc)
        meta_text = (
            "course_lenet_v1\n"
            "input_mode={0}\n"
            "shape=1,28,28,10\n"
            "checkpoint_crc32={1}\n".format(input_mode, "{0:08x}".format(
                checkpoint_crc)))
        meta_path = os.path.join(model_dir, "course.meta")
        with open(meta_path, "w", encoding="utf-8", newline="\n") as output:
            output.write(meta_text)

        binary_path = os.path.join(bin_dir, "infer")
        with open(binary_path, "wb") as output:
            output.write(b"synthetic current infer binary\n")

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
        stderr = (
            "event=course_infer status=pass samples={0} batches={1} "
            "input_mode={2}\n".format(n, (n + 1023) // 1024, input_mode))

        if walls is None:
            walls = list(FAST_WALLS)
        walls = list(walls)[:runs]
        run_records = []
        for wall_ms in walls:
            run_records.append({
                "wall_ms": wall_ms,
                "returncode": 0,
                "correct": correct,
                "stdout": stdout,
                "stderr": stderr,
            })
        median_ms = statistics.median(walls)
        accuracy = float(correct) / float(n)
        benchmark = {
            "schema": BENCHMARK_SCHEMA,
            "executable": binary_path,
            "images_path": images_path,
            "labels_path": labels_path,
            "model_dir": model_dir,
            "model_source": model_source,
            "n": n,
            "batch": 1024,
            "input_mode": input_mode,
            "profiled": profiled,
            "runs_requested": runs,
            "sha256": {
                "binary": file_digest(binary_path),
                "model": file_digest(checkpoint_path),
                "meta": file_digest(meta_path),
                "images": file_digest(images_path),
                "labels": file_digest(labels_path),
            },
            "runs": run_records,
            "summary": {
                "median_ms": median_ms,
                "max_ms": max(walls),
                "correct": correct,
                "accuracy": accuracy,
                "accuracy_pass": accuracy >= 0.96,
                "all_wall_below_100ms": all(
                    wall < 100.0 for wall in walls),
            },
        }
        benchmark_path = os.path.join(base, "benchmark.json")
        write_json(benchmark_path, benchmark)

        log_lines = []
        for epoch in range(1, epochs + 1):
            log_lines.append(
                "event=course_train_epoch epoch={0} weighted_loss=0.5 "
                "val_accuracy=0.97 elapsed_ms=1000.000".format(epoch))
        if write_completion:
            completion = (
                "event=course_train status={0} epochs={1} best_epoch={1} "
                "validation_accuracy=0.97 input_mode={2} "
                "checkpoint_crc32={3}".format(status, epochs, input_mode,
                                              crc_text))
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
            "images": images_path,
            "labels": labels_path,
            "benchmark_json": benchmark_path,
            "training_log": training_log,
            "output_zip": os.path.join(
                output_dir, "course-submission-{0}.zip".format(
                    self.fixture_count)),
            "predictions": predictions,
            "correct": correct,
            "n": n,
            "walls": walls,
            "benchmark": benchmark,
        }

    def _package(self, fixture):
        return course_package.run_package(
            fixture["benchmark_json"], fixture["training_log"],
            fixture["output_zip"], root=fixture["root"])

    def _rewrite_benchmark(self, fixture, value):
        write_json(fixture["benchmark_json"], value)
        fixture["benchmark"] = value

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

    def test_whitelist_is_exact_cmake_runtime_source_graph(self):
        self.assertEqual(
            tuple(course_package.WHITELIST_FILES), EXPECTED_WHITELIST)
        self.assertEqual(
            len(set(course_package.WHITELIST_FILES)),
            len(course_package.WHITELIST_FILES))
        for name in course_package.WHITELIST_FILES:
            self.assertFalse(name.startswith("/"))
            self.assertNotIn("\\", name)
            self.assertNotIn("..", name.split("/"))
            self.assertNotIn(name.split("/")[0], (
                "data", "build", "weights", "tests", "scripts", "docs",
                "diagnostics", "runtime-output", "course/tests",
                "course/tools"))
        self.assertNotIn("course/tests/gpu_tests.cu",
                         course_package.WHITELIST_FILES)
        self.assertNotIn("course/tools/model_bundle.cpp",
                         course_package.WHITELIST_FILES)

    def test_default_root_is_repository_from_script_location(self):
        self.assertEqual(
            os.path.abspath(course_package._repository_root()),
            os.path.abspath(ROOT))
        self.assertEqual(
            os.path.abspath(course_package._repository_root()),
            os.path.dirname(os.path.dirname(
                os.path.abspath(course_package.__file__))))

    @requires_pillow
    def test_evaluation_json_content_is_exact(self):
        fixture = self._make_fixture()
        self._package(fixture)
        path = os.path.join(fixture["root"], "evaluation.json")
        with open(path, "r", encoding="utf-8") as input_file:
            text = input_file.read()
        self.assertEqual(text, EXPECTED_EVALUATION_TEXT)
        self.assertEqual(json.loads(text), EXPECTED_EVALUATION)
        self.assertEqual(
            set(json.loads(text)), {"mode", "image_origin", "result_image"})
        self.assertEqual(course_package.EVALUATION_CONTENT,
                         EXPECTED_EVALUATION)
        self.assertEqual(course_package.EVALUATION_JSON_TEXT,
                         EXPECTED_EVALUATION_TEXT)

    @requires_pillow
    def test_successful_package_has_exact_members_and_hashes(self):
        fixture = self._make_fixture()
        result = self._package(fixture)

        self.assertEqual(tuple(sorted(result["members"])), EXPECTED_MEMBERS)
        self.assertEqual(result["accuracy"], 0.9642)
        self.assertEqual(result["correct"], 9642)
        self.assertEqual(result["performance_target"], "achieved")
        self.assertAlmostEqual(result["median_ms"], 88.1)
        self.assertAlmostEqual(result["max_ms"], 91.5)

        with zipfile.ZipFile(fixture["output_zip"], "r") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(tuple(sorted(archive.namelist())), EXPECTED_MEMBERS)
            self.assertEqual(len(archive.namelist()), len(EXPECTED_MEMBERS))
            unpacked = 0
            for info in archive.infolist():
                name = info.filename
                self.assertFalse(name.startswith("/"))
                self.assertNotIn("\\", name)
                self.assertNotIn("..", name.split("/"))
                self.assertNotIn(name, ("", ".", ".."))
                self.assertFalse(name.endswith("/"))
                self.assertFalse(name.startswith("./"))
                self.assertNotIn("//", name)
                unpacked += info.file_size
            self.assertLess(unpacked, 300000000)
            self.assertLess(os.path.getsize(fixture["output_zip"]), 100000000)
            self.assertEqual(
                archive.read("evaluation.json").decode("utf-8"),
                EXPECTED_EVALUATION_TEXT)
            for relative in EXPECTED_WHITELIST + GENERATED_MEMBERS:
                with open(os.path.join(fixture["root"], relative),
                          "rb") as input_file:
                    self.assertEqual(archive.read(relative),
                                     input_file.read())

        for name in GENERATED_MEMBERS:
            self.assertTrue(
                os.path.isfile(os.path.join(fixture["root"], name)))

    @requires_pillow
    def test_report_lists_every_run_with_provenance_and_scope(self):
        fixture = self._make_fixture()
        self._package(fixture)
        report_path = os.path.join(fixture["root"], "report")
        with open(report_path, "r", encoding="utf-8") as input_file:
            report = input_file.read()

        self.assertIn("Test Accuracy: 96.42% (9642/10000)", report)
        self.assertIn(
            "Test Time: 88.10 ms (median of 5 full-process runs)", report)
        self.assertIn("Median full-process wall: 88.100 ms", report)
        self.assertIn("Max full-process wall: 91.500 ms", report)
        self.assertIn("n=10000", report)
        self.assertIn("batch=1024", report)
        self.assertIn("runs=5", report)
        self.assertIn("profiled=false", report)
        self.assertIn("Model source: course_train", report)
        self.assertIn("Preprocessing assumption:", report)
        self.assertIn("Leaderboard timing: unresolved", report)
        self.assertIn("Accuracy target (>=96%): PASS", report)
        self.assertIn("Performance target (full-process <100 ms): ACHIEVED",
                      report)
        for index, wall_ms in enumerate(fixture["walls"], start=1):
            self.assertIn(
                "Run {0}: wall_ms={1:.3f} correct=9642".format(index, wall_ms),
                report)
        self.assertIn("max_ms=91.500", report)
        with open(fixture["model_path"], "rb") as input_file:
            checkpoint = input_file.read()
        expected_crc = "{0:08x}".format(
            zlib.crc32(checkpoint) & 0xFFFFFFFF)
        self.assertIn("checkpoint_crc32={0}".format(expected_crc), report)
        for key in ("binary_sha256", "model_sha256", "meta_sha256",
                    "images_sha256", "labels_sha256"):
            self.assertIn(key + "=", report)
        with open(fixture["binary"], "rb") as input_file:
            self.assertIn(
                "binary_sha256={0}".format(
                    hashlib.sha256(input_file.read()).hexdigest()), report)
        self.assertIn(
            "Training completion: event=course_train status=pass epochs=20",
            report)
        self.assertNotIn("epochs=19", report)
        for absolute_path in (fixture["base"], fixture["root"],
                              fixture["model_dir"], fixture["images"],
                              fixture["binary"], fixture["benchmark_json"]):
            self.assertNotIn(absolute_path, report)

    def test_metric_lines_are_one_accuracy_group_and_one_time_group(self):
        lines = course_package.metric_lines(0.9642, 88.1)
        self.assertEqual(list(lines), [
            "Test Accuracy: 96.42%",
            "Test Time: 88.10 ms (median of 5 full-process runs)",
        ])
        self.assertEqual(sum(1 for line in lines if "Test Accuracy:" in line), 1)
        self.assertEqual(sum(1 for line in lines if "Test Time:" in line), 1)
        slow_lines = course_package.metric_lines(0.971, 150.0)
        self.assertEqual(slow_lines[1],
                         "Test Time: 150.00 ms (median of 5 full-process runs)")
        self.assertEqual(
            sum(1 for line in slow_lines if "Test Time:" in line), 1)

    @requires_pillow
    def test_rendered_png_is_within_size_and_pixel_caps(self):
        fixture = self._make_fixture()
        self._package(fixture)
        path = os.path.join(fixture["root"], "result.png")
        self.assertTrue(os.path.isfile(path))
        self.assertLessEqual(os.path.getsize(path), 4000000)
        with Image.open(path) as image:
            self.assertEqual(image.format, "PNG")
            self.assertIn(image.mode, ("RGB", "RGBA"))
            self.assertEqual(image.size,
                             (image.width, image.height))
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)
            self.assertLessEqual(
                image.width * image.height, 16000000)
            image.verify()
        with Image.open(path) as image:
            image.load()
            self.assertLessEqual(
                image.width * image.height, 16000000)

    @requires_pillow
    def test_rendered_image_draws_single_metric_group(self):
        from PIL import ImageDraw

        fixture = self._make_fixture()
        drawn = []
        original = ImageDraw.ImageDraw.text

        def record(draw, xy, text, *args, **kwargs):
            drawn.append(text)
            return original(draw, xy, text, *args, **kwargs)

        with mock.patch.object(ImageDraw.ImageDraw, "text", record):
            self._package(fixture)
        joined = "\n".join(drawn)
        self.assertIn("Test Accuracy: 96.42%", joined)
        self.assertIn(
            "Test Time: 88.10 ms (median of 5 full-process runs)", joined)
        self.assertEqual(joined.count("Test Accuracy:"), 1)
        self.assertEqual(joined.count("Test Time:"), 1)
        self.assertNotIn("Test Time: 91.50", joined)
        self.assertNotIn("Test Time: 85.30", joined)
        for absolute_path in (fixture["base"], fixture["root"],
                              fixture["model_dir"], fixture["images"],
                              fixture["binary"], fixture["benchmark_json"]):
            self.assertNotIn(absolute_path, joined)

    @requires_pillow
    def test_slow_truthful_result_is_accepted_and_labelled_failed(self):
        fixture = self._make_fixture(correct=9670, walls=SLOW_WALLS)
        before = file_digest(fixture["benchmark_json"])
        result = self._package(fixture)
        self.assertEqual(result["performance_target"], "failed")
        self.assertAlmostEqual(result["median_ms"], 150.0)
        self.assertAlmostEqual(result["max_ms"], 152.0)
        self.assertEqual(file_digest(fixture["benchmark_json"]), before)

        with open(os.path.join(fixture["root"], "report"),
                  "r", encoding="utf-8") as input_file:
            report = input_file.read()
        self.assertIn(
            "Test Time: 150.00 ms (median of 5 full-process runs)", report)
        self.assertIn("Performance target (full-process <100 ms): FAILED",
                      report)
        self.assertIn("Accuracy target (>=96%): PASS", report)
        self.assertNotIn("Performance target (full-process <100 ms): ACHIEVED",
                         report)
        self.assertIn("Run 5: wall_ms=152.000 correct=9670", report)
        for wall_ms in SLOW_WALLS:
            self.assertIn("{0:.3f}".format(wall_ms), report)

    @requires_pillow
    def test_accuracy_exactly_9600_passes(self):
        fixture = self._make_fixture(correct=9600)
        result = self._package(fixture)
        self.assertEqual(result["correct"], 9600)
        self.assertEqual(result["accuracy"], 0.96)
        with open(os.path.join(fixture["root"], "report"),
                  "r", encoding="utf-8") as input_file:
            self.assertIn("Test Accuracy: 96.00% (9600/10000)",
                          input_file.read())

    def test_accuracy_9599_is_rejected(self):
        fixture = self._make_fixture(correct=9599)
        self._assert_rejected(fixture)

    def test_incomplete_training_epochs_are_rejected(self):
        for epochs in (1, 19):
            with self.subTest(epochs=epochs):
                fixture = self._make_fixture(epochs=epochs)
                self._assert_rejected(fixture, "epochs")

    def test_missing_training_completion_is_rejected(self):
        fixture = self._make_fixture(write_completion=False)
        self._assert_rejected(fixture)

    def test_failed_training_completion_is_rejected(self):
        fixture = self._make_fixture(status="fail")
        self._assert_rejected(fixture, "status")

    def test_duplicate_training_completion_is_rejected(self):
        fixture = self._make_fixture(duplicate_completion=True)
        self._assert_rejected(fixture)

    def test_training_completion_crc_mismatch_is_rejected(self):
        checkpoint = b"synthetic course checkpoint bytes; CRC only\n"
        actual = zlib.crc32(checkpoint) & 0xFFFFFFFF
        wrong = (actual ^ 0xFFFFFFFF) & 0xFFFFFFFF
        fixture = self._make_fixture(
            completion_crc="{0:08x}".format(wrong))
        self._assert_rejected(fixture, "crc")

    def test_stale_binary_model_or_meta_is_rejected(self):
        cases = ("binary", "model", "meta")
        for case in cases:
            with self.subTest(case=case):
                fixture = self._make_fixture()
                if case == "binary":
                    with open(fixture["binary"], "wb") as output:
                        output.write(b"rebuilt after measurement\n")
                elif case == "model":
                    with open(fixture["model_path"], "wb") as output:
                        output.write(b"replaced model bytes\n")
                else:
                    with open(fixture["model_path"], "rb") as input_file:
                        checkpoint = input_file.read()
                    with open(fixture["meta_path"], "w", encoding="utf-8",
                              newline="\n") as output:
                        output.write(
                            "course_lenet_v1\ninput_mode=byte\n"
                            "shape=1,28,28,10\ncheckpoint_crc32={0:08x}\n".format(
                                zlib.crc32(checkpoint) & 0xFFFFFFFF))
                self._assert_rejected(fixture)

    def test_profiled_benchmark_is_rejected(self):
        fixture = self._make_fixture(profiled=True)
        self._assert_rejected(fixture, "profiled")

    def test_profile_event_in_stderr_is_rejected_when_flag_is_false(self):
        fixture = self._make_fixture()
        benchmark = read_json(fixture["benchmark_json"])
        benchmark["runs"][2]["stderr"] += (
            "event=course_profile_batch batch=1 h2d_ms=0.100 "
            "prepare_ms=0.100 forward_ms=0.100\n"
            "event=course_profile profile_batches=10 gpu_h2d_ms=1.000\n")
        self._rewrite_benchmark(fixture, benchmark)
        self._assert_rejected(fixture, "profile")

    def test_duplicate_or_nonfinite_benchmark_json_is_rejected(self):
        cases = ("duplicate", "nonfinite")
        for case in cases:
            with self.subTest(case=case):
                fixture = self._make_fixture()
                path = fixture["benchmark_json"]
                with open(path, "r", encoding="utf-8") as input_file:
                    text = input_file.read()
                if case == "duplicate":
                    original = '  "schema": "course-benchmark-v1",'
                    self.assertIn(original, text)
                    text = text.replace(
                        original,
                        original + '\n  "schema": "course-benchmark-v1",',
                        1)
                else:
                    original = '  "accuracy": 0.9642,'
                    self.assertIn(original, text)
                    text = text.replace(
                        original, '  "accuracy": NaN,', 1)
                with open(path, "w", encoding="utf-8", newline="\n") as output:
                    output.write(text)
                self._assert_rejected(fixture)

    @requires_pillow
    def test_personal_path_or_credential_in_payload_is_rejected(self):
        payloads = (
            ("path", '\n// C:\\Users\\alice\\private\\notes.txt\n'),
            ("credential",
             "\n-----BEGIN PRIVATE KEY-----\n"
             "c3ludGhldGljIHRlc3QgZGF0YQ==\n"
             "-----END PRIVATE KEY-----\n"),
        )
        for name, payload in payloads:
            with self.subTest(name=name):
                fixture = self._make_fixture()
                target = os.path.join(self.root, "src", "random.cpp")
                with open(target, "a", encoding="utf-8", newline="\n") as output:
                    output.write(payload)
                self._assert_rejected(fixture)

    def test_metadata_crc_is_recomputed_from_checkpoint_bytes(self):
        fixture = self._make_fixture()
        with open(fixture["model_path"], "rb") as input_file:
            checkpoint = input_file.read()
        actual = zlib.crc32(checkpoint) & 0xFFFFFFFF
        expected_crc = "{0:08x}".format(actual)
        wrong = "{0:08x}".format((actual ^ 0xFFFFFFFF) & 0xFFFFFFFF)
        with open(fixture["meta_path"], "w", encoding="utf-8",
                  newline="\n") as output:
            output.write(
                "course_lenet_v1\ninput_mode=unit\nshape=1,28,28,10\n"
                "checkpoint_crc32={0}\n".format(wrong))
        benchmark = read_json(fixture["benchmark_json"])
        benchmark["sha256"]["meta"] = file_digest(fixture["meta_path"])
        self._rewrite_benchmark(fixture, benchmark)
        log_lines = []
        with open(fixture["training_log"], "r", encoding="utf-8") as input_file:
            for line in input_file:
                line = line.rstrip("\n")
                if line.startswith("event=course_train status="):
                    line = line.replace(expected_crc, wrong)
                log_lines.append(line)
        with open(fixture["training_log"], "w", encoding="utf-8",
                  newline="\n") as output:
            output.write("\n".join(log_lines) + "\n")
        self._assert_rejected(fixture, "crc")

    def test_incomplete_run_set_is_rejected(self):
        cases = ("missing_run", "wrong_requested_count")
        for case in cases:
            with self.subTest(case=case):
                fixture = self._make_fixture()
                benchmark = read_json(fixture["benchmark_json"])
                if case == "missing_run":
                    benchmark["runs"] = benchmark["runs"][:4]
                    benchmark["runs_requested"] = 5
                else:
                    benchmark["runs_requested"] = 6
                self._rewrite_benchmark(fixture, benchmark)
                self._assert_rejected(fixture)

    def test_nonzero_run_is_rejected(self):
        fixture = self._make_fixture()
        benchmark = read_json(fixture["benchmark_json"])
        benchmark["runs"][2]["returncode"] = 1
        self._rewrite_benchmark(fixture, benchmark)
        self._assert_rejected(fixture, "returncode")

    def test_run_correct_count_is_recomputed_from_predictions(self):
        fixture = self._make_fixture()
        benchmark = read_json(fixture["benchmark_json"])
        benchmark["runs"][0]["correct"] = 10000
        self._rewrite_benchmark(fixture, benchmark)
        self._assert_rejected(fixture, "correct")

    def test_predictions_must_match_labels_independently(self):
        fixture = self._make_fixture()
        benchmark = read_json(fixture["benchmark_json"])
        wrong = [(prediction + 1) % 10
                 for prediction in fixture["predictions"]]
        benchmark["runs"][1]["stdout"] = prediction_lines(wrong)
        self._rewrite_benchmark(fixture, benchmark)
        self._assert_rejected(fixture)

    def test_fabricated_summary_is_rejected(self):
        mutations = (
            ("correct", lambda summary: summary.update(correct=10000)),
            ("accuracy", lambda summary: summary.update(accuracy=1.0)),
            ("accuracy_pass", lambda summary: summary.update(
                accuracy_pass=not summary["accuracy_pass"])),
            ("median_ms", lambda summary: summary.update(
                median_ms=summary["median_ms"] - 7.5)),
            ("max_ms", lambda summary: summary.update(
                max_ms=summary["max_ms"] + 3.0)),
            ("wall_flag", lambda summary: summary.update(
                all_wall_below_100ms=not summary["all_wall_below_100ms"])),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                fixture = self._make_fixture(
                    correct=9642, walls=[90.0, 120.0, 92.0, 93.0, 94.0])
                benchmark = read_json(fixture["benchmark_json"])
                mutate(benchmark["summary"])
                self._rewrite_benchmark(fixture, benchmark)
                self._assert_rejected(fixture)

    def test_legacy_model_source_is_rejected(self):
        fixture = self._make_fixture(model_source="legacy")
        self._assert_rejected(fixture, "course_train")

    def test_non_full_test_sample_count_is_rejected(self):
        fixture = self._make_fixture(n=9999, correct=9630)
        self._assert_rejected(fixture, "10000")

    def test_missing_wrong_size_or_modified_inputs_are_rejected(self):
        cases = ("missing_images", "short_images", "short_labels",
                 "modified_images", "modified_labels")
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
                elif case == "modified_images":
                    with open(fixture["images"], "r+b") as output:
                        output.seek(IMAGE_PIXELS * 4)
                        output.write(b"\x01\x00\x00\x00")
                else:
                    with open(fixture["labels"], "r+b") as output:
                        output.seek(4)
                        output.write(struct.pack("<i", 9))
                self._assert_rejected(fixture)

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
            self.assertFalse(os.path.lexists(
                os.path.join(fixture["root"], name)))

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
                    self.assertEqual(input_file.read(),
                                     b"preexisting owner data\n")
                self.assertFalse(os.path.lexists(fixture["output_zip"]))
                for other in GENERATED_MEMBERS:
                    if other != name:
                        self.assertFalse(os.path.lexists(
                            os.path.join(fixture["root"], other)))
                os.remove(path)

    def test_missing_whitelist_source_is_rejected(self):
        os.remove(os.path.join(self.root, "include", "tensor.h"))
        fixture = self._make_fixture()
        self._assert_rejected(fixture, "tensor.h")

    def test_symlinked_whitelist_member_is_rejected(self):
        outside = os.path.join(self.directory, "outside-interface.h")
        with open(outside, "w", encoding="utf-8") as output:
            output.write("outside repository\n")
        target = os.path.join(self.root, "course", "interface.h")
        os.remove(target)
        try:
            os.symlink(outside, target)
        except (OSError, NotImplementedError) as error:
            self.skipTest(
                "symlinks are unavailable on this host: {0}".format(error))
        fixture = self._make_fixture()
        self._assert_rejected(fixture)

    def test_unsafe_member_names_are_rejected(self):
        unsafe = (
            "", ".", "..", "../escape.cu", "/absolute/main.cu",
            "C:/main.cu", "C:\\main.cu", "course\\interface.h",
            "./main.cu", "course//interface.h", "course/../main.cu",
            "course/./interface.h", "course/interface.h/",
        )
        for name in unsafe:
            with self.subTest(name=name):
                with self.assertRaises(course_package.PackageError):
                    course_package.validate_member_name(name)
        for name in EXPECTED_WHITELIST + GENERATED_MEMBERS:
            self.assertEqual(course_package.validate_member_name(name), name)

    def test_size_cap_boundaries_are_pure_checks(self):
        course_package.check_archive_limits(99999999, 299999999)
        for compressed, unpacked in (
                (100000000, 1),
                (1, 300000000),
                (100000001, 1),
                (1, 300000001)):
            with self.subTest(compressed=compressed, unpacked=unpacked):
                with self.assertRaises(course_package.PackageError):
                    course_package.check_archive_limits(
                        compressed, unpacked)

        course_package.check_image_limits(4000000, 16000000)
        course_package.check_image_limits(1, 1)
        for image_bytes, pixels in (
                (4000001, 100),
                (100, 16000001),
                (4000001, 16000001)):
            with self.subTest(image_bytes=image_bytes, pixels=pixels):
                with self.assertRaises(course_package.PackageError):
                    course_package.check_image_limits(image_bytes, pixels)

    def test_missing_pillow_is_a_clear_error_and_writes_nothing(self):
        fixture = self._make_fixture()
        blocked = {
            "PIL": None,
            "PIL.Image": None,
            "PIL.ImageDraw": None,
            "PIL.ImageFont": None,
        }
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaises(course_package.PackageError) as context:
                self._package(fixture)
        message = str(context.exception)
        self.assertIn("Pillow", message)
        self.assertIn("development", message.lower())
        self.assertIn("install", message.lower())
        self._assert_no_artifacts(fixture)

    def test_cli_usage_errors_exit_two(self):
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            with self.assertRaises(SystemExit) as context:
                course_package.main([])
        self.assertEqual(context.exception.code, 2)
        self.assertIn("benchmark", stderr.getvalue().lower())

    def test_cli_maps_arguments_and_package_failures(self):
        fixture = self._make_fixture()
        with mock.patch.object(
                course_package, "_repository_root",
                return_value=fixture["root"]), mock.patch.object(
                    course_package, "run_package",
                    return_value={"members": []}) as runner:
            status = course_package.main([
                "--benchmark", fixture["benchmark_json"],
                "--training-log", fixture["training_log"],
                "--output", fixture["output_zip"],
            ])
        self.assertEqual(status, 0)
        runner.assert_called_once()
        arguments = runner.call_args[0]
        self.assertEqual(arguments[0], fixture["benchmark_json"])
        self.assertEqual(arguments[1], fixture["training_log"])
        self.assertEqual(arguments[2], fixture["output_zip"])

        stderr = io.StringIO()
        with mock.patch.object(
                course_package, "_repository_root",
                return_value=fixture["root"]), mock.patch.object(
                    course_package, "run_package",
                    side_effect=course_package.PackageError(
                        "invalid provenance")), mock.patch.object(
                            sys, "stderr", stderr):
            status = course_package.main([
                "--benchmark", fixture["benchmark_json"],
                "--training-log", fixture["training_log"],
                "--output", fixture["output_zip"],
            ])
        self.assertEqual(status, 1)
        self.assertIn("invalid provenance", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
