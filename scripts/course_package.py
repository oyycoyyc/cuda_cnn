from __future__ import print_function

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import statistics
import struct
import sys
import zipfile
import zlib


# Package generation is an external acceptance/publication helper, not part of
# the CMake runtime. It never executes inference, never performs neural-network
# math, and uses only the development-time renderer below.
PACKAGE_SCHEMA = "course-package-v1"
EVALUATION_CONTENT = {
    "mode": "image",
    "image_origin": "archive",
    "result_image": "result.png",
}
EVALUATION_JSON_TEXT = (
    '{"mode":"image","image_origin":"archive","result_image":"result.png"}\n')
BENCHMARK_SCHEMA = "course-benchmark-v1"
REQUIRED_N = 10000
REQUIRED_RUNS = 5
REQUIRED_EPOCHS = 20
ACCURACY_THRESHOLD = 0.96
FULL_WALL_LIMIT_MS = 100.0
INPUT_MODES = ("unit", "byte", "normalized")

# Decimal caps. The brief states MB, and the decimal limits are also stricter
# than the common MiB interpretation.
MAX_ARCHIVE_BYTES = 100000000
MAX_UNPACKED_BYTES = 300000000
MAX_IMAGE_BYTES = 4000000
MAX_IMAGE_PIXELS = 16000000

IMAGE_PIXELS = 28 * 28
CHANNELS = 1
ROWS = 28
COLUMNS = 28
CLASSES = 10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CRC32_RE = re.compile(r"^[0-9a-f]{8}$")
CANONICAL_CLASS_LINE = re.compile(r"^[0-9]$")

# Exact submission source whitelist for the default CMake train/infer graph.
# Sorted, relative, and deliberately excluding tests, tools, Python helpers,
# data, weights, build outputs, diagnostics, evidence, Dockerfile, credentials,
# and symlinks.
WHITELIST_FILES = (
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

_TEXT_SUFFIXES = (".cu", ".cpp", ".h", ".txt", ".json")
_PERSONAL_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s\"']+"),
    re.compile(r"(?i)(?:^|[\s\"'(])/home/[A-Za-z0-9._-]+/"),
    re.compile(r"(?i)(?:^|[\s\"'(])/Users/[A-Za-z0-9._-]+/"),
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bssh-(?:rsa|ed25519)[ \t]+AAAA"),
    re.compile(r"(?i)\baws_secret_access_key\b"),
    re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key)\b[ \t]*[:=][ \t]*"
        r"[\"'][^\"']{8,}[\"']"),
)


class PackageError(ValueError):
    """Invalid or missing provenance, configuration, or artifact."""


def _repository_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _require(condition, message):
    if not condition:
        raise PackageError(message)


def _require_int(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise PackageError("{0} must be an integer".format(name))
    if minimum is not None and value < minimum:
        raise PackageError("{0} must be >= {1}".format(name, minimum))
    if maximum is not None and value > maximum:
        raise PackageError("{0} must be <= {1}".format(name, maximum))
    return value


def _require_number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageError("{0} must be a finite number".format(name))
    number = float(value)
    if not math.isfinite(number):
        raise PackageError("{0} must be a finite number".format(name))
    if minimum is not None and number < minimum:
        raise PackageError("{0} must be >= {1}".format(name, minimum))
    if maximum is not None and number > maximum:
        raise PackageError("{0} must be <= {1}".format(name, maximum))
    return number


def _near(left, right, tolerance=1e-6):
    return abs(float(left) - float(right)) <= tolerance


def validate_member_name(name):
    if not isinstance(name, str):
        raise PackageError("archive member name must be text")
    if not name or name in (".", ".."):
        raise PackageError("archive member name is empty or relative-dot")
    if "\x00" in name:
        raise PackageError("archive member name contains NUL")
    if "\\" in name:
        raise PackageError("archive member name must use forward slashes")
    if name.startswith("/"):
        raise PackageError("archive member name must be relative")
    if ":" in name:
        raise PackageError("archive member name must not contain a drive")
    if name.endswith("/"):
        raise PackageError("archive member name must be a file, not a directory")
    if name.startswith("./"):
        raise PackageError("archive member name must not start with ./")
    components = name.split("/")
    for component in components:
        if component in ("", ".", ".."):
            raise PackageError(
                "archive member name contains an empty/dot/traversal component")
    return name


def check_archive_limits(compressed_bytes, unpacked_bytes):
    _require_int(compressed_bytes, "compressed archive size", 0)
    _require_int(unpacked_bytes, "unpacked archive size", 0)
    _require(compressed_bytes < MAX_ARCHIVE_BYTES,
             "compressed archive must be < {0} bytes".format(
                 MAX_ARCHIVE_BYTES))
    _require(unpacked_bytes < MAX_UNPACKED_BYTES,
             "unpacked archive must be < {0} bytes".format(
                 MAX_UNPACKED_BYTES))


def check_image_limits(image_bytes, pixels):
    _require_int(image_bytes, "PNG size", 0)
    _require_int(pixels, "PNG pixel count", 0)
    _require(image_bytes <= MAX_IMAGE_BYTES,
             "PNG must be <= {0} bytes".format(MAX_IMAGE_BYTES))
    _require(pixels <= MAX_IMAGE_PIXELS,
             "PNG must be <= {0} pixels".format(MAX_IMAGE_PIXELS))


def _require_regular_file(path, description):
    _require(isinstance(path, str) and path,
             "{0} path must be non-empty text".format(description))
    try:
        status = os.lstat(path)
    except OSError as error:
        raise PackageError(
            "{0} is unreadable: {1}".format(description, error))
    if stat.S_ISLNK(status.st_mode):
        raise PackageError("{0} must not be a symlink".format(description))
    if not stat.S_ISREG(status.st_mode):
        raise PackageError("{0} must be a regular file".format(description))
    return status


def _require_directory(path, description):
    _require(isinstance(path, str) and path,
             "{0} path must be non-empty text".format(description))
    try:
        status = os.lstat(path)
    except OSError as error:
        raise PackageError(
            "{0} is unreadable: {1}".format(description, error))
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise PackageError("{0} must be a directory".format(description))


def _read_bytes(path, description):
    _require_regular_file(path, description)
    try:
        with open(path, "rb") as input_file:
            return input_file.read()
    except OSError as error:
        raise PackageError(
            "{0} is unreadable: {1}".format(description, error))


def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path, description):
    _require_regular_file(path, description)
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as input_file:
            while True:
                block = input_file.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise PackageError(
            "{0} is unreadable: {1}".format(description, error))
    return digest.hexdigest()


def _load_json_strict(path):
    raw = _read_bytes(path, "benchmark JSON")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackageError(
            "benchmark JSON must be UTF-8: {0}".format(error))

    def reject_constant(value):
        raise PackageError(
            "benchmark JSON must not contain non-finite value {0}".format(
                value))

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PackageError(
                    "benchmark JSON contains duplicate key {0!r}".format(key))
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except PackageError:
        raise
    except ValueError as error:
        raise PackageError("benchmark JSON is invalid: {0}".format(error))


def _field(mapping, key, description):
    if not isinstance(mapping, dict) or key not in mapping:
        raise PackageError("{0} is missing {1}".format(description, key))
    return mapping[key]


def _parse_training_log(path):
    raw = _read_bytes(path, "training log")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackageError("training log must be UTF-8: {0}".format(error))
    completions = [line for line in text.splitlines()
                   if line.startswith("event=course_train ")]
    if len(completions) != 1:
        raise PackageError(
            "training log must contain exactly one course_train completion")
    line = completions[0]
    values = {}
    for token in line.split():
        if "=" not in token:
            raise PackageError(
                "training completion contains a malformed field")
        key, value = token.split("=", 1)
        if not key or key in values:
            raise PackageError(
                "training completion contains a duplicate/empty field")
        values[key] = value

    for key in ("status", "epochs", "best_epoch", "validation_accuracy",
                "input_mode", "checkpoint_crc32"):
        if key not in values:
            raise PackageError(
                "training completion is missing {0}".format(key))
    if values["status"] != "pass":
        raise PackageError("training completion status must be pass")
    try:
        epochs = int(values["epochs"])
        best_epoch = int(values["best_epoch"])
        validation_accuracy = float(values["validation_accuracy"])
    except ValueError:
        raise PackageError("training completion has malformed numbers")
    _require_int(epochs, "training epochs", REQUIRED_EPOCHS, REQUIRED_EPOCHS)
    _require_int(best_epoch, "best epoch", 1, REQUIRED_EPOCHS)
    _require_number(validation_accuracy, "validation accuracy", 0.0, 1.0)
    if values["input_mode"] not in INPUT_MODES:
        raise PackageError("training completion input mode is unsupported")
    if not CRC32_RE.match(values["checkpoint_crc32"]):
        raise PackageError(
            "training completion checkpoint_crc32 must be eight lowercase hex")
    values["epochs"] = epochs
    values["best_epoch"] = best_epoch
    values["validation_accuracy"] = validation_accuracy
    values["line"] = line
    return values


def _parse_course_meta(path):
    raw = _read_bytes(path, "course.meta")
    if len(raw) > 256:
        raise PackageError("course.meta must be at most 256 bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackageError("course.meta must be UTF-8: {0}".format(error))
    if not text.endswith("\n") or "\r" in text:
        raise PackageError("course.meta must be LF-terminated four lines")
    lines = text[:-1].split("\n")
    if len(lines) != 4:
        raise PackageError("course.meta must contain exactly four lines")
    if lines[0] != "course_lenet_v1":
        raise PackageError("course.meta header must equal course_lenet_v1")
    if not lines[1].startswith("input_mode="):
        raise PackageError("course.meta must contain one input_mode line")
    input_mode = lines[1][len("input_mode="):]
    if input_mode not in INPUT_MODES:
        raise PackageError("course.meta input mode is unsupported")
    if lines[2] != "shape=1,28,28,10":
        raise PackageError("course.meta shape must equal 1,28,28,10")
    if not lines[3].startswith("checkpoint_crc32="):
        raise PackageError(
            "course.meta must contain one checkpoint_crc32 line")
    crc = lines[3][len("checkpoint_crc32="):]
    if not CRC32_RE.match(crc):
        raise PackageError(
            "course.meta checkpoint_crc32 must be eight lowercase hex")
    return input_mode, crc


def _parse_predictions(stdout, count):
    if not isinstance(stdout, str):
        raise PackageError("run stdout must be text")
    lines = stdout.splitlines()
    if len(lines) != count:
        raise PackageError(
            "run stdout must contain exactly {0} class lines".format(count))
    predictions = []
    for index, line in enumerate(lines):
        if not CANONICAL_CLASS_LINE.match(line):
            raise PackageError(
                "run stdout line {0} is not a canonical class integer".format(
                    index + 1))
        predictions.append(int(line))
    return predictions


def _validate_benchmark(path, training):
    benchmark = _load_json_strict(path)
    if not isinstance(benchmark, dict):
        raise PackageError("benchmark JSON must be an object")

    _require(_field(benchmark, "schema", "benchmark") == BENCHMARK_SCHEMA,
             "benchmark schema must equal {0}".format(BENCHMARK_SCHEMA))
    _require(_field(benchmark, "model_source", "benchmark") == "course_train",
             "benchmark model_source must equal course_train")
    _require(_field(benchmark, "profiled", "benchmark") is False,
             "benchmark must be unprofiled")
    n = _require_int(_field(benchmark, "n", "benchmark"),
                     "benchmark n", REQUIRED_N, REQUIRED_N)
    runs_requested = _require_int(
        _field(benchmark, "runs_requested", "benchmark"),
        "benchmark runs_requested", REQUIRED_RUNS, REQUIRED_RUNS)
    _require_int(_field(benchmark, "batch", "benchmark"),
                 "benchmark batch", 1, 16384)
    input_mode = _field(benchmark, "input_mode", "benchmark")
    if input_mode not in INPUT_MODES:
        raise PackageError("benchmark input_mode is unsupported")
    if input_mode != training["input_mode"]:
        raise PackageError(
            "benchmark input mode does not match training completion")

    executable = _field(benchmark, "executable", "benchmark")
    images_path = _field(benchmark, "images_path", "benchmark")
    labels_path = _field(benchmark, "labels_path", "benchmark")
    model_dir = _field(benchmark, "model_dir", "benchmark")
    for value, description in (
            (executable, "infer executable"),
            (images_path, "images file"),
            (labels_path, "labels file")):
        _require(isinstance(value, str) and value,
                 "{0} path must be non-empty text".format(description))
    _require(isinstance(model_dir, str) and model_dir,
             "model directory path must be non-empty text")
    _require_regular_file(executable, "infer executable")
    _require_regular_file(images_path, "images file")
    _require_regular_file(labels_path, "labels file")
    _require_directory(model_dir, "model directory")

    model_path = os.path.join(model_dir, "lenet.bin")
    meta_path = os.path.join(model_dir, "course.meta")
    checkpoint = _read_bytes(model_path, "lenet.bin")
    meta_mode, meta_crc = _parse_course_meta(meta_path)
    if meta_mode != input_mode:
        raise PackageError(
            "course.meta input mode does not match the benchmark")
    if meta_crc != training["checkpoint_crc32"]:
        raise PackageError(
            "training completion CRC does not match course.meta")
    actual_crc = "{0:08x}".format(zlib.crc32(checkpoint) & 0xFFFFFFFF)
    if actual_crc != meta_crc:
        raise PackageError(
            "course.meta checkpoint_crc32 does not match lenet.bin")

    expected_images = n * CHANNELS * ROWS * COLUMNS * 4
    try:
        images_size = os.path.getsize(images_path)
    except OSError as error:
        raise PackageError("images file is unreadable: {0}".format(error))
    if images_size != expected_images:
        raise PackageError(
            "images file must contain exactly {0} bytes, found {1}".format(
                expected_images, images_size))
    labels_raw = _read_bytes(labels_path, "labels file")
    if len(labels_raw) != n * 4:
        raise PackageError(
            "labels file must contain exactly {0} bytes, found {1}".format(
                n * 4, len(labels_raw)))
    labels = struct.unpack("<{0}i".format(n), labels_raw)
    for index, label in enumerate(labels):
        if label < 0 or label >= CLASSES:
            raise PackageError(
                "label {0} must be in [0, 10)".format(index))

    recorded = _field(benchmark, "sha256", "benchmark")
    if not isinstance(recorded, dict):
        raise PackageError("benchmark sha256 must be an object")
    computed = {
        "binary": _sha256_file(executable, "infer executable"),
        "model": _sha256_bytes(checkpoint),
        "meta": _sha256_file(meta_path, "course.meta"),
        "images": _sha256_file(images_path, "images file"),
        "labels": _sha256_bytes(labels_raw),
    }
    for key in ("binary", "model", "meta", "images", "labels"):
        value = _field(recorded, key, "benchmark sha256")
        if not isinstance(value, str) or not SHA256_RE.match(value):
            raise PackageError(
                "benchmark sha256.{0} must be 64 lowercase hex".format(key))
        if value != computed[key]:
            raise PackageError(
                "benchmark sha256.{0} does not match the current file".format(
                    key))

    runs = _field(benchmark, "runs", "benchmark")
    if not isinstance(runs, list) or len(runs) != REQUIRED_RUNS:
        raise PackageError(
            "benchmark must contain exactly five measured runs")
    if runs_requested != REQUIRED_RUNS:
        raise PackageError(
            "benchmark runs_requested must equal five")

    walls = []
    run_correct = []
    predictions = None
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise PackageError("benchmark run {0} must be an object".format(
                index + 1))
        returncode = _require_int(
            _field(run, "returncode", "benchmark run"),
            "run {0} returncode".format(index + 1))
        if returncode != 0:
            raise PackageError(
                "run {0} returncode must be 0".format(index + 1))
        wall_ms = _require_number(
            _field(run, "wall_ms", "benchmark run"),
            "run {0} wall_ms".format(index + 1), 0.0)
        correct = _require_int(
            _field(run, "correct", "benchmark run"),
            "run {0} correct".format(index + 1), 0, n)
        stdout = _field(run, "stdout", "benchmark run")
        stderr = _field(run, "stderr", "benchmark run")
        if not isinstance(stderr, str):
            raise PackageError(
                "run {0} stderr must be text".format(index + 1))
        if "event=course_profile" in stderr:
            raise PackageError(
                "run {0} contains profile evidence but must be unprofiled".format(
                    index + 1))
        current = _parse_predictions(stdout, n)
        recomputed = sum(1 for predicted, expected in zip(current, labels)
                         if predicted == expected)
        if recomputed != correct:
            raise PackageError(
                "run {0} correct does not match raw predictions".format(
                    index + 1))
        if predictions is None:
            predictions = current
        elif current != predictions:
            raise PackageError(
                "all five runs must produce identical predictions")
        walls.append(wall_ms)
        run_correct.append(correct)

    correct = run_correct[0]
    for index, value in enumerate(run_correct):
        if value != correct:
            raise PackageError(
                "run {0} correct count differs from run 1".format(index + 1))
    if correct < int(ACCURACY_THRESHOLD * n):
        raise PackageError(
            "accuracy must be at least {0:.0%} ({1}/{2})".format(
                ACCURACY_THRESHOLD, int(ACCURACY_THRESHOLD * n), n))
    accuracy = float(correct) / float(n)
    if accuracy < ACCURACY_THRESHOLD:
        raise PackageError("accuracy is below the 0.96 threshold")

    summary = _field(benchmark, "summary", "benchmark")
    if not isinstance(summary, dict):
        raise PackageError("benchmark summary must be an object")
    summary_correct = _require_int(
        _field(summary, "correct", "benchmark summary"),
        "summary correct", 0, n)
    if summary_correct != correct:
        raise PackageError(
            "summary correct does not match raw predictions")
    summary_accuracy = _require_number(
        _field(summary, "accuracy", "benchmark summary"),
        "summary accuracy", 0.0, 1.0)
    if not _near(summary_accuracy, accuracy, 1e-12):
        raise PackageError("summary accuracy does not match its correct count")
    summary_pass = _field(summary, "accuracy_pass", "benchmark summary")
    if not isinstance(summary_pass, bool):
        raise PackageError("summary accuracy_pass must be boolean")
    if summary_pass != (accuracy >= ACCURACY_THRESHOLD):
        raise PackageError("summary accuracy_pass is inconsistent")
    if not summary_pass:
        raise PackageError("summary accuracy_pass must be true")

    median_ms = statistics.median(walls)
    summary_median = _require_number(
        _field(summary, "median_ms", "benchmark summary"),
        "summary median_ms", 0.0)
    if not _near(summary_median, median_ms):
        raise PackageError("summary median_ms does not match the five runs")
    maximum_ms = max(walls)
    summary_max = _require_number(
        _field(summary, "max_ms", "benchmark summary"),
        "summary max_ms", 0.0)
    if not _near(summary_max, maximum_ms):
        raise PackageError("summary max_ms does not match the five runs")
    all_below = _field(summary, "all_wall_below_100ms",
                       "benchmark summary")
    if not isinstance(all_below, bool):
        raise PackageError(
            "summary all_wall_below_100ms must be boolean")
    if all_below != all(wall < FULL_WALL_LIMIT_MS for wall in walls):
        raise PackageError(
            "summary all_wall_below_100ms is inconsistent with run walls")

    if "reference_mismatches" in benchmark:
        mismatches = _require_int(
            benchmark["reference_mismatches"],
            "reference_mismatches", 0)
        if mismatches != 0:
            raise PackageError(
                "benchmark reference_mismatches must be zero when present")

    performance_target = (
        "achieved" if all(wall < FULL_WALL_LIMIT_MS for wall in walls)
        else "failed")
    return {
        "benchmark": benchmark,
        "sha256": computed,
        "labels": labels,
        "predictions": predictions,
        "walls": walls,
        "run_correct": run_correct,
        "correct": correct,
        "accuracy": accuracy,
        "median_ms": median_ms,
        "max_ms": maximum_ms,
        "performance_target": performance_target,
        "input_mode": input_mode,
        "checkpoint_crc32": meta_crc,
        "training": training,
    }


def _validate_whitelist(root):
    _require(os.path.isdir(root), "repository root must be a directory")
    if len(set(WHITELIST_FILES)) != len(WHITELIST_FILES):
        raise PackageError("submission whitelist contains duplicate names")
    root_real = os.path.realpath(root)
    files = {}
    for relative in WHITELIST_FILES:
        validate_member_name(relative)
        path = os.path.join(root, *relative.split("/"))
        status = _require_regular_file(path, relative)
        if status.st_nlink != 1:
            raise PackageError(
                "{0} must be a single-link regular file".format(relative))
        real = os.path.realpath(path)
        try:
            common = os.path.commonpath([root_real, real])
        except ValueError:
            common = None
        if common != root_real:
            raise PackageError(
                "{0} resolves outside the repository root".format(relative))
        files[relative] = path
    return files


def _scan_payload(name, data):
    is_text = (name in ("report", "evaluation.json") or
               name.endswith(_TEXT_SUFFIXES))
    if name == "result.png" or not is_text:
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackageError(
            "packaged text member {0} is not UTF-8: {1}".format(name, error))
    for pattern in _PERSONAL_PATH_PATTERNS:
        if pattern.search(text):
            raise PackageError(
                "packaged member {0} contains a personal absolute path".format(
                    name))
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            raise PackageError(
                "packaged member {0} contains credential material".format(
                    name))


def _input_assumption(input_mode):
    if input_mode == "unit":
        return (
            "caller supplies unit floats [0,1]; inference applies the "
            "persisted GPU transform (x - 0.1307) / 0.3081 from course.meta; "
            "the mode is never guessed from test images or overridden")
    if input_mode == "byte":
        return (
            "caller supplies byte-domain floats [0,255]; inference applies "
            "the persisted GPU transform (x / 255 - 0.1307) / 0.3081 from "
            "course.meta; the mode is never guessed or overridden")
    return (
        "caller supplies already-normalized floats; inference applies the "
        "persisted normalized mode from course.meta without re-normalizing "
        "or overriding it")


def metric_lines(accuracy, median_ms):
    return [
        "Test Accuracy: {0:.2f}%".format(float(accuracy) * 100.0),
        "Test Time: {0:.2f} ms (median of 5 full-process runs)".format(
            float(median_ms)),
    ]


def _import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise PackageError(
            "Pillow is required only for development image rendering; it is "
            "not installed and this helper performs no automatic installation")
    return Image, ImageDraw, ImageFont


def render_result_png(accuracy, median_ms, performance_target):
    Image, ImageDraw, ImageFont = _import_pillow()
    image = Image.new("RGB", (900, 320), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=24)
    except TypeError:  # pragma: no cover - older development Pillow fallback
        font = ImageFont.load_default()
    draw.text((24, 20), "Course LeNet submission", font=font,
              fill=(0, 0, 0))
    y = 90
    for line in metric_lines(accuracy, median_ms):
        draw.text((24, y), line, font=font, fill=(0, 0, 0))
        y += 50
    verdict = (
        "FAILED" if performance_target == "failed" else "ACHIEVED")
    draw.text(
        (24, y + 20),
        "Performance target (full-process <100 ms): {0}".format(verdict),
        font=font, fill=(160, 0, 0) if verdict == "FAILED" else (0, 96, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    content = output.getvalue()
    check_image_limits(len(content), image.width * image.height)
    return content


def _report_text(validated):
    benchmark = validated["benchmark"]
    training = validated["training"]
    sha256 = validated["sha256"]
    accuracy = validated["accuracy"]
    correct = validated["correct"]
    n = benchmark["n"]
    lines = [
        "Course submission report",
        "Model source: course_train",
        "Input mode: {0}".format(validated["input_mode"]),
        "Preprocessing assumption: {0}".format(
            _input_assumption(validated["input_mode"])),
        "Training completion: {0}".format(training["line"]),
        "Training epochs: {0}".format(training["epochs"]),
        "Configuration: n={0} batch={1} runs={2} profiled={3}".format(
            n, benchmark["batch"], benchmark["runs_requested"],
            "true" if benchmark["profiled"] else "false"),
        "Accuracy target (>=96%): PASS",
    ]
    for index, (wall_ms, run_correct) in enumerate(
            zip(validated["walls"], validated["run_correct"]), start=1):
        lines.append(
            "Run {0}: wall_ms={1:.3f} correct={2}".format(
                index, wall_ms, run_correct))
    lines.extend([
        "Median full-process wall: {0:.3f} ms".format(
            validated["median_ms"]),
        "Max full-process wall: {0:.3f} ms".format(validated["max_ms"]),
        "max_ms={0:.3f}".format(validated["max_ms"]),
        "Test Accuracy: {0:.2f}% ({1}/{2})".format(
            accuracy * 100.0, correct, n),
        metric_lines(accuracy, validated["median_ms"])[1],
    ])
    verdict = (
        "FAILED" if validated["performance_target"] == "failed"
        else "ACHIEVED")
    lines.extend([
        "Performance target (full-process <100 ms): {0}".format(verdict),
        "Leaderboard timing: unresolved; the official leaderboard timing "
        "scope is unknown and no leaderboard performance is claimed here.",
        "Provenance:",
        "checkpoint_crc32={0}".format(validated["checkpoint_crc32"]),
        "binary_sha256={0}".format(sha256["binary"]),
        "model_sha256={0}".format(sha256["model"]),
        "meta_sha256={0}".format(sha256["meta"]),
        "images_sha256={0}".format(sha256["images"]),
        "labels_sha256={0}".format(sha256["labels"]),
    ])
    return "\n".join(lines) + "\n"


def _build_archive(root, source_files, generated):
    payloads = {}
    for relative in WHITELIST_FILES:
        _require(relative in source_files,
                 "{0} is missing from the validated whitelist".format(
                     relative))
        payloads[relative] = _read_bytes(source_files[relative], relative)
    for name, content in generated.items():
        payloads[name] = content
    names = sorted(payloads)
    expected = tuple(sorted(WHITELIST_FILES + tuple(generated.keys())))
    if tuple(names) != expected or len(set(names)) != len(names):
        raise PackageError("archive member set does not match the whitelist")
    for name in names:
        validate_member_name(name)
        _scan_payload(name, payloads[name])
    unpacked = sum(len(payloads[name]) for name in names)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=False) as archive:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name])
    content = buffer.getvalue()
    check_archive_limits(len(content), unpacked)
    return content, names


def _write_exclusive(path, content):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        output = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        try:
            os.remove(path)
        except OSError:
            pass
        raise
    try:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
        output.close()
    except BaseException:
        try:
            output.close()
        except OSError:
            pass
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def _check_output_paths(root, output_zip):
    for name in GENERATED_MEMBERS:
        path = os.path.join(root, name)
        if os.path.lexists(path):
            raise FileExistsError(
                "root artifact already exists: {0}".format(path))
    parent = os.path.dirname(os.path.abspath(output_zip))
    if not os.path.isdir(parent):
        raise PackageError(
            "archive output parent directory does not exist: {0}".format(
                parent))
    if os.path.lexists(output_zip):
        raise FileExistsError(
            "archive already exists: {0}".format(output_zip))


def _publish(root, contents, output_zip):
    published = []
    try:
        for name in GENERATED_MEMBERS:
            path = os.path.join(root, name)
            _write_exclusive(path, contents[name])
            published.append(path)
        _write_exclusive(output_zip, contents["archive"])
    except BaseException:
        for path in published:
            try:
                os.remove(path)
            except OSError:
                pass
        raise


def run_package(benchmark_json, training_log, output_zip, root=None):
    repository_root = _repository_root() if root is None else root
    _check_output_paths(repository_root, output_zip)
    source_files = _validate_whitelist(repository_root)
    training = _parse_training_log(training_log)
    validated = _validate_benchmark(benchmark_json, training)
    png = render_result_png(
        validated["accuracy"], validated["median_ms"],
        validated["performance_target"])
    report = _report_text(validated).encode("utf-8")
    evaluation = EVALUATION_JSON_TEXT.encode("utf-8")
    archive, members = _build_archive(
        repository_root, source_files,
        {
            "evaluation.json": evaluation,
            "report": report,
            "result.png": png,
        })
    contents = {
        "evaluation.json": evaluation,
        "report": report,
        "result.png": png,
        "archive": archive,
    }
    _publish(repository_root, contents, output_zip)
    return {
        "members": members,
        "correct": validated["correct"],
        "accuracy": validated["accuracy"],
        "median_ms": validated["median_ms"],
        "max_ms": validated["max_ms"],
        "performance_target": validated["performance_target"],
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Validate final course model evidence and build the "
                    "submission archive")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args(arguments)
    try:
        run_package(options.benchmark, options.training_log, options.output,
                    root=_repository_root())
    except (PackageError, OSError) as error:
        print("course_package: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
