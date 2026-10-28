from __future__ import print_function

import argparse
import io
import math
import os
import statistics
import struct
import sys
import zlib

# Import both ways so the helper works as `scripts.course_warm_package`, as a
# direct script, and from the original cold package dispatcher. The warm module
# reuses the reviewed collector parser and the cold whitelist/training/CRC/
# publication safeguards instead of forking them.
try:
    from scripts import course_package
    from scripts import course_warm_benchmark
except ImportError:  # pragma: no cover - direct script execution
    import course_package
    import course_warm_benchmark


PackageError = course_package.PackageError
WARM_BENCHMARK_SCHEMA = course_warm_benchmark.WARM_BENCHMARK_SCHEMA
TIMING_SCOPE = course_warm_benchmark.TIMING_SCOPE
HASH_FIELDS = course_warm_benchmark.HASH_FIELDS
parse_tool_output = course_warm_benchmark.parse_tool_output
WHITELIST_FILES = course_package.WHITELIST_FILES
GENERATED_MEMBERS = course_package.GENERATED_MEMBERS
EVALUATION_JSON_TEXT = course_package.EVALUATION_JSON_TEXT
INPUT_MODES = course_package.INPUT_MODES

REQUIRED_N = 10000
MIN_RUNS = 101
MIN_WARMUPS = 1
REQUIRED_EPOCHS = 20
ACCURACY_THRESHOLD = 0.96
WARM_LIMIT_MS = 100.0
MAX_BATCH = 16384
MAX_ROUND_COUNT = 100000

_TOP_LEVEL_FIELDS = frozenset((
    "schema", "timing_scope", "executable", "official_infer", "images_path",
    "labels_path", "model_dir", "model_source", "n", "batch", "input_mode",
    "warmups", "runs", "sha256", "sha256_after", "hashes_stable",
    "returncode", "process_wall_ms", "stdout", "stderr", "warmup_records",
    "measured_records", "predictions_identical_all", "status", "errors",
    "summary"))
_SUMMARY_FIELDS = frozenset((
    "mean_ms", "median_ms", "min_ms", "max_ms", "correct", "accuracy",
    "accuracy_pass", "mean_below_100ms", "median_below_100ms"))


def _repository_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _require(condition, message):
    if not condition:
        raise PackageError(message)


def _require_bool(value, name, expected):
    _require(isinstance(value, bool), "{0} must be boolean".format(name))
    _require(value is expected, "{0} must be {1}".format(name, expected))


def _require_text(value, name):
    _require(isinstance(value, str) and value,
             "{0} must be non-empty text".format(name))
    return value


def _require_hash_map(mapping, name):
    _require(isinstance(mapping, dict), "{0} must be an object".format(name))
    keys = set(mapping)
    if keys != set(HASH_FIELDS):
        missing = sorted(set(HASH_FIELDS) - keys)
        extra = sorted(keys - set(HASH_FIELDS))
        raise PackageError(
            "{0} must contain exactly the six identities "
            "(missing={1}, extra={2})".format(name, missing, extra))
    for key in HASH_FIELDS:
        value = mapping[key]
        if not isinstance(value, str) or not course_package.SHA256_RE.match(
                value):
            raise PackageError(
                "{0}.{1} must be 64 lowercase hex".format(name, key))
    return mapping


def _recount(predictions, labels, n):
    correct = sum(
        1 for predicted, expected in zip(predictions, labels)
        if predicted == expected)
    _require(len(predictions) == n, "raw predictions must contain {0} "
             "class values".format(n))
    return correct


# Validates the complete warm evidence record without executing anything.
def _validate_benchmark(path, training):
    benchmark = course_package._load_json_strict(path)
    if not isinstance(benchmark, dict):
        raise PackageError("warm benchmark must be an object")

    keys = set(benchmark)
    if keys != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - keys)
        extra = sorted(keys - _TOP_LEVEL_FIELDS)
        raise PackageError(
            "warm benchmark must contain exactly the 26 reviewed fields "
            "(missing={0}, extra={1})".format(missing, extra))

    if benchmark["schema"] != WARM_BENCHMARK_SCHEMA:
        raise PackageError(
            "warm benchmark schema must equal {0}".format(
                WARM_BENCHMARK_SCHEMA))
    if benchmark["timing_scope"] != TIMING_SCOPE:
        raise PackageError(
            "warm benchmark timing_scope must equal {0}".format(TIMING_SCOPE))
    if benchmark["model_source"] != "course_train":
        raise PackageError(
            "warm benchmark model_source must equal course_train")

    n = course_package._require_int(
        benchmark["n"], "warm benchmark n", REQUIRED_N, REQUIRED_N)
    batch = course_package._require_int(
        benchmark["batch"], "warm benchmark batch", 1, MAX_BATCH)
    warmups = course_package._require_int(
        benchmark["warmups"], "warm benchmark warmups", MIN_WARMUPS,
        MAX_ROUND_COUNT)
    runs = course_package._require_int(
        benchmark["runs"], "warm benchmark runs", MIN_RUNS, MAX_ROUND_COUNT)
    returncode = course_package._require_int(
        benchmark["returncode"], "warm benchmark returncode", 0, 0)
    process_wall_ms = course_package._require_number(
        benchmark["process_wall_ms"], "warm benchmark process_wall_ms", 0.0)
    _require(process_wall_ms > 0.0,
             "warm benchmark process_wall_ms must be positive finite")

    if benchmark["status"] != "pass":
        raise PackageError("warm benchmark status must equal pass")
    errors = benchmark["errors"]
    _require(isinstance(errors, list),
             "warm benchmark errors must be a list")
    if errors:
        raise PackageError("warm benchmark errors must be empty")
    _require_bool(benchmark["hashes_stable"],
                  "warm benchmark hashes_stable", True)
    _require_bool(benchmark["predictions_identical_all"],
                  "warm benchmark predictions_identical_all", True)

    input_mode = benchmark["input_mode"]
    if input_mode not in INPUT_MODES:
        raise PackageError("warm benchmark input_mode is unsupported")
    if input_mode != training["input_mode"]:
        raise PackageError(
            "warm benchmark input mode does not match training completion")

    executable = _require_text(
        benchmark["executable"], "warm benchmark executable")
    official_infer = _require_text(
        benchmark["official_infer"], "official infer executable")
    images_path = _require_text(benchmark["images_path"], "images file")
    labels_path = _require_text(benchmark["labels_path"], "labels file")
    model_dir = _require_text(benchmark["model_dir"], "model directory")

    course_package._require_regular_file(
        executable, "warm benchmark executable")
    course_package._require_regular_file(
        official_infer, "official infer executable")
    course_package._require_regular_file(images_path, "images file")
    course_package._require_regular_file(labels_path, "labels file")
    course_package._require_directory(model_dir, "model directory")

    model_path = os.path.join(model_dir, "lenet.bin")
    meta_path = os.path.join(model_dir, "course.meta")
    checkpoint = course_package._read_bytes(model_path, "lenet.bin")
    meta_mode, meta_crc = course_package._parse_course_meta(meta_path)
    if meta_mode != input_mode:
        raise PackageError(
            "course.meta input mode does not match the warm benchmark")
    if meta_crc != training["checkpoint_crc32"]:
        raise PackageError(
            "training completion CRC does not match course.meta")
    actual_crc = "{0:08x}".format(zlib.crc32(checkpoint) & 0xFFFFFFFF)
    if actual_crc != meta_crc:
        raise PackageError(
            "course.meta checkpoint_crc32 does not match lenet.bin")

    try:
        images_size = os.path.getsize(images_path)
    except OSError as error:
        raise PackageError("images file is unreadable: {0}".format(error))
    expected_images = (
        n * course_package.CHANNELS * course_package.ROWS *
        course_package.COLUMNS * 4)
    if images_size != expected_images:
        raise PackageError(
            "images file must contain exactly {0} bytes, found {1}".format(
                expected_images, images_size))

    labels_raw = course_package._read_bytes(labels_path, "labels file")
    if len(labels_raw) != n * 4:
        raise PackageError(
            "labels file must contain exactly {0} bytes, found {1}".format(
                n * 4, len(labels_raw)))
    labels = struct.unpack("<{0}i".format(n), labels_raw)
    for index, label in enumerate(labels):
        if label < 0 or label >= course_package.CLASSES:
            raise PackageError(
                "label {0} must be in [0, 10)".format(index))

    recorded = _require_hash_map(benchmark["sha256"], "warm benchmark sha256")
    recorded_after = _require_hash_map(
        benchmark["sha256_after"], "warm benchmark sha256_after")
    computed = {
        "binary": course_package._sha256_file(
            executable, "warm benchmark executable"),
        "official_infer": course_package._sha256_file(
            official_infer, "official infer executable"),
        "model": course_package._sha256_bytes(checkpoint),
        "meta": course_package._sha256_file(meta_path, "course.meta"),
        "images": course_package._sha256_file(images_path, "images file"),
        "labels": course_package._sha256_bytes(labels_raw),
    }
    for key in HASH_FIELDS:
        if recorded[key] != computed[key]:
            raise PackageError(
                "warm benchmark sha256.{0} does not match the current "
                "file".format(key))
        if recorded_after[key] != computed[key]:
            raise PackageError(
                "warm benchmark sha256_after.{0} does not match the current "
                "file".format(key))

    stdout = benchmark["stdout"]
    stderr = benchmark["stderr"]
    if not isinstance(stdout, str):
        raise PackageError("warm benchmark stdout must be text")
    if not isinstance(stderr, str):
        raise PackageError("warm benchmark stderr must be text")
    try:
        parsed = parse_tool_output(
            stdout, stderr, returncode, n, batch, warmups, runs, input_mode)
    except ValueError as error:
        raise PackageError("warm raw protocol: {0}".format(error))

    stored_warmups = benchmark["warmup_records"]
    stored_measured = benchmark["measured_records"]
    if stored_warmups != parsed["warmup_records"]:
        raise PackageError(
            "warm benchmark warmup_records do not match the raw protocol")
    if stored_measured != parsed["measured_records"]:
        raise PackageError(
            "warm benchmark measured_records do not match the raw protocol")

    predictions = parsed["predictions"]
    correct = _recount(predictions, labels, n)
    accuracy = float(correct) / float(n)
    if correct < int(ACCURACY_THRESHOLD * n):
        raise PackageError(
            "accuracy must be at least {0:.0%} ({1}/{2})".format(
                ACCURACY_THRESHOLD, int(ACCURACY_THRESHOLD * n), n))
    if accuracy < ACCURACY_THRESHOLD:
        raise PackageError("accuracy is below the 0.96 threshold")

    walls = [record["wall_ms"] for record in parsed["measured_records"]]
    mean_ms = statistics.mean(walls)
    median_ms = statistics.median(walls)
    min_ms = min(walls)
    max_ms = max(walls)

    summary = benchmark["summary"]
    if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
        raise PackageError(
            "warm benchmark summary must contain exactly the nine reviewed "
            "fields")
    summary_mean = course_package._require_number(
        summary["mean_ms"], "warm benchmark summary mean_ms", 0.0)
    if not course_package._near(summary_mean, mean_ms):
        raise PackageError(
            "warm benchmark summary mean_ms does not match the measured "
            "rounds")
    summary_median = course_package._require_number(
        summary["median_ms"], "warm benchmark summary median_ms", 0.0)
    if not course_package._near(summary_median, median_ms):
        raise PackageError(
            "warm benchmark summary median_ms does not match the measured "
            "rounds")
    summary_min = course_package._require_number(
        summary["min_ms"], "warm benchmark summary min_ms", 0.0)
    if not course_package._near(summary_min, min_ms):
        raise PackageError(
            "warm benchmark summary min_ms does not match the measured "
            "rounds")
    summary_max = course_package._require_number(
        summary["max_ms"], "warm benchmark summary max_ms", 0.0)
    if not course_package._near(summary_max, max_ms):
        raise PackageError(
            "warm benchmark summary max_ms does not match the measured "
            "rounds")
    summary_correct = course_package._require_int(
        summary["correct"], "warm benchmark summary correct", 0, n)
    if summary_correct != correct:
        raise PackageError(
            "warm benchmark summary correct does not match the raw "
            "predictions")
    summary_accuracy = course_package._require_number(
        summary["accuracy"], "warm benchmark summary accuracy", 0.0, 1.0)
    if not course_package._near(summary_accuracy, accuracy, 1e-12):
        raise PackageError(
            "warm benchmark summary accuracy does not match its correct "
            "count")
    _require_bool(summary["accuracy_pass"],
                  "warm benchmark summary accuracy_pass",
                  accuracy >= ACCURACY_THRESHOLD)
    _require_bool(summary["mean_below_100ms"],
                  "warm benchmark summary mean_below_100ms",
                  mean_ms < WARM_LIMIT_MS)
    _require_bool(summary["median_below_100ms"],
                  "warm benchmark summary median_below_100ms",
                  median_ms < WARM_LIMIT_MS)

    performance_target = (
        "achieved" if (mean_ms < WARM_LIMIT_MS and
                       median_ms < WARM_LIMIT_MS) else "failed")
    return {
        "benchmark": benchmark,
        "sha256": computed,
        "labels": labels,
        "predictions": predictions,
        "training": training,
        "input_mode": input_mode,
        "checkpoint_crc32": meta_crc,
        "warmup_walls": [record["wall_ms"]
                         for record in parsed["warmup_records"]],
        "measured_walls": walls,
        "correct": correct,
        "accuracy": accuracy,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "performance_target": performance_target,
        "warmups": warmups,
        "runs": runs,
        "n": n,
        "batch": batch,
    }


def metric_lines(accuracy, mean_ms, median_ms):
    return [
        "Test Accuracy: {0:.2f}%".format(float(accuracy) * 100.0),
        "Test Time: mean {0:.2f} ms; median {1:.2f} ms".format(
            float(mean_ms), float(median_ms)),
    ]


def render_result_png(accuracy, mean_ms, median_ms, performance_target,
                      min_ms, max_ms, warmups, runs, n):
    Image, ImageDraw, ImageFont = course_package._import_pillow()
    image = Image.new("RGB", (1200, 620), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # pragma: no cover - older development Pillow fallback
        font = ImageFont.load_default()
    lines = [
        "Course LeNet warm full-request submission",
        metric_lines(accuracy, mean_ms, median_ms)[0],
        metric_lines(accuracy, mean_ms, median_ms)[1],
        "{0} warmups; {1} measured rounds; {2} images per round".format(
            int(warmups), int(runs), int(n)),
        "Warm full-request range: {0:.2f} - {1:.2f} ms".format(
            float(min_ms), float(max_ms)),
        "Scope: H2D + GPU input preparation + forward + argmax + D2H +",
        "synchronization + host validation (per measured round)",
    ]
    y = 24
    for line in lines:
        draw.text((24, y), line, font=font, fill=(0, 0, 0))
        y += 56
    verdict = "FAILED" if performance_target == "failed" else "ACHIEVED"
    draw.text(
        (24, y),
        "Performance target (warm full-request mean and median <100 ms): "
        "{0}".format(verdict),
        font=font, fill=(160, 0, 0) if verdict == "FAILED" else (0, 96, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    content = output.getvalue()
    course_package.check_image_limits(len(content), image.width * image.height)
    return content


def _report_text(validated):
    benchmark = validated["benchmark"]
    training = validated["training"]
    sha256 = validated["sha256"]
    accuracy = validated["accuracy"]
    correct = validated["correct"]
    n = validated["n"]
    lines = [
        "Course warm full-request submission report",
        "Model source: course_train",
        "Input mode: {0}".format(validated["input_mode"]),
        "Preprocessing assumption: {0}".format(
            course_package._input_assumption(validated["input_mode"])),
        "Preprocessing caveat: official course preprocessing remains "
        "unconfirmed.",
        "Training completion: {0}".format(training["line"]),
        "Training epochs: {0}".format(training["epochs"]),
        "Configuration: n={0} batch={1} warmups={2} runs={3} "
        "model_source=course_train profile=false".format(
            n, validated["batch"], validated["warmups"],
            validated["runs"]),
        "Timing scope: warm full-request per measured round, from before the "
        "sticky-flag reset through H2D input transfer, GPU input preparation, "
        "the original forward pass, argmax, D2H readback, stream "
        "synchronization, and host output validation.",
        "Warmups: {0} retained separately and excluded from measured "
        "statistics; every warmup and measured round passed the tool's "
        "prediction-invariance check.".format(validated["warmups"]),
    ]
    for index, wall_ms in enumerate(validated["warmup_walls"], start=1):
        lines.append(
            "Warmup {0}: wall_ms={1:.3f} ms".format(index, wall_ms))
    for index, wall_ms in enumerate(validated["measured_walls"], start=1):
        lines.append(
            "Measured {0}: wall_ms={1:.3f} ms".format(index, wall_ms))
    lines.extend([
        "Mean measured: {0:.3f} ms".format(validated["mean_ms"]),
        "Median measured: {0:.3f} ms".format(validated["median_ms"]),
        "Min measured: {0:.3f} ms".format(validated["min_ms"]),
        "Max measured: {0:.3f} ms".format(validated["max_ms"]),
        "Mean and median include every measured outlier; no measured sample "
        "was discarded.",
        "Test Accuracy: {0:.2f}% ({1}/{2})".format(
            accuracy * 100.0, correct, n),
        metric_lines(accuracy, validated["mean_ms"],
                     validated["median_ms"])[1],
        "Accuracy target (>=96%): PASS",
    ])
    verdict = (
        "FAILED" if validated["performance_target"] == "failed"
        else "ACHIEVED")
    lines.extend([
        "Performance target (warm full-request mean and median <100 ms): "
        "{0}".format(verdict),
        "Included per measured round: H2D input transfer, GPU input "
        "preparation, original forward pass, argmax, D2H readback, stream "
        "synchronization, and host output validation.",
        "Excluded from the measured interval: process setup, model/image "
        "disk loading, CUDA initialization, allocation, weight upload, and "
        "final stdout formatting; these are outside initialization by the "
        "user's definition, not hidden unreported overhead.",
        "The historical cold-process measurement and its scope are "
        "unchanged; this warm result does not claim the earlier cold-startup "
        "goal changed.",
        "Leaderboard timing: unresolved; the official leaderboard timing "
        "scope is unknown and no leaderboard performance is claimed here.",
        "Provenance:",
        "checkpoint_crc32={0}".format(validated["checkpoint_crc32"]),
        "binary_sha256={0}".format(sha256["binary"]),
        "official_infer_sha256={0}".format(sha256["official_infer"]),
        "model_sha256={0}".format(sha256["model"]),
        "meta_sha256={0}".format(sha256["meta"]),
        "images_sha256={0}".format(sha256["images"]),
        "labels_sha256={0}".format(sha256["labels"]),
    ])
    return "\n".join(lines) + "\n"


def run_warm_package(benchmark_json, training_log, output_zip, root=None):
    repository_root = _repository_root() if root is None else root
    course_package._check_output_paths(repository_root, output_zip)
    source_files = course_package._validate_whitelist(repository_root)
    training = course_package._parse_training_log(training_log)
    validated = _validate_benchmark(benchmark_json, training)
    png = render_result_png(
        validated["accuracy"], validated["mean_ms"], validated["median_ms"],
        validated["performance_target"], validated["min_ms"],
        validated["max_ms"], validated["warmups"], validated["runs"],
        validated["n"])
    report = _report_text(validated).encode("utf-8")
    evaluation = EVALUATION_JSON_TEXT.encode("utf-8")
    archive, members = course_package._build_archive(
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
    course_package._publish(repository_root, contents, output_zip)
    return {
        "members": members,
        "correct": validated["correct"],
        "accuracy": validated["accuracy"],
        "mean_ms": validated["mean_ms"],
        "median_ms": validated["median_ms"],
        "min_ms": validated["min_ms"],
        "max_ms": validated["max_ms"],
        "performance_target": validated["performance_target"],
    }


def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Validate warm full-request evidence and build the "
                    "submission archive")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--output", required=True)
    options = parser.parse_args(arguments)
    try:
        run_warm_package(options.benchmark, options.training_log,
                         options.output, root=_repository_root())
    except (PackageError, OSError) as error:
        print("course_warm_package: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
