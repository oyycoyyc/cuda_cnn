from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import time


WARM_BENCHMARK_SCHEMA = "course-warm-benchmark-v1"
TIMING_SCOPE = (
    "warm_full_request_h2d_prepare_forward_argmax_d2h_validation")
ACCURACY_THRESHOLD = 0.96
DEFAULT_BATCH = 1024
DEFAULT_WARMUPS = 5
DEFAULT_RUNS = 101
FULL_REQUEST_LIMIT_MS = 100.0
PROCESS_TIMEOUT_SECONDS = 600
MODEL_SOURCES = ("legacy", "course_train")
INPUT_MODES = ("unit", "byte", "normalized")
IMAGE_PIXELS = 28 * 28
CLASS_COUNT = 10
SHAPE = (1, 28, 28, 10)
MAX_ROUND_COUNT = 100000
MAX_META_BYTES = 256

ROUND_FIELDS = (
    "event", "phase", "iteration", "samples", "wall_ms",
    "predictions_identical")
COMPLETION_FIELDS = (
    "event", "status", "warmups", "runs", "samples", "batch", "input_mode")
HASH_FIELDS = (
    "binary", "official_infer", "model", "meta", "images", "labels")
CRC_PATTERN = re.compile(r"^[0-9a-f]{8}$")


# Requires an integer that cannot be a bool and is at least the given minimum.
def _require_int(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            "{0} must be an integer >= {1}".format(name, minimum))
    return value


# Parses a canonical non-negative decimal field, rejecting signs and padding.
def _parse_uint(token, name):
    if not token or not token.isdigit():
        raise ValueError("{0} must be canonical decimal digits".format(name))
    if len(token) > 1 and token[0] == "0":
        raise ValueError("{0} must not have leading zeros".format(name))
    return int(token)


# Parses a strictly positive finite wall duration.
def _parse_wall_ms(token):
    if not token:
        raise ValueError("wall_ms must be a positive finite number")
    try:
        value = float(token)
    except ValueError:
        raise ValueError("wall_ms must be a positive finite number")
    if not math.isfinite(value) or not value > 0.0:
        raise ValueError("wall_ms must be a positive finite number")
    return value


# Splits one classic single-space record and enforces the exact field order.
def _parse_record_fields(line, expected_fields):
    if not line or line != line.strip():
        raise ValueError("protocol record has leading or trailing whitespace")
    if "  " in line or "\t" in line:
        raise ValueError("protocol fields must be separated by one space")
    chunks = line.split(" ")
    if len(chunks) != len(expected_fields):
        raise ValueError(
            "protocol record needs exactly {0} fields".format(
                len(expected_fields)))
    fields = {}
    for chunk, expected in zip(chunks, expected_fields):
        name, separator, value = chunk.partition("=")
        if separator != "=" or not name or not value:
            raise ValueError("malformed protocol field: {0}".format(chunk))
        if name != expected:
            raise ValueError(
                "protocol field order must be {0}".format(expected_fields))
        fields[name] = value
    return fields


# Validates one warmup or measured record against the requested phase shape.
def _parse_round(line, phase, iteration, count):
    fields = _parse_record_fields(line, ROUND_FIELDS)
    if fields["event"] != "course_warm_iteration":
        raise ValueError("round event must be course_warm_iteration")
    if fields["phase"] != phase:
        raise ValueError("round phase must be {0}".format(phase))
    parsed_iteration = _parse_uint(fields["iteration"], "iteration")
    if parsed_iteration != iteration:
        raise ValueError(
            "round iteration must be {0}".format(iteration))
    samples = _parse_uint(fields["samples"], "samples")
    if samples != count:
        raise ValueError("round samples must equal {0}".format(count))
    wall_ms = _parse_wall_ms(fields["wall_ms"])
    if fields["predictions_identical"] != "1":
        raise ValueError("round predictions_identical must be 1")
    return {
        "iteration": parsed_iteration,
        "samples": samples,
        "wall_ms": wall_ms,
        "predictions_identical": True,
    }


# Parses one final canonical N-line class vector.
def _parse_predictions(stdout, count):
    lines = stdout.splitlines()
    if len(lines) != count:
        raise ValueError(
            "stdout must contain exactly {0} prediction lines".format(count))
    predictions = []
    for index, line in enumerate(lines):
        value = _parse_uint(line, "prediction")
        if value >= CLASS_COUNT:
            raise ValueError(
                "prediction {0} must be in [0, 10)".format(index))
        predictions.append(value)
    return predictions


# Reusable pure parser for the reviewed tool's raw stdout/stderr protocol.
def parse_tool_output(stdout, stderr, returncode, count, batch, warmups,
                      runs, input_mode):
    _require_int(count, "count", 1)
    _require_int(batch, "batch", 1)
    _require_int(warmups, "warmups", 1)
    _require_int(runs, "runs", 1)
    if warmups > MAX_ROUND_COUNT or runs > MAX_ROUND_COUNT:
        raise ValueError("warmups and runs must be in [1, 100000]")
    if input_mode not in INPUT_MODES:
        raise ValueError("input mode must be unit, byte, or normalized")
    if isinstance(returncode, bool) or returncode != 0:
        raise ValueError(
            "warm benchmark exited with status {0}".format(returncode))

    lines = stderr.splitlines()
    if len(lines) != warmups + runs + 1:
        raise ValueError(
            "stderr must contain {0} round records and one completion".format(
                warmups + runs))

    warmup_records = [
        _parse_round(lines[index], "warmup", index + 1, count)
        for index in range(warmups)
    ]
    measured_records = [
        _parse_round(lines[warmups + index], "measure", index + 1, count)
        for index in range(runs)
    ]

    completion_fields = _parse_record_fields(lines[-1], COMPLETION_FIELDS)
    if completion_fields["event"] != "course_warm_complete":
        raise ValueError("completion event must be course_warm_complete")
    if completion_fields["status"] != "pass":
        raise ValueError("completion status must be pass")
    completion = {
        "status": "pass",
        "warmups": _parse_uint(completion_fields["warmups"], "warmups"),
        "runs": _parse_uint(completion_fields["runs"], "runs"),
        "samples": _parse_uint(completion_fields["samples"], "samples"),
        "batch": _parse_uint(completion_fields["batch"], "batch"),
        "input_mode": completion_fields["input_mode"],
    }
    expected = {
        "warmups": warmups,
        "runs": runs,
        "samples": count,
        "batch": batch,
        "input_mode": input_mode,
    }
    for name in ("warmups", "runs", "samples", "batch", "input_mode"):
        if completion[name] != expected[name]:
            raise ValueError(
                "completion {0} does not match the request".format(name))

    return {
        "warmup_records": warmup_records,
        "measured_records": measured_records,
        "predictions": _parse_predictions(stdout, count),
        "completion": completion,
    }


# Streams one file through SHA-256.
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Requires an exact file size before the file participates in a measurement.
def _require_size(path, expected_bytes, description):
    try:
        actual_bytes = os.path.getsize(path)
    except OSError as error:
        raise ValueError("{0} is unreadable: {1}".format(description, error))
    if actual_bytes != expected_bytes:
        raise ValueError(
            "{0} must contain exactly {1} bytes, found {2}".format(
                description, expected_bytes, actual_bytes))


# Reads and validates external int32 labels; the tool never receives them.
def _read_labels(path, count):
    _require_size(path, count * 4, "labels file")
    with open(path, "rb") as labels_file:
        content = labels_file.read()
    labels = struct.unpack("<{0}i".format(count), content)
    for index, label in enumerate(labels):
        if label < 0 or label >= CLASS_COUNT:
            raise ValueError("label {0} must be in [0, 10)".format(index))
    return labels


# Validates the exact four-line metadata grammar without decoding the model or
# recomputing its CRC32. The compiled tool and the later packager own those.
def _read_model_mode(model_dir):
    meta_path = os.path.join(model_dir, "course.meta")
    try:
        with open(meta_path, "rb") as meta_file:
            raw = meta_file.read(MAX_META_BYTES + 1)
    except OSError as error:
        raise ValueError("course.meta is unreadable: {0}".format(error))
    if len(raw) > MAX_META_BYTES:
        raise ValueError("course.meta must be at most 256 bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("course.meta must be UTF-8: {0}".format(error))
    if not text.endswith("\n"):
        raise ValueError("course.meta must be LF-terminated four lines")
    lines = text[:-1].split("\n")
    if len(lines) != 4:
        raise ValueError("course.meta must contain exactly four lines")
    if lines[0] != "course_lenet_v1":
        raise ValueError("course.meta header must equal course_lenet_v1")
    if not lines[1].startswith("input_mode="):
        raise ValueError("course.meta must contain one input_mode line")
    input_mode = lines[1][len("input_mode="):]
    if input_mode not in INPUT_MODES:
        raise ValueError(
            "course.meta input mode must be unit, byte, or normalized")
    if lines[2] != "shape=1,28,28,10":
        raise ValueError("course.meta shape must equal 1,28,28,10")
    if not lines[3].startswith("checkpoint_crc32="):
        raise ValueError(
            "course.meta must contain one checkpoint_crc32 line")
    if not CRC_PATTERN.match(lines[3][len("checkpoint_crc32="):]):
        raise ValueError(
            "course.meta checkpoint_crc32 must be eight lowercase hex digits")
    return input_mode


# Decodes captured process output without dropping any byte.
def _decode_output(content):
    if content is None:
        return ""
    if isinstance(content, bytes):
        return content.decode("utf-8", "replace")
    return content


# Writes the complete JSON with exclusive creation and a safe rollback.
def _write_json_exclusive(path, result):
    payload = json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False) + "\n"
    encoded = payload.encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            descriptor = None
            output_file.write(encoded)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.remove(path)
        except OSError:
            pass
        raise


# Returns a normalized path for output-alias comparisons.
def _path_identity(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


# Refuses an existing output or any alias of a protected input before launch.
def _check_output(output_json, protected_paths):
    if os.path.lexists(output_json):
        raise FileExistsError(
            "output JSON already exists: {0}".format(output_json))
    target = _path_identity(output_json)
    for path in protected_paths:
        if _path_identity(path) == target:
            raise FileExistsError(
                "output JSON aliases a protected input: {0}".format(path))


# Empty failure summary; every acceptance boolean is false.
def _failure_summary():
    return {
        "mean_ms": None,
        "median_ms": None,
        "min_ms": None,
        "max_ms": None,
        "correct": 0,
        "accuracy": 0.0,
        "accuracy_pass": False,
        "mean_below_100ms": False,
        "median_below_100ms": False,
    }


# Builds a result dict with exactly the pinned top-level schema.
def _result_record(warm_exe, infer_exe, images, labels, model_dir,
                   model_source, count, batch, input_mode, warmups, runs,
                   sha256, sha256_after, hashes_stable, returncode,
                   process_wall_ms, stdout, stderr, warmup_records,
                   measured_records, predictions_identical_all, status,
                   errors, summary):
    return {
        "schema": WARM_BENCHMARK_SCHEMA,
        "timing_scope": TIMING_SCOPE,
        "executable": warm_exe,
        "official_infer": infer_exe,
        "images_path": images,
        "labels_path": labels,
        "model_dir": model_dir,
        "model_source": model_source,
        "n": count,
        "batch": batch,
        "input_mode": input_mode,
        "warmups": warmups,
        "runs": runs,
        "sha256": sha256,
        "sha256_after": sha256_after,
        "hashes_stable": hashes_stable,
        "returncode": returncode,
        "process_wall_ms": process_wall_ms,
        "stdout": stdout,
        "stderr": stderr,
        "warmup_records": warmup_records,
        "measured_records": measured_records,
        "predictions_identical_all": predictions_identical_all,
        "status": status,
        "errors": errors,
        "summary": summary,
    }


# Computes the successful summary from measured rounds only, retaining every
# sample and excluding all warmups from the statistics.
def _success_summary(records, predictions, labels, count):
    durations = [record["wall_ms"] for record in records]
    correct = sum(
        1 for predicted, expected in zip(predictions, labels)
        if predicted == expected)
    accuracy = float(correct) / float(count)
    mean_ms = statistics.mean(durations)
    median_ms = statistics.median(durations)
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min(durations),
        "max_ms": max(durations),
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_pass": accuracy >= ACCURACY_THRESHOLD,
        "mean_below_100ms": mean_ms < FULL_REQUEST_LIMIT_MS,
        "median_below_100ms": median_ms < FULL_REQUEST_LIMIT_MS,
    }


# Runs one warm full-request tool process and collects raw observations.
def run_benchmark(warm_exe, infer_exe, images, labels, model_dir,
                  output_json, count, batch=DEFAULT_BATCH,
                  warmups=DEFAULT_WARMUPS, runs=DEFAULT_RUNS,
                  model_source="course_train"):
    count = _require_int(count, "count", 1)
    batch = _require_int(batch, "batch", 1)
    warmups = _require_int(warmups, "warmups", 1)
    runs = _require_int(runs, "runs", 1)
    if warmups > MAX_ROUND_COUNT or runs > MAX_ROUND_COUNT:
        raise ValueError("warmups and runs must be in [1, 100000]")
    if model_source not in MODEL_SOURCES:
        raise ValueError("model source must be legacy or course_train")

    meta_path = os.path.join(model_dir, "course.meta")
    model_path = os.path.join(model_dir, "lenet.bin")
    _check_output(output_json, (
        warm_exe, infer_exe, images, labels, meta_path, model_path))

    label_values = _read_labels(labels, count)
    _require_size(images, count * IMAGE_PIXELS * 4, "images file")
    input_mode = _read_model_mode(model_dir)

    sha256 = {
        "binary": _sha256_file(warm_exe),
        "official_infer": _sha256_file(infer_exe),
        "model": _sha256_file(model_path),
        "meta": _sha256_file(meta_path),
        "images": _sha256_file(images),
        "labels": _sha256_file(labels),
    }

    argv = [
        warm_exe, images, str(count), str(SHAPE[0]), str(SHAPE[1]),
        str(SHAPE[2]), str(SHAPE[3]), str(warmups), str(runs),
    ]
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith("COURSE_")
    }
    environment["COURSE_MODEL_DIR"] = model_dir
    environment["COURSE_INFER_BATCH"] = str(batch)

    failure = None
    stdout = ""
    stderr = ""
    returncode = None
    started = time.monotonic()
    try:
        child = subprocess.Popen(
            argv, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except OSError as error:
        process_wall_ms = (time.monotonic() - started) * 1000.0
        stderr = str(error)
        failure = "warm benchmark process could not start: {0}".format(error)
    else:
        try:
            output, diagnostics = child.communicate(
                timeout=PROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            # A child can close both pipes before it exits. The POSIX
            # communicate timeout then reports no exception payload even
            # though it already buffered the raw bytes. Kill and drain the
            # same child again instead of trusting or concatenating an
            # exception duplicate; this is the only timeout recovery.
            try:
                child.kill()
            except OSError:
                pass
            output, diagnostics = child.communicate()
            failure = (
                "warm benchmark process timed out after {0} seconds".format(
                    PROCESS_TIMEOUT_SECONDS))
        # Cleanup through child exit is part of the actual process wall.
        process_wall_ms = (time.monotonic() - started) * 1000.0
        returncode = child.returncode
        stdout = _decode_output(output)
        stderr = _decode_output(diagnostics)

    sha256_after = {}
    for name, path in (
            ("binary", warm_exe),
            ("official_infer", infer_exe),
            ("model", model_path),
            ("meta", meta_path),
            ("images", images),
            ("labels", labels)):
        try:
            sha256_after[name] = _sha256_file(path)
        except OSError:
            sha256_after[name] = None
    hashes_stable = sha256_after == sha256

    errors = []
    if failure is not None:
        errors.append(failure)
    parsed = None
    if not errors:
        try:
            parsed = parse_tool_output(
                stdout, stderr, returncode, count, batch, warmups, runs,
                input_mode)
        except ValueError as error:
            errors.append(str(error))
    if not hashes_stable:
        changed = [name for name in HASH_FIELDS
                   if sha256[name] != sha256_after[name]]
        errors.append(
            "after-execution sha256 differs for: {0}".format(
                ", ".join(changed)))

    if errors:
        result = _result_record(
            warm_exe, infer_exe, images, labels, model_dir, model_source,
            count, batch, input_mode, warmups, runs, sha256, sha256_after,
            hashes_stable, returncode, process_wall_ms, stdout, stderr,
            [], [], False, "fail", errors, _failure_summary())
    else:
        result = _result_record(
            warm_exe, infer_exe, images, labels, model_dir, model_source,
            count, batch, input_mode, warmups, runs, sha256, sha256_after,
            True, returncode, process_wall_ms, stdout, stderr,
            parsed["warmup_records"], parsed["measured_records"], True,
            "pass", [],
            _success_summary(
                parsed["measured_records"], parsed["predictions"],
                label_values, count))

    _write_json_exclusive(output_json, result)
    return result


# Parses the documented collector CLI and returns the acceptance exit status.
def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Collect warm full-request inference measurements")
    parser.add_argument("warm_exe")
    parser.add_argument("infer_exe")
    parser.add_argument("images")
    parser.add_argument("labels")
    parser.add_argument("model_dir")
    parser.add_argument("output_json")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--model-source", required=True,
                        choices=MODEL_SOURCES)
    options = parser.parse_args(arguments)

    try:
        result = run_benchmark(
            options.warm_exe, options.infer_exe, options.images,
            options.labels, options.model_dir, options.output_json,
            options.count, batch=options.batch, warmups=options.warmups,
            runs=options.runs, model_source=options.model_source)
    except (ValueError, OSError) as error:
        print("course_warm_benchmark: {0}".format(error), file=sys.stderr)
        return 1

    summary = result["summary"]
    accepted = (
        result["status"] == "pass" and
        summary["accuracy_pass"] and
        summary["mean_below_100ms"] and
        summary["median_below_100ms"])
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
