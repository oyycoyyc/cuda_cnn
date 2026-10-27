from __future__ import print_function

import argparse
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import time


# Pinned external benchmark schema and acceptance constants.
BENCHMARK_SCHEMA = "course-benchmark-v1"
ACCURACY_THRESHOLD = 0.96
DEFAULT_BATCH = 1024
DEFAULT_RUNS = 5
FULL_WALL_LIMIT_MS = 100.0
MODEL_SOURCES = ("legacy", "course_train")
INPUT_MODES = ("unit", "byte", "normalized")
IMAGE_PIXELS = 28 * 28
CLASS_COUNT = 10
SHAPE = (1, 28, 28, 10)

# These inherited settings must never leak into an external inference run.
CLEARED_ENVIRONMENT = ("COURSE_PROFILE", "COURSE_INPUT_MODE", "COURSE_EPOCHS")


# Hashes one file in bounded blocks.
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Requires an exact file size before the file participates in a measurement.
def _require_size(path, expected_bytes, description):
    try:
        actual_bytes = os.path.getsize(path)
    except OSError as error:
        raise ValueError("{0} is unreadable: {1}".format(
            description, error))
    if actual_bytes != expected_bytes:
        raise ValueError(
            "{0} must contain exactly {1} bytes, found {2}".format(
                description, expected_bytes, actual_bytes))


# Reads and validates the external int32 labels; inference never sees them.
def _read_labels(path, count):
    expected_bytes = count * 4
    _require_size(path, expected_bytes, "labels file")
    with open(path, "rb") as labels_file:
        content = labels_file.read()
    labels = struct.unpack("<{0}i".format(count), content)
    for index, label in enumerate(labels):
        if label < 0 or label >= CLASS_COUNT:
            raise ValueError(
                "label {0} must be in [0, 10)".format(index))
    return labels


# Reads the persisted input mode from the model bundle. Full model validity is
# deliberately left to the real compiled inference loader.
def _read_model_mode(model_dir):
    meta_path = os.path.join(model_dir, "course.meta")
    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            lines = [line.rstrip("\r\n") for line in meta_file]
    except OSError as error:
        raise ValueError("course.meta is unreadable: {0}".format(error))
    if not lines or lines[0] != "course_lenet_v1":
        raise ValueError("course.meta header must equal course_lenet_v1")

    modes = [line.split("=", 1)[1] for line in lines
             if line.startswith("input_mode=")]
    if len(modes) != 1:
        raise ValueError("course.meta must contain one input_mode line")
    if modes[0] not in INPUT_MODES:
        raise ValueError("course.meta input mode must be unit, byte, or normalized")

    shapes = [line.split("=", 1)[1] for line in lines
              if line.startswith("shape=")]
    if len(shapes) != 1 or shapes[0] != "1,28,28,10":
        raise ValueError("course.meta shape must equal 1,28,28,10")
    return modes[0]


# Reads the optional reference as exactly N uint8 predictions, never labels.
def _read_reference(path, count):
    with open(path, "rb") as reference_file:
        content = reference_file.read()
    if len(content) != count:
        raise ValueError(
            "reference predictions must contain exactly {0} bytes".format(
                count))
    for index, prediction in enumerate(bytearray(content)):
        if prediction >= CLASS_COUNT:
            raise ValueError(
                "reference prediction {0} must be in [0, 10)".format(index))
    return content


# Requires a positive decimal integer that cannot be a bool.
def _require_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("{0} must be a positive integer".format(name))
    return value


# Parses exactly N decimal class-integer lines from inference stdout.
def _parse_predictions(stdout, count):
    lines = stdout.splitlines()
    if len(lines) != count:
        return None
    predictions = []
    for line in lines:
        if not line or any(character < "0" or character > "9"
                           for character in line):
            return None
        value = int(line)
        if value < 0 or value >= CLASS_COUNT:
            return None
        predictions.append(value)
    return predictions


# Decodes captured process output as UTF-8 without discarding any content.
def _decode(stream):
    return stream.decode("utf-8", "replace")


# Writes the result JSON with exclusive creation; an existing target is never
# replaced, including through a symlink alias.
def _write_json_exclusive(path, result):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            descriptor = None
            json.dump(result, output_file, indent=2, sort_keys=True,
                      ensure_ascii=False)
            output_file.write("\n")
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.remove(path)
        except OSError:
            pass
        raise


# Runs one benchmark artifact. Every requested run is executed and retained;
# profiled calls simply set COURSE_PROFILE=1 for all of those runs.
def run_benchmark(infer_path, images_path, labels_path, model_dir,
                  output_json, count, batch=DEFAULT_BATCH,
                  runs=DEFAULT_RUNS, model_source="legacy", profile=False,
                  reference_predictions=None):
    count = _require_positive_int(count, "count")
    runs = _require_positive_int(runs, "runs")
    _require_positive_int(batch, "batch")
    if batch > 16384:
        raise ValueError("batch must be in [1, 16384]")
    if model_source not in MODEL_SOURCES:
        raise ValueError("model source must be legacy or course_train")
    if os.path.lexists(output_json):
        raise FileExistsError(
            "output JSON already exists: {0}".format(output_json))

    labels = _read_labels(labels_path, count)
    _require_size(images_path, count * IMAGE_PIXELS * 4, "images file")
    input_mode = _read_model_mode(model_dir)

    meta_path = os.path.join(model_dir, "course.meta")
    model_path = os.path.join(model_dir, "lenet.bin")
    sha256 = {
        "binary": _sha256_file(infer_path),
        "model": _sha256_file(model_path),
        "meta": _sha256_file(meta_path),
        "images": _sha256_file(images_path),
        "labels": _sha256_file(labels_path),
    }

    reference = None
    reference_sha256 = None
    if reference_predictions is not None:
        reference = _read_reference(reference_predictions, count)
        reference_sha256 = _sha256_file(reference_predictions)

    argv = [infer_path, images_path, str(count), str(SHAPE[0]),
            str(SHAPE[1]), str(SHAPE[2]), str(SHAPE[3])]
    environment = os.environ.copy()
    for name in CLEARED_ENVIRONMENT:
        environment.pop(name, None)
    environment["COURSE_MODEL_DIR"] = model_dir
    environment["COURSE_INFER_BATCH"] = str(batch)
    if profile:
        environment["COURSE_PROFILE"] = "1"

    run_records = []
    run_predictions = []
    for _ in range(runs):
        start = time.monotonic()
        try:
            completed = subprocess.run(
                argv, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wall_ms = (time.monotonic() - start) * 1000.0
            returncode = completed.returncode
            stdout = _decode(completed.stdout)
            stderr = _decode(completed.stderr)
        except OSError as error:
            wall_ms = (time.monotonic() - start) * 1000.0
            returncode = -1
            stdout = ""
            stderr = str(error)

        predictions = (_parse_predictions(stdout, count)
                       if returncode == 0 else None)
        if predictions is None:
            correct = 0
        else:
            correct = sum(1 for predicted, expected in zip(predictions, labels)
                          if predicted == expected)
        run_records.append({
            "wall_ms": wall_ms,
            "returncode": returncode,
            "correct": correct,
            "stdout": stdout,
            "stderr": stderr,
        })
        run_predictions.append(predictions)

    all_valid = all(record["returncode"] == 0 and predictions is not None
                    for record, predictions in zip(run_records,
                                                   run_predictions))
    identical = all_valid and all(
        predictions == run_predictions[0]
        for predictions in run_predictions[1:])

    if identical:
        common_predictions = run_predictions[0]
        correct = sum(1 for predicted, expected in zip(common_predictions,
                                                       labels)
                      if predicted == expected)
        accuracy = float(correct) / float(count)
        accuracy_pass = accuracy >= ACCURACY_THRESHOLD
    else:
        correct = 0
        accuracy = 0.0
        accuracy_pass = False

    walls = [record["wall_ms"] for record in run_records]
    summary = {
        "median_ms": statistics.median(walls),
        "max_ms": max(walls),
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_pass": accuracy_pass,
        "all_wall_below_100ms": all(
            wall < FULL_WALL_LIMIT_MS for wall in walls),
    }

    result = {
        "schema": BENCHMARK_SCHEMA,
        "executable": infer_path,
        "images_path": images_path,
        "labels_path": labels_path,
        "model_dir": model_dir,
        "model_source": model_source,
        "n": count,
        "batch": batch,
        "input_mode": input_mode,
        "profiled": bool(profile),
        "runs_requested": runs,
        "sha256": sha256,
        "runs": run_records,
        "summary": summary,
    }

    if reference is not None:
        result["reference_sha256"] = reference_sha256
        if identical:
            result["reference_mismatches"] = sum(
                1 for predicted, expected in zip(common_predictions,
                                                 bytearray(reference))
                if predicted != expected)
        else:
            result["reference_mismatches"] = count

    _write_json_exclusive(output_json, result)
    return result


# Parses the documented benchmark CLI and returns its acceptance exit status.
def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Run external full-process inference measurements")
    parser.add_argument("infer")
    parser.add_argument("images")
    parser.add_argument("labels")
    parser.add_argument("model_dir")
    parser.add_argument("output_json")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--model-source", required=True,
                        choices=MODEL_SOURCES)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--reference-predictions")
    options = parser.parse_args(arguments)

    try:
        result = run_benchmark(
            options.infer, options.images, options.labels, options.model_dir,
            options.output_json, options.count, batch=options.batch,
            runs=options.runs, model_source=options.model_source,
            profile=options.profile,
            reference_predictions=options.reference_predictions)
    except (ValueError, OSError) as error:
        print("course_benchmark: {0}".format(error), file=sys.stderr)
        return 1

    passed = result["summary"]["accuracy_pass"]
    if result.get("reference_mismatches", 0) != 0:
        passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
