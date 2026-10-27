from __future__ import print_function

import argparse
import hashlib
import json
import os
import struct
import sys


# Exact MNISTC1 v1 layout: a 24-byte header, every image byte, then every label.
MAGIC = b"MNISTC1\0"
VERSION = 1
ROWS = 28
COLUMNS = 28
HEADER_BYTES = 24
IMAGE_PIXELS = ROWS * COLUMNS
BYTES_PER_SAMPLE = IMAGE_PIXELS + 1
INPUT_MODES = ("unit", "byte", "normalized")
MANIFEST_SCHEMA = "course-fixture-v1"

# Bounded conversion chunks keep memory independent of the dataset count.
IMAGE_CHUNK_BYTES = 64 * 1024
LABEL_CHUNK_BYTES = 64 * 1024

# Models are trained with binary32 mean/std constants; round each staged value
# explicitly so the normalized fixture matches that persisted domain.
F32_MEAN = 0.1307
F32_STD = 0.3081


# Rounds one Python float to its nearest binary32 value.
def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


F32_MEAN = _float32(F32_MEAN)
F32_STD = _float32(F32_STD)
F32_255 = _float32(255.0)


# Converts one uint8 pixel to the requested raw input mode.
def _pixel_value(pixel, input_mode):
    unit = _float32(_float32(pixel) / F32_255)
    if input_mode == "unit":
        return unit
    if input_mode == "byte":
        return _float32(pixel)
    if input_mode == "normalized":
        return _float32(_float32(unit - F32_MEAN) / F32_STD)
    raise ValueError("input mode must be exactly unit, byte, or normalized")


# Precomputes every possible uint8 pixel once per supported mode.
PIXEL_TABLES = {
    mode: tuple(_pixel_value(pixel, mode) for pixel in range(256))
    for mode in INPUT_MODES
}


# Reads exactly size bytes or rejects a truncated source.
def _read_exact(stream, size, description):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("source is truncated: {0}".format(description))
    return data


# Removes only the artifacts this converter created in an exclusively new dir.
def _remove_owned_output(directory):
    for name in ("images.f32", "labels.i32", "manifest.json"):
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            pass
    try:
        os.rmdir(directory)
    except OSError:
        pass


# Converts one MNISTC1 v1 file into raw little-endian FP32/INT32 arrays plus an
# exact manifest. The source is opened read-only and never modified.
def convert_mnistc1(source_path, output_dir, input_mode):
    if input_mode not in INPUT_MODES:
        raise ValueError("input mode must be exactly unit, byte, or normalized")
    if not os.path.isfile(source_path):
        raise ValueError("source must be a readable file: {0}".format(
            source_path))
    if os.path.lexists(output_dir):
        raise FileExistsError(
            "output directory already exists: {0}".format(output_dir))

    source_absolute = os.path.abspath(source_path)
    table = PIXEL_TABLES[input_mode]
    digest = hashlib.sha256()

    with open(source_path, "rb") as source:
        header = _read_exact(source, HEADER_BYTES, "header")
        digest.update(header)
        if header[:8] != MAGIC:
            raise ValueError("source magic must equal MNISTC1\\0")
        version, count, rows, columns = struct.unpack("<IIII", header[8:])
        if version != VERSION:
            raise ValueError("source version must equal 1")
        if count == 0:
            raise ValueError("source sample count must be positive")
        if rows != ROWS or columns != COLUMNS:
            raise ValueError("source shape must be 28x28")
        expected_size = HEADER_BYTES + count * BYTES_PER_SAMPLE
        if os.fstat(source.fileno()).st_size != expected_size:
            raise ValueError(
                "source size must equal header plus count * 785 bytes")

        created = False
        try:
            os.makedirs(output_dir)
            created = True
            images_path = os.path.join(output_dir, "images.f32")
            labels_path = os.path.join(output_dir, "labels.i32")
            manifest_path = os.path.join(output_dir, "manifest.json")

            with open(images_path, "wb") as images_file, \
                    open(labels_path, "wb") as labels_file:
                remaining = count * IMAGE_PIXELS
                while remaining:
                    chunk = source.read(min(IMAGE_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError(
                            "source is truncated: image payload")
                    digest.update(chunk)
                    values = [table[pixel] for pixel in bytearray(chunk)]
                    images_file.write(struct.pack(
                        "<{0}f".format(len(values)), *values))
                    remaining -= len(chunk)

                remaining = count
                label_struct = struct.Struct("<i")
                while remaining:
                    chunk = source.read(min(LABEL_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise ValueError(
                            "source is truncated: label payload")
                    digest.update(chunk)
                    packed_labels = bytearray()
                    for label in bytearray(chunk):
                        if label > 9:
                            raise ValueError(
                                "label values must be in [0, 9]")
                        packed_labels.extend(label_struct.pack(label))
                    labels_file.write(packed_labels)
                    remaining -= len(chunk)

                if source.read(1):
                    raise ValueError("source must contain no trailing bytes")
                images_file.flush()
                labels_file.flush()

            manifest = {
                "schema": MANIFEST_SCHEMA,
                "source": source_absolute,
                "source_sha256": digest.hexdigest(),
                "n": count,
                "shape": [1, ROWS, COLUMNS, 10],
                "input_mode": input_mode,
                "images": "images.f32",
                "labels": "labels.i32",
            }
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(manifest, manifest_file, indent=2, sort_keys=True,
                          ensure_ascii=False)
                manifest_file.write("\n")
            return manifest
        except Exception:
            if created:
                _remove_owned_output(output_dir)
            raise


# Parses the documented fixture CLI and maps conversion failures to exit 1.
def main(arguments=None):
    parser = argparse.ArgumentParser(
        description="Convert a validated MNISTC1 v1 file into raw arrays")
    parser.add_argument("source")
    parser.add_argument("output_dir")
    parser.add_argument("--input-mode", required=True, choices=INPUT_MODES)
    options = parser.parse_args(arguments)

    try:
        convert_mnistc1(options.source, options.output_dir,
                        options.input_mode)
    except (ValueError, OSError) as error:
        print("course_fixture: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
