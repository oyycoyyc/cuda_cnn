from __future__ import print_function

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

from scripts import course_fixture

try:
    from unittest import mock
except ImportError:  # pragma: no cover - Python 3.6 includes unittest.mock
    import mock


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAGIC = b"MNISTC1\0"
HEADER = struct.Struct("<IIII")
IMAGE_PIXELS = 28 * 28
SOURCE_HEADER_BYTES = 24

# The manifest is a small, exact metadata record for one converted fixture.
MANIFEST_KEYS = {
    "schema",
    "source",
    "source_sha256",
    "n",
    "shape",
    "input_mode",
    "images",
    "labels",
}


# Rounds one Python float to its nearest binary32 value.
def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


# Computes the documented fixture value with binary32 intermediate rounding.
# The normalized mode mirrors the persisted inference domain
# (x / 255 - 0.1307) / 0.3081 without a uint8 roundtrip.
def expected_pixel(pixel, input_mode):
    unit = float32(float32(pixel) / float32(255.0))
    if input_mode == "unit":
        return unit
    if input_mode == "byte":
        return float32(pixel)
    if input_mode == "normalized":
        return float32(
            float32(unit - float32(0.1307)) / float32(0.3081))
    raise AssertionError("unsupported test input mode {0}".format(input_mode))


# Builds the exact MNISTC1 v1 layout: 24-byte header, all image bytes, then all
# label bytes. This is the layout produced by prepare_mnist.py and consumed by
# src/dataset.cpp; the body is deliberately not interleaved.
def dataset_bytes(images, labels, version=1, rows=28, columns=28,
                  count=None, magic=MAGIC):
    if count is None:
        count = len(images)
    return (magic + HEADER.pack(version, count, rows, columns) +
            b"".join(images) + bytes(bytearray(labels)))


# Returns the SHA-256 digest of one file for immutability checks.
def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class CourseFixtureTest(unittest.TestCase):
    # Every test owns an isolated source/output directory.
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = self.temporary.name

    def tearDown(self):
        self.temporary.cleanup()

    def path(self, name):
        return os.path.join(self.directory, name)

    def write_source(self, content, name="source.bin"):
        path = self.path(name)
        with open(path, "wb") as source_file:
            source_file.write(content)
        return path

    def output_path(self, name="output"):
        return self.path(name)

    def convert(self, source, output, input_mode):
        return course_fixture.convert_mnistc1(source, output, input_mode)

    def read_manifest(self, output):
        with open(os.path.join(output, "manifest.json"), "r") as manifest_file:
            return json.load(manifest_file)

    # A rejected conversion must not leave any fixture artifact behind.
    def assert_no_artifacts(self, output):
        for name in ("images.f32", "labels.i32", "manifest.json"):
            self.assertFalse(
                os.path.exists(os.path.join(output, name)),
                "unexpected {0} in {1}".format(name, output))
        if os.path.isdir(output):
            self.assertEqual([], os.listdir(output))

    # Runs the documented CLI in a stdlib-only interpreter. The source and
    # output paths are caller-owned temporary files, never a mounted dataset.
    def run_cli(self, arguments):
        return subprocess.run(
            [sys.executable, "-S", "-m", "scripts.course_fixture"] +
            list(arguments),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    # Unit, byte, and normalized modes all write little-endian binary32 values.
    def test_all_modes_write_exact_little_endian_pixels_and_labels(self):
        endpoint_pixels = bytes(bytearray(
            [0, 1, 2, 127, 128, 129, 254, 255] +
            [0] * (IMAGE_PIXELS - 8)))
        all_255 = bytes(bytearray([255]) * IMAGE_PIXELS)
        all_zero = bytes(bytearray(IMAGE_PIXELS))
        labels = [0, 7, 9]
        source = self.write_source(
            dataset_bytes([endpoint_pixels, all_255, all_zero], labels))
        source_hash = file_digest(source)

        for input_mode in ("unit", "byte", "normalized"):
            with self.subTest(input_mode=input_mode):
                output = self.output_path("output-" + input_mode)
                self.convert(source, output, input_mode)

                self.assertEqual(
                    ["images.f32", "labels.i32", "manifest.json"],
                    sorted(os.listdir(output)))

                with open(os.path.join(output, "images.f32"), "rb") as images:
                    image_bytes = images.read()
                self.assertEqual(3 * IMAGE_PIXELS * 4, len(image_bytes))
                values = struct.unpack("<2352f", image_bytes)
                for index, pixel in enumerate(
                        [0, 1, 2, 127, 128, 129, 254, 255]):
                    self.assertEqual(expected_pixel(pixel, input_mode),
                                     values[index])
                for index in range(IMAGE_PIXELS):
                    self.assertEqual(expected_pixel(255, input_mode),
                                     values[IMAGE_PIXELS + index])
                    self.assertEqual(expected_pixel(0, input_mode),
                                     values[2 * IMAGE_PIXELS + index])

                with open(os.path.join(output, "labels.i32"), "rb") as labels_file:
                    label_bytes = labels_file.read()
                self.assertEqual(3 * 4, len(label_bytes))
                self.assertEqual(tuple(labels), struct.unpack("<3i", label_bytes))

                manifest = self.read_manifest(output)
                self.assertEqual(input_mode, manifest["input_mode"])

                # The source remains byte-for-byte identical across modes.
                self.assertEqual(source_hash, file_digest(source))

        # Mode-specific endpoints prove the converter does not quantize.
        with open(os.path.join(self.output_path("output-unit"), "images.f32"),
                  "rb") as images:
            unit_values = struct.unpack("<2352f", images.read())
        self.assertEqual(float32(1.0 / 255.0), unit_values[1])
        self.assertNotEqual(float32(0.5), unit_values[4])
        self.assertNotEqual(float32(128.0), unit_values[4])

        with open(os.path.join(self.output_path("output-byte"), "images.f32"),
                  "rb") as images:
            byte_values = struct.unpack("<2352f", images.read())
        self.assertEqual(0.0, byte_values[0])
        self.assertEqual(255.0, byte_values[IMAGE_PIXELS])

        with open(
                os.path.join(self.output_path("output-normalized"), "images.f32"),
                "rb") as images:
            normalized_values = struct.unpack("<2352f", images.read())
        self.assertLess(normalized_values[0], 0.0)
        self.assertGreater(normalized_values[IMAGE_PIXELS], 1.0)
        self.assertEqual(expected_pixel(128, "normalized"),
                         normalized_values[4])

    # Chunk boundaries do not change values, labels, or the source digest.
    def test_conversion_streams_across_small_chunk_boundaries(self):
        image = bytes(bytearray(
            [index % 256 for index in range(IMAGE_PIXELS)]))
        labels = [0, 9, 3, 7]
        source = self.write_source(
            dataset_bytes([image, image, image, image], labels))
        source_hash = file_digest(source)
        output = self.output_path("chunked")

        with mock.patch.object(course_fixture, "IMAGE_CHUNK_BYTES", 100), \
                mock.patch.object(course_fixture, "LABEL_CHUNK_BYTES", 1):
            self.convert(source, output, "byte")

        with open(os.path.join(output, "images.f32"), "rb") as images:
            values = struct.unpack("<3136f", images.read())
        for image_index in range(4):
            for pixel in range(IMAGE_PIXELS):
                self.assertEqual(float(pixel % 256),
                                 values[image_index * IMAGE_PIXELS + pixel])
        with open(os.path.join(output, "labels.i32"), "rb") as labels_file:
            self.assertEqual(tuple(labels),
                             struct.unpack("<4i", labels_file.read()))
        self.assertEqual(source_hash, self.read_manifest(output)[
            "source_sha256"])

    # The label block follows every image byte, not one label per image row.
    def test_labels_are_read_from_the_trailing_block(self):
        image = bytes(bytearray([9]) * IMAGE_PIXELS)
        labels = [3, 4]
        source = self.write_source(dataset_bytes([image, image], labels))
        output = self.output_path("block-layout")

        self.convert(source, output, "byte")

        with open(os.path.join(output, "images.f32"), "rb") as images:
            values = struct.unpack("<1568f", images.read())
        for index in range(2 * IMAGE_PIXELS):
            self.assertEqual(9.0, values[index])
        with open(os.path.join(output, "labels.i32"), "rb") as labels_file:
            self.assertEqual((3, 4), struct.unpack("<2i", labels_file.read()))

    # The manifest records the exact count, fixed shape, mode, and source hash.
    def test_manifest_records_exact_fixture_metadata(self):
        image = bytes(bytearray([0, 255]) + bytearray(IMAGE_PIXELS - 2))
        source = self.write_source(dataset_bytes([image], [5]))
        output = self.output_path("manifest")

        self.convert(source, output, "normalized")

        manifest = self.read_manifest(output)
        self.assertEqual(MANIFEST_KEYS, set(manifest))
        self.assertEqual("course-fixture-v1", manifest["schema"])
        self.assertEqual(os.path.abspath(source), manifest["source"])
        self.assertEqual(file_digest(source), manifest["source_sha256"])
        self.assertEqual(1, manifest["n"])
        self.assertEqual([1, 28, 28, 10], manifest["shape"])
        self.assertEqual("normalized", manifest["input_mode"])
        self.assertEqual("images.f32", manifest["images"])
        self.assertEqual("labels.i32", manifest["labels"])

    # Malformed magic, header fields, counts, and sizes are all rejected.
    def test_rejects_invalid_headers_and_exact_size_mismatches(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        valid = dataset_bytes([zero], [0])
        header = valid[:SOURCE_HEADER_BYTES]
        image_bytes = valid[SOURCE_HEADER_BYTES:SOURCE_HEADER_BYTES +
                            IMAGE_PIXELS]
        label_bytes = valid[-1:]
        cases = [
            ("short-header", MAGIC + b"\x01\x00"),
            ("bad-magic", b"X" + valid[1:]),
            ("bad-version", dataset_bytes([zero], [0], version=2)),
            ("zero-count", dataset_bytes([], [], count=0)),
            ("bad-rows", dataset_bytes([zero], [0], rows=27)),
            ("bad-columns", dataset_bytes([zero], [0], columns=29)),
            ("truncated-image", header + image_bytes[:700] + label_bytes),
            ("truncated-labels", header + image_bytes),
            ("trailing-byte", valid + b"\x00"),
            ("count-mismatch", dataset_bytes([zero], [0], count=2)),
        ]
        for name, content in cases:
            with self.subTest(case=name):
                source = self.write_source(content, name=name + ".bin")
                output = self.output_path("reject-" + name)
                with self.assertRaises(ValueError):
                    self.convert(source, output, "unit")
                self.assert_no_artifacts(output)

    # Labels are uint8 class IDs; only 0 through 9 are valid.
    def test_rejects_labels_outside_the_decimal_class_range(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        for label in (10, 11, 255):
            with self.subTest(label=label):
                source = self.write_source(
                    dataset_bytes([zero], [label]),
                    name="label-{0}.bin".format(label))
                output = self.output_path("label-{0}".format(label))
                with self.assertRaises(ValueError):
                    self.convert(source, output, "byte")
                self.assert_no_artifacts(output)

    # Existing output directories are never reused or mutated.
    def test_refuses_existing_output_directory_and_preserves_contents(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        source = self.write_source(dataset_bytes([zero], [0]))
        source_hash = file_digest(source)

        for name, contents in (("empty", {}),
                               ("occupied", {"keep.txt": b"keep me\n"})):
            with self.subTest(output=name):
                output = self.output_path("existing-" + name)
                os.makedirs(output)
                for child_name, child_bytes in contents.items():
                    with open(os.path.join(output, child_name), "wb") as child:
                        child.write(child_bytes)
                before = sorted(os.listdir(output))

                with self.assertRaises(FileExistsError):
                    self.convert(source, output, "unit")

                self.assertEqual(before, sorted(os.listdir(output)))
                for child_name, child_bytes in contents.items():
                    with open(os.path.join(output, child_name), "rb") as child:
                        self.assertEqual(child_bytes, child.read())
                self.assertEqual(source_hash, file_digest(source))

    # The output directory may not be the source file or its parent.
    def test_refuses_output_aliases_of_the_source(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        source = self.write_source(dataset_bytes([zero], [0]))
        source_hash = file_digest(source)

        for output in (source, self.directory):
            with self.subTest(output=output):
                with self.assertRaises(FileExistsError):
                    self.convert(source, output, "unit")
                self.assertEqual(source_hash, file_digest(source))

    # A symlink alias of an existing directory is not a fresh output directory.
    def test_refuses_symlinked_output_directory(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        source = self.write_source(dataset_bytes([zero], [0]))
        source_hash = file_digest(source)
        real_output = self.output_path("real-output")
        os.makedirs(real_output)
        alias = self.output_path("alias-output")
        try:
            os.symlink(real_output, alias, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlinks are unavailable on this host")

        with self.assertRaises(FileExistsError):
            self.convert(source, alias, "unit")

        self.assertEqual([], os.listdir(real_output))
        self.assertEqual(source_hash, file_digest(source))

    # A read-only source reached through a symlink is read, never rewritten.
    def test_conversion_through_source_symlink_keeps_target_intact(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        target = self.write_source(dataset_bytes([zero], [0]), name="target.bin")
        target_hash = file_digest(target)
        alias = self.path("source-alias.bin")
        try:
            os.symlink(target, alias)
        except (OSError, NotImplementedError):
            self.skipTest("file symlinks are unavailable on this host")
        output = self.output_path("symlink-source")

        self.convert(alias, output, "unit")

        self.assertTrue(os.path.islink(alias))
        self.assertEqual(target_hash, file_digest(target))
        self.assertEqual(["images.f32", "labels.i32", "manifest.json"],
                         sorted(os.listdir(output)))

    # The documented CLI converts a caller-owned source in a stdlib-only run.
    def test_cli_converts_an_explicit_mode_without_third_party_packages(self):
        image = bytes(bytearray([0, 128, 255]) + bytearray(IMAGE_PIXELS - 3))
        source = self.write_source(dataset_bytes([image], [8]))
        output = self.output_path("cli-success")

        result = self.run_cli([source, output, "--input-mode", "byte"])

        self.assertEqual(0, result.returncode, result.stderr)
        manifest = self.read_manifest(output)
        self.assertEqual("byte", manifest["input_mode"])
        self.assertEqual(1, manifest["n"])

    # Usage and conversion failures exit nonzero and leave no artifacts.
    def test_cli_reports_usage_and_conversion_errors_without_artifacts(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        source = self.write_source(dataset_bytes([zero], [0]))
        bad_source = self.write_source(b"not a dataset", name="bad-source.bin")
        cases = [
            ("missing-mode", [source, self.output_path("cli-missing-mode")]),
            ("unknown-mode", [source, self.output_path("cli-unknown-mode"),
                              "--input-mode", "Unit"]),
            ("bad-source", [bad_source, self.output_path("cli-bad-source"),
                            "--input-mode", "unit"]),
            ("extra-argument", [source, self.output_path("cli-extra"),
                                "--input-mode", "unit", "extra"]),
        ]
        for name, arguments in cases:
            with self.subTest(case=name):
                result = self.run_cli(arguments)
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertTrue(result.stderr)
                self.assert_no_artifacts(arguments[1])

        existing = self.output_path("cli-existing")
        os.makedirs(existing)
        sentinel = os.path.join(existing, "keep.txt")
        with open(sentinel, "wb") as keep:
            keep.write(b"keep\n")
        result = self.run_cli([source, existing, "--input-mode", "unit"])
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual(["keep.txt"], sorted(os.listdir(existing)))
        with open(sentinel, "rb") as keep:
            self.assertEqual(b"keep\n", keep.read())

    # main() is the direct CLI entry point and returns zero on success.
    def test_main_returns_zero_for_a_fresh_output_directory(self):
        zero = bytes(bytearray(IMAGE_PIXELS))
        source = self.write_source(dataset_bytes([zero], [4]))
        output = self.output_path("main-success")

        self.assertEqual(0, course_fixture.main(
            [source, output, "--input-mode", "normalized"]))

        self.assertEqual("normalized", self.read_manifest(output)["input_mode"])


if __name__ == "__main__":
    unittest.main()
