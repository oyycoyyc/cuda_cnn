from __future__ import print_function

import hashlib
import io
import os
import struct
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts import prepare_mnist

try:
    from unittest import mock
except ImportError:  # pragma: no cover - Python 3.6 includes unittest.mock
    import mock


# Encodes an in-memory grayscale PNG for Parquet image payloads.
def encoded_image(mode="L", size=(28, 28), value=0):
    image = Image.new(mode, size, value)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class PrepareMnistTest(unittest.TestCase):
    # Each test owns an isolated temporary directory for inputs and outputs.
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = self.temporary_directory.name

    def tearDown(self):
        self.temporary_directory.cleanup()

    # Resolves a Parquet fixture path inside the temporary directory.
    def parquet_path(self, name="input.parquet"):
        return os.path.join(self.directory, name)

    # Resolves a binary output path inside the temporary directory.
    def output_path(self, name="output.bin"):
        return os.path.join(self.directory, name)

    # Writes a Parquet fixture, optionally with an explicit image column type.
    def write_table(self, images, labels, image_type=None, name="input.parquet"):
        if image_type is None:
            table = pa.table({"image": images, "label": labels})
        else:
            table = pa.table({
                "image": pa.array(images, type=image_type),
                "label": labels,
            })
        path = self.parquet_path(name)
        pq.write_table(table, path)
        return path

    # Asserts conversion rejects a fixture and leaves no partial output behind.
    def assert_conversion_fails(self, images, labels, image_type=None):
        parquet_path = self.write_table(images, labels, image_type=image_type)
        output_path = self.output_path()
        with self.assertRaises(ValueError):
            prepare_mnist.convert_parquet(parquet_path, output_path, len(labels))
        self.assertFalse(os.path.exists(output_path))

    # Converts the Hugging Face image struct into the exact MNISTC1 layout.
    def test_converts_hugging_face_image_struct_to_exact_binary_layout(self):
        image0 = bytes(bytearray([0]) * 784)
        image1 = bytes(bytearray([127]) * 784)
        image2 = bytes(bytearray(range(256)) * 3 + bytearray(range(16)))
        image_type = pa.struct([pa.field("bytes", pa.binary()),
                                pa.field("path", pa.string())])
        rows = [
            {"bytes": encoded_image(value=0), "path": None},
            {"bytes": encoded_image(value=127), "path": None},
            {"bytes": self._encoded_pixels(image2), "path": None},
        ]
        parquet_path = self.write_table(rows, [0, 5, 9], image_type=image_type)
        output_path = self.output_path()

        prepare_mnist.convert_parquet(parquet_path, output_path, 3)

        expected = (b"MNISTC1\0" + struct.pack("<IIII", 1, 3, 28, 28) +
                    image0 + image1 + image2 + bytes(bytearray([0, 5, 9])))
        with open(output_path, "rb") as converted:
            self.assertEqual(expected, converted.read())

    # Accepts a synthetic raw-binary image column.
    def test_converts_synthetic_binary_image_column(self):
        pixels = bytes(bytearray([42]) * 784)
        parquet_path = self.write_table([self._encoded_pixels(pixels)], [4],
                                        image_type=pa.binary())
        output_path = self.output_path()

        prepare_mnist.convert_parquet(parquet_path, output_path, 1)

        with open(output_path, "rb") as converted:
            self.assertEqual(pixels, converted.read()[24:24 + 784])

    # Conversion rejection: non-grayscale and wrongly sized images.
    def test_rejects_non_grayscale_and_wrong_sized_images(self):
        cases = [
            (encoded_image(mode="RGB"), "RGB image"),
            (encoded_image(size=(27, 28)), "wrong width"),
            (encoded_image(size=(28, 27)), "wrong height"),
        ]
        for index, (image_bytes, description) in enumerate(cases):
            with self.subTest(description=description):
                self.assert_conversion_fails([image_bytes], [0],
                                             image_type=pa.binary())

    def test_rejects_missing_required_columns(self):
        for columns in ({"label": [0]}, {"image": [encoded_image()]}):
            path = self.parquet_path("missing-%d.parquet" % len(columns))
            pq.write_table(pa.table(columns), path)
            with self.assertRaises(ValueError):
                prepare_mnist.convert_parquet(path, self.output_path(), 1)

    def test_rejects_null_image_data(self):
        image_type = pa.struct([pa.field("bytes", pa.binary()),
                                pa.field("path", pa.string())])
        self.assert_conversion_fails(
            [{"bytes": None, "path": None}], [0], image_type=image_type)

    def test_rejects_malformed_encoded_image(self):
        self.assert_conversion_fails([b"not an image"], [0],
                                     image_type=pa.binary())

    def test_rejects_labels_outside_decimal_digit_range(self):
        for label in (-1, 10):
            with self.subTest(label=label):
                self.assert_conversion_fails([encoded_image()], [label],
                                             image_type=pa.binary())

    def test_rejects_zero_and_mismatched_expected_counts(self):
        empty_path = self.write_table([], [], image_type=pa.binary(),
                                      name="empty.parquet")
        with self.assertRaises(ValueError):
            prepare_mnist.convert_parquet(empty_path, self.output_path(), 0)

        one_row_path = self.write_table([encoded_image()], [0],
                                        image_type=pa.binary(), name="one.parquet")
        with self.assertRaises(ValueError):
            prepare_mnist.convert_parquet(one_row_path, self.output_path(), 2)

    # A cached-file checksum mismatch stops before the Parquet is opened.
    def test_checksum_mismatch_stops_before_parquet_open(self):
        parquet_path = self.parquet_path()
        with open(parquet_path, "wb") as cached:
            cached.write(b"corrupt cache")
        output_path = self.output_path()

        with mock.patch.object(prepare_mnist.pq, "read_table",
                               side_effect=AssertionError("Parquet opened")) as read_table:
            with self.assertRaises(ValueError):
                prepare_mnist.prepare_split(
                    "https://example.invalid/input.parquet", parquet_path,
                    output_path, hashlib.sha256(b"expected").hexdigest(), 1)

        read_table.assert_not_called()
        self.assertFalse(os.path.exists(output_path))

    # Every cached download use is rehashed, never trusted from the cache.
    def test_cached_download_is_rehashed_on_every_use(self):
        destination = self.parquet_path()
        good_content = b"verified parquet bytes"
        expected_hash = hashlib.sha256(good_content).hexdigest()
        with open(destination, "wb") as cached:
            cached.write(good_content)

        with mock.patch.object(prepare_mnist.requests, "get",
                               side_effect=AssertionError("network used")):
            self.assertEqual(destination, prepare_mnist.download_verified(
                "https://example.invalid/data", destination, expected_hash))
            with open(destination, "wb") as cached:
                cached.write(b"changed after first verification")
            with self.assertRaises(ValueError):
                prepare_mnist.download_verified(
                    "https://example.invalid/data", destination, expected_hash)

    # A failed download leaves neither destination nor temporary part file.
    def test_failed_download_leaves_no_destination_or_temporary_file(self):
        destination = self.parquet_path()
        response = FakeResponse([b"wrong ", b"content"])
        with mock.patch.object(prepare_mnist.requests, "get", return_value=response):
            with self.assertRaises(ValueError):
                prepare_mnist.download_verified(
                    "https://example.invalid/data", destination,
                    hashlib.sha256(b"expected").hexdigest())

        self.assertFalse(os.path.exists(destination))
        self.assertEqual([], [name for name in os.listdir(self.directory)
                              if name.endswith(".part")])
        self.assertTrue(response.closed)

    # Atomic preservation: a failed conversion keeps the previous complete output.
    def test_conversion_failure_preserves_existing_output(self):
        parquet_path = self.write_table(
            [encoded_image(value=1), b"malformed"], [1, 2],
            image_type=pa.binary())
        output_path = self.output_path()
        with open(output_path, "wb") as existing:
            existing.write(b"previous complete output")

        with self.assertRaises(ValueError):
            prepare_mnist.convert_parquet(parquet_path, output_path, 2)

        with open(output_path, "rb") as existing:
            self.assertEqual(b"previous complete output", existing.read())
        self.assertEqual([], [name for name in os.listdir(self.directory)
                              if name.endswith(".part")])

    # Dependency pinning: requirements carry exact target wheel hashes.
    def test_requirements_include_exact_target_wheel_hashes(self):
        expected = {
            "numpy": ("1.19.5", {
                "8b5e972b43c8fc27d56550b4120fe6257fdc15f9301914380b27f74856299fea",
                "a4646724fba402aa7504cd48b4b50e783296b5e10a524c7a6da62e4a8ac9698d",
            }),
            "pillow": ("8.4.0", {
                "25a49dc2e2f74e65efaa32b153527fc5ac98508d502fa46e74fa4fd678ed6645",
            }),
            "pyarrow": ("6.0.1", {
                "02baee816456a6e64486e587caaae2bf9f084fa3a891354ff18c3e945a1cb72f",
                "fab8132193ae095c43b1e8d6d7f393451ac198de5aaf011c6b576b1442966fec",
            }),
            "requests": ("2.27.1", {
                "f22fa1e554c9ddfd16e6e41ac79759e17be9e492b3587efa038054674760e72d",
            }),
            "certifi": ("2021.10.8", {
                "d62a0163eb4c2344ac042ab2bdf75399a71a2d8c7d47eac2e2ee91b9d6339569",
            }),
            "charset-normalizer": ("2.0.12", {
                "6881edbebdb17b39b4eaaa821b438bf6eddffb4468cf344f09f89def34a8b1df",
            }),
            "idna": ("3.3", {
                "84d9dd047ffa80596e0f246e2eab0b391788b0503584e8945f2368256d2735ff",
            }),
            "urllib3": ("1.26.18", {
                "34b97092d7e0a3a8cf7cd10e386f401b3737364026c45e622aa02903dffe0f07",
            }),
        }
        requirements_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "requirements-py36.txt")

        actual = {}
        current_name = None
        with open(requirements_path, "r") as requirements_file:
            for raw_line in requirements_file:
                line = raw_line.strip().rstrip("\\").strip()
                if "==" in line:
                    current_name, remainder = line.split("==", 1)
                    current_name = current_name.lower()
                    actual[current_name] = (remainder.split()[0], set())
                if "--hash=sha256:" in line:
                    actual[current_name][1].add(
                        line.split("--hash=sha256:", 1)[1])

        self.assertEqual(expected, actual)

    # The explicit conversion CLI writes a header plus one 784-byte image.
    def test_explicit_conversion_cli(self):
        parquet_path = self.write_table([encoded_image(value=8)], [8],
                                        image_type=pa.binary())
        output_path = self.output_path()

        self.assertEqual(0, prepare_mnist.main([
            "--input-parquet", parquet_path,
            "--output-bin", output_path,
            "--expected-count", "1",
        ]))

        self.assertEqual(24 + 784 + 1, os.path.getsize(output_path))

    @staticmethod
    def _encoded_pixels(pixels):
        image = Image.frombytes("L", (28, 28), pixels)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


# Minimal requests response double used to inject download content.
class FakeResponse(object):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        del chunk_size
        return iter(self.chunks)

    def close(self):
        self.closed = True


if __name__ == "__main__":
    unittest.main()
