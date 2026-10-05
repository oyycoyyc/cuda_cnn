from __future__ import print_function

import argparse
import hashlib
import io
import os
import struct
import tempfile

import pyarrow.parquet as pq
import requests
from PIL import Image


# Exact binary layout constants: magic, version, and image shape.
MAGIC = b"MNISTC1\0"
VERSION = 1
ROWS = 28
COLUMNS = 28

# Pinned dataset revision, base URL, and per-split checksum and count metadata.
REVISION = "77f3279092a1c1579b2250db8eafed0ad422088c"
BASE_URL = "https://huggingface.co/datasets/ylecun/mnist/resolve/" + REVISION
SPLITS = (
    ("train", BASE_URL + "/mnist/train-00000-of-00001.parquet",
     "f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf",
     60000),
    ("test", BASE_URL + "/mnist/test-00000-of-00001.parquet",
     "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
     10000),
)


# Hashes a file in fixed-size blocks without buffering the whole download.
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# Creates a closed temporary file beside the destination for atomic replacement.
def _temporary_path(destination):
    directory = os.path.dirname(os.path.abspath(destination))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = tempfile.NamedTemporaryFile(
        prefix=os.path.basename(destination) + ".", suffix=".part",
        dir=directory, delete=False)
    path = temporary.name
    temporary.close()
    return path


# Reuses a cached file only after rehashing; downloads and verifies otherwise.
def download_verified(url, destination, expected_sha256):
    """Download destination atomically, or verify an existing cached file."""
    if os.path.exists(destination):
        actual_sha256 = _sha256_file(destination)
        if actual_sha256 != expected_sha256:
            raise ValueError("cached download checksum mismatch for %s" % destination)
        return destination

    temporary_path = _temporary_path(destination)
    response = None
    try:
        response = requests.get(url, stream=True, timeout=(10, 120))
        response.raise_for_status()
        digest = hashlib.sha256()
        with open(temporary_path, "wb") as output_file:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    digest.update(block)
                    output_file.write(block)
            output_file.flush()
            os.fsync(output_file.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError("download checksum mismatch for %s" % url)
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if response is not None:
            response.close()
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


# Extracts raw encoded image bytes from a parquet image column value.
def _encoded_image_bytes(value, row_index):
    if isinstance(value, dict):
        value = value.get("bytes")
    if value is None:
        raise ValueError("image %d has null encoded data" % row_index)
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("image %d has unsupported encoded data" % row_index)
    return bytes(value)


# Decodes one image and enforces grayscale mode and the expected dimensions.
def _decode_image(value, row_index):
    encoded = _encoded_image_bytes(value, row_index)
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            if image.mode != "L":
                raise ValueError("image %d mode is %s, expected L" %
                                 (row_index, image.mode))
            if image.size != (COLUMNS, ROWS):
                raise ValueError("image %d size is %r, expected (28, 28)" %
                                 (row_index, image.size))
            return image.tobytes()
    except ValueError:
        raise
    except (OSError, SyntaxError) as error:
        raise ValueError("image %d is not a valid encoded image: %s" %
                         (row_index, error))


# Validates a split and writes the exact binary layout through atomic replacement.
def convert_parquet(parquet_path, output_path, expected_count):
    """Validate a Parquet split and atomically write the MNIST binary format."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")

    table = pq.read_table(parquet_path)
    missing_columns = [name for name in ("image", "label")
                       if name not in table.column_names]
    if missing_columns:
        raise ValueError("missing required column(s): %s" %
                         ", ".join(missing_columns))
    if table.num_rows != expected_count:
        raise ValueError("row count is %d, expected %d" %
                         (table.num_rows, expected_count))

    temporary_path = _temporary_path(output_path)
    try:
        labels = bytearray()
        with open(temporary_path, "wb") as output_file:
            output_file.write(MAGIC)
            output_file.write(struct.pack(
                "<IIII", VERSION, expected_count, ROWS, COLUMNS))
            for row_index in range(expected_count):
                image_value = table["image"][row_index].as_py()
                label = table["label"][row_index].as_py()
                if (not isinstance(label, int) or isinstance(label, bool) or
                        label < 0 or label > 9):
                    raise ValueError("label %d is outside [0, 9]" % row_index)
                output_file.write(_decode_image(image_value, row_index))
                labels.append(label)
            output_file.write(labels)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


def prepare_split(url, parquet_path, output_path, expected_sha256,
                  expected_count):
    download_verified(url, parquet_path, expected_sha256)
    convert_parquet(parquet_path, output_path, expected_count)


# Parses options and enforces the exclusive CLI conversion modes.
def _parse_arguments(arguments):
    parser = argparse.ArgumentParser(
        description="Download, verify, and convert the pinned MNIST dataset")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-parquet")
    parser.add_argument("--output-bin")
    parser.add_argument("--expected-count", type=int)
    options = parser.parse_args(arguments)

    explicit_values = (options.input_parquet, options.output_bin,
                       options.expected_count)
    if options.output_dir is not None:
        if any(value is not None for value in explicit_values):
            parser.error("--output-dir cannot be combined with explicit conversion")
    elif any(value is not None for value in explicit_values):
        if not all(value is not None for value in explicit_values):
            parser.error("explicit conversion requires --input-parquet, "
                         "--output-bin, and --expected-count")
    else:
        parser.error("provide --output-dir or explicit conversion arguments")
    return options


# Dispatches bulk dataset preparation or explicit parquet conversion.
def main(arguments=None):
    options = _parse_arguments(arguments)
    if options.output_dir is not None:
        if not os.path.isdir(options.output_dir):
            os.makedirs(options.output_dir)
        download_directory = "downloads"
        if not os.path.isdir(download_directory):
            os.makedirs(download_directory)
        for name, url, expected_sha256, expected_count in SPLITS:
            prepare_split(
                url,
                os.path.join(download_directory, name + ".parquet"),
                os.path.join(options.output_dir, name + ".bin"),
                expected_sha256,
                expected_count)
    else:
        convert_parquet(options.input_parquet, options.output_bin,
                        options.expected_count)
    return 0


if __name__ == "__main__":
    main()
