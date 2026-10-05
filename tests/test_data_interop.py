from __future__ import print_function

import os
import struct
import subprocess
import tempfile
import unittest


# Builds a little-endian MNISTC1 buffer with zero, constant, and range-pixel
# rows so the C++ loader must honor headers, payload, and labels.
def three_row_fixture():
    image0 = bytes(bytearray([0]) * 784)
    image1 = bytes(bytearray([127]) * 784)
    image2 = bytes(bytearray(range(256)) * 3 + bytearray(range(16)))
    return (b"MNISTC1\0" + struct.pack("<IIII", 1, 3, 28, 28) +
            image0 + image1 + image2 + bytes(bytearray([0, 5, 9])))


# Verifies the Python writer and the compiled C++ probe agree byte for byte.
class DataInteropTest(unittest.TestCase):
    # Runs the built probe on the temporary fixture and checks its exact report.
    def test_cpp_loader_agrees_with_python_little_endian_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_path = os.path.join(directory, "three-row.bin")
            with open(fixture_path, "wb") as fixture:
                fixture.write(three_row_fixture())

            probe = os.path.join("build", "dataset_probe")
            if os.name == "nt" and os.path.exists(probe + ".exe"):
                probe += ".exe"
            output = subprocess.check_output(
                [probe, fixture_path], universal_newlines=True)

        self.assertEqual(
            "count=3 rows=28 columns=28 checksum=197608 labels=0,5,9\n",
            output)


if __name__ == "__main__":
    unittest.main()
