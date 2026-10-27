from __future__ import print_function

import hashlib
import math
import os
import shutil
import struct
import subprocess
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")


# Resolves an environment-selected path against the repository root.
def environment_path(name, default):
    value = os.environ.get(name, default)
    if os.path.isabs(value):
        return value
    return os.path.join(ROOT, value)


# The utility name is platform neutral; Linux/H20 is the execution target.
DEFAULT_EXE = os.path.join(
    BUILD_DIR, "evaluate_snapshot.exe" if os.name == "nt" else "evaluate_snapshot")
EXE = environment_path("SNAPSHOT_EXE", DEFAULT_EXE)

# Defaults select the untouched official inputs used by the comparison runs.
DEFAULT_DATA = "/gemini/data-1/test.bin"
DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "lenet.bin")
DATA = environment_path("SNAPSHOT_DATA", DEFAULT_DATA)
WEIGHTS = environment_path("SNAPSHOT_WEIGHTS", DEFAULT_WEIGHTS)

SAMPLE_COUNT = 10000
CLASS_COUNT = 10
IMAGE_PIXELS = 28 * 28
DATASET_HEADER_BYTES = 24
EXPECTED_SNAPSHOT_BYTES = (
    SAMPLE_COUNT * CLASS_COUNT * 4 * 2 + SAMPLE_COUNT)
LOGITS_END = SAMPLE_COUNT * CLASS_COUNT * 4
PROBABILITIES_END = LOGITS_END + SAMPLE_COUNT * CLASS_COUNT * 4

MOUNT_VARIABLES = (
    "GEMINI_DATA_IN1", "GEMINI_DATA_IN2", "GEMINI_DATA_IN3",
    "GEMINI_PRETRAIN")


# Reads one little-endian MNISTC1 header and returns its count and payloads.
def read_dataset_layout(content):
    if len(content) < DATASET_HEADER_BYTES or content[:8] != b"MNISTC1\0":
        raise AssertionError("source dataset is not a MNISTC1 file")
    count, rows, columns = struct.unpack("<III", content[12:24])
    images_end = DATASET_HEADER_BYTES + count * IMAGE_PIXELS
    if rows != 28 or columns != 28 or images_end + count != len(content):
        raise AssertionError("source dataset layout is inconsistent")
    return count, images_end


# Writes a smaller but fully valid dataset so count validation can be observed.
def write_dataset_subset(source_path, destination_path, requested_count):
    with open(source_path, "rb") as source_file:
        content = source_file.read()
    count, images_end = read_dataset_layout(content)
    if requested_count < 1 or requested_count > count:
        raise AssertionError("requested dataset subset is invalid")
    header = bytearray(content[:DATASET_HEADER_BYTES])
    header[12:16] = struct.pack("<I", requested_count)
    images = content[DATASET_HEADER_BYTES:DATASET_HEADER_BYTES +
                     requested_count * IMAGE_PIXELS]
    labels = content[images_end:images_end + requested_count]
    with open(destination_path, "wb") as destination_file:
        destination_file.write(bytes(header))
        destination_file.write(images)
        destination_file.write(labels)


# Returns the SHA-256 digest of one file for preservation/identity checks.
def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while True:
            block = input_file.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class EvaluateSnapshotTest(unittest.TestCase):
    # Creates a scratch area under build/ so no input mount receives test data.
    def setUp(self):
        if not os.path.isdir(BUILD_DIR):
            os.makedirs(BUILD_DIR)
        self.temporary = tempfile.mkdtemp(
            prefix="evaluate-snapshot-", dir=BUILD_DIR)

    # Removes every fixture and output created by one test.
    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    # Fails with an actionable message when the utility was never built.
    def require_executable(self):
        if not os.path.isfile(EXE):
            self.fail(
                "snapshot utility is unavailable at {0}; build it with the "
                "project Makefile or set SNAPSHOT_EXE".format(EXE))
        if not os.access(EXE, os.X_OK):
            self.fail("snapshot utility is not executable: {0}".format(EXE))

    # Skips tests that need the untouched official inputs on this host.
    def require_inputs(self):
        for label, path in (("SNAPSHOT_DATA", DATA),
                            ("SNAPSHOT_WEIGHTS", WEIGHTS)):
            if not os.path.isfile(path):
                self.skipTest("{0} is unavailable: {1}".format(label, path))

    # Returns a fresh path inside this test's owned scratch directory.
    def scratch_path(self, name):
        return os.path.join(self.temporary, name)

    # Runs the utility with explicit arguments and captures both streams.
    def run_tool(self, arguments, environment=None):
        return subprocess.run(
            [EXE] + list(arguments),
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=900,
        )

    # The very first check gives a clear RED before the utility exists.
    def test_000_snapshot_utility_is_built(self):
        self.require_executable()

    # Exactly four positional arguments are required; no output may appear.
    def test_requires_exactly_four_positional_arguments(self):
        self.require_executable()
        output = self.scratch_path("argument-count.bin")
        cases = (
            (),
            (DATA,),
            (DATA, WEIGHTS),
            (DATA, WEIGHTS, output),
            (DATA, WEIGHTS, output, "1", "extra"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.run_tool(arguments)
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertIn("usage", result.stderr.lower())
                self.assertFalse(os.path.exists(output))

    # Batch sizes must be plain decimal digits in the inclusive [1,128] range.
    def test_rejects_out_of_range_and_malformed_batch_sizes(self):
        self.require_executable()
        self.require_inputs()
        cases = ("0", "129", "256", "-1", "+1", "1.0", "abc", "", " 1",
                 "1 ", "0x10", "999999999999999999999")
        for index, value in enumerate(cases):
            output = self.scratch_path("batch-{0}.bin".format(index))
            with self.subTest(batch=value):
                result = self.run_tool([DATA, WEIGHTS, output, value])
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertFalse(os.path.exists(output))

    # A full official snapshot is exactly 810000 bytes in the documented order.
    def test_full_official_dataset_snapshot_has_exact_layout_and_values(self):
        self.require_executable()
        self.require_inputs()
        output = self.scratch_path("official-full.bin")
        result = self.run_tool([DATA, WEIGHTS, output, "128"])
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(EXPECTED_SNAPSHOT_BYTES, os.path.getsize(output))

        with open(output, "rb") as input_file:
            content = input_file.read()
        logits = struct.unpack("<{0}f".format(SAMPLE_COUNT * CLASS_COUNT),
                               content[:LOGITS_END])
        probabilities = struct.unpack(
            "<{0}f".format(SAMPLE_COUNT * CLASS_COUNT),
            content[LOGITS_END:PROBABILITIES_END])
        predictions = content[PROBABILITIES_END:]
        self.assertEqual(SAMPLE_COUNT, len(predictions))

        for index, value in enumerate(logits):
            if not math.isfinite(value):
                self.fail("logit {0} is not finite: {1}".format(index, value))
        for index, value in enumerate(probabilities):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                self.fail(
                    "probability {0} is outside [0,1] or not finite: "
                    "{1}".format(index, value))

        for sample in range(SAMPLE_COUNT):
            base = sample * CLASS_COUNT
            row_sum = math.fsum(probabilities[base:base + CLASS_COUNT])
            if abs(row_sum - 1.0) > 1.0e-3:
                self.fail(
                    "probability row {0} does not sum to one: {1}".format(
                        sample, row_sum))
            best = 0
            for class_index in range(1, CLASS_COUNT):
                if logits[base + class_index] > logits[base + best]:
                    best = class_index
            if predictions[sample] >= CLASS_COUNT:
                self.fail(
                    "prediction {0} is outside the class range: {1}".format(
                        sample, predictions[sample]))
            if predictions[sample] != best:
                self.fail(
                    "prediction {0} does not match the logits argmax: "
                    "{1} versus {2}".format(
                        sample, predictions[sample], best))

    # Re-running the same command must reproduce the snapshot byte for byte.
    def test_repeated_full_snapshots_are_byte_identical(self):
        self.require_executable()
        self.require_inputs()
        first = self.scratch_path("repeat-first.bin")
        second = self.scratch_path("repeat-second.bin")
        for output in (first, second):
            result = self.run_tool([DATA, WEIGHTS, output, "128"])
            self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(file_digest(first), file_digest(second))

    # An existing output must be preserved; exclusive creation forbids reuse.
    def test_refuses_to_overwrite_an_existing_output_file(self):
        self.require_executable()
        self.require_inputs()
        output = self.scratch_path("existing.bin")
        sentinel = b"existing output must survive\n"
        with open(output, "wb") as existing_file:
            existing_file.write(sentinel)

        result = self.run_tool([DATA, WEIGHTS, output, "128"])
        self.assertNotEqual(0, result.returncode, result.stdout)
        with open(output, "rb") as existing_file:
            self.assertEqual(sentinel, existing_file.read())

    # Symlink aliases of weight/data copies must not be truncated or replaced.
    def test_refuses_symlink_aliases_of_weight_and_data_files(self):
        self.require_executable()
        self.require_inputs()
        for label, source in (("weights", WEIGHTS), ("data", DATA)):
            target = self.scratch_path(label + "-copy.bin")
            alias = self.scratch_path(label + "-alias.bin")
            shutil.copyfile(source, target)
            original_digest = file_digest(target)
            try:
                os.symlink(target, alias)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable on this host")
            with self.subTest(alias=label):
                result = self.run_tool([DATA, WEIGHTS, alias, "128"])
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertTrue(os.path.islink(alias))
                self.assertEqual(original_digest, file_digest(target))

    # Simulated data and pretrain mounts are rejected before any output file
    # can be created. These are build-owned temp directories only; no test
    # probes or writes to a real mounted input root.
    def test_rejects_outputs_inside_simulated_environment_mounts(self):
        self.require_executable()
        self.require_inputs()
        environment = os.environ.copy()
        for index, variable in enumerate(MOUNT_VARIABLES):
            mount = self.scratch_path("simulated-mount-{0}".format(index))
            os.makedirs(mount)
            environment[variable] = mount
        for variable in MOUNT_VARIABLES:
            mount = environment[variable]
            output = os.path.join(mount, "snapshot.bin")
            with self.subTest(variable=variable):
                result = self.run_tool([DATA, WEIGHTS, output, "128"],
                                       environment)
                self.assertNotEqual(0, result.returncode, result.stdout)
                self.assertEqual([], os.listdir(mount))
                self.assertFalse(os.path.exists(output))

    # An environment-selected mount is rejected even when it is writable.
    def test_rejects_outputs_under_environment_selected_mount(self):
        self.require_executable()
        self.require_inputs()
        mount = self.scratch_path("fake-mounted-data")
        os.makedirs(mount)
        output = os.path.join(mount, "snapshot.bin")
        environment = os.environ.copy()
        environment["GEMINI_DATA_IN3"] = mount
        result = self.run_tool([DATA, WEIGHTS, output, "128"], environment)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual([], os.listdir(mount))

    # A symlinked output parent that resolves into a mount is also rejected.
    def test_rejects_symlinked_output_parent_inside_a_mount(self):
        self.require_executable()
        self.require_inputs()
        mount = self.scratch_path("fake-mount-for-symlink")
        os.makedirs(mount)
        alias_parent = self.scratch_path("mount-alias")
        try:
            os.symlink(mount, alias_parent)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable")
        output = os.path.join(alias_parent, "snapshot.bin")
        environment = os.environ.copy()
        environment["GEMINI_DATA_IN3"] = mount
        result = self.run_tool([DATA, WEIGHTS, output, "128"], environment)
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertEqual([], os.listdir(mount))

    # Regression: an alias whose resolved parent is a NESTED directory below a
    # mount must also be rejected. Lexical ancestor comparison alone matches the
    # mount root but misses the nested inode. Everything here is build-owned.
    def test_rejects_symlinked_parent_into_nested_mount_directory(self):
        self.require_executable()
        self.require_inputs()
        mount = self.scratch_path("fake-mount-nested")
        level_one = os.path.join(mount, "level-one")
        nested = os.path.join(level_one, "level-two")
        os.makedirs(nested)
        original_mount_children = sorted(os.listdir(mount))
        original_level_one_children = sorted(os.listdir(level_one))
        alias_parent = self.scratch_path("nested-mount-alias")
        try:
            os.symlink(nested, alias_parent)
        except (OSError, NotImplementedError):
            self.skipTest("directory symlink creation is unavailable")
        output = os.path.join(alias_parent, "snapshot.bin")
        environment = os.environ.copy()
        environment["GEMINI_DATA_IN3"] = mount
        result = self.run_tool([DATA, WEIGHTS, output, "128"], environment)
        self.assertEqual([], os.listdir(nested),
                         "nested mount directory must remain empty")
        self.assertFalse(os.path.exists(output))
        self.assertEqual(original_mount_children, sorted(os.listdir(mount)))
        self.assertEqual(original_level_one_children,
                         sorted(os.listdir(level_one)))
        self.assertNotEqual(0, result.returncode, result.stdout)

    # A dataset with any other sample count must fail before GPU allocation.
    def test_rejects_dataset_with_nonproduction_sample_count(self):
        self.require_executable()
        self.require_inputs()
        fixture = self.scratch_path("dataset-9999.bin")
        write_dataset_subset(DATA, fixture, SAMPLE_COUNT - 1)
        output = self.scratch_path("wrong-count.bin")
        result = self.run_tool([fixture, WEIGHTS, output, "128"])
        self.assertNotEqual(0, result.returncode, result.stdout)
        self.assertIn("10000", result.stderr)
        self.assertFalse(os.path.exists(output))


if __name__ == "__main__":
    unittest.main()
