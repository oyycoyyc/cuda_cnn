import os
import re
import subprocess
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OutputFormatDocumentationTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "README.md"), "r") as input_file:
            self.readme = input_file.read()

    def test_host_reporting_suite_validates_production_formatters(self):
        make = os.environ.get("MAKE", "make")
        build = subprocess.run(
            [make, "build/reporting_tests"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertEqual(0, build.returncode, build.stdout)
        executable = os.path.join(ROOT, "build", "reporting_tests")
        if os.name == "nt":
            executable += ".exe"
        result = subprocess.run(
            [executable],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertEqual(0, result.returncode, result.stdout)
        self.assertIn(
            "event=test_suite name=reporting_tests status=pass", result.stdout
        )

    def assert_documented_record(self, record, pattern):
        self.assertIn(record, self.readme)
        self.assertIsNotNone(re.fullmatch(pattern, record))

    def test_device_record_grammar_is_documented(self):
        self.assert_documented_record(
            "event=device index=0 name=NVIDIA H20 compute_capability=9.0",
            r"event=device index=\d+ name=.+ compute_capability=\d+\.\d+",
        )

    def test_epoch_record_grammar_is_documented(self):
        self.assert_documented_record(
            "event=epoch epoch=1 train_loss=0.123456 "
            "validation_accuracy=0.987600 elapsed_ms=1234.567",
            r"event=epoch epoch=\d+ train_loss=\d+\.\d{6} "
            r"validation_accuracy=\d+\.\d{6} elapsed_ms=\d+\.\d{3}",
        )

    def test_final_test_record_grammar_is_documented(self):
        self.assert_documented_record(
            "event=final_test samples=10000 final_test_accuracy=0.991200 "
            "best_epoch=17 validation_accuracy=0.992000",
            r"event=final_test samples=\d+ final_test_accuracy=\d+\.\d{6} "
            r"best_epoch=\d+ validation_accuracy=\d+\.\d{6}",
        )

    def test_evaluation_record_grammar_is_documented(self):
        self.assert_documented_record(
            "event=evaluate samples=10000 accuracy=0.991200 "
            "mean_forward_ms=0.321 images_per_second=398753.875 "
            "min_accuracy=0.990000 status=pass",
            r"event=evaluate samples=\d+ accuracy=\d+\.\d{6} "
            r"mean_forward_ms=\d+\.\d{3} images_per_second=\d+\.\d{3} "
            r"min_accuracy=\d+\.\d{6} status=(?:pass|fail)",
        )

    def test_inference_record_has_exactly_ten_logits_and_probabilities(self):
        values = ",".join(["0.000000000"] * 10)
        record = (
            "event=infer index=0 logits={0} probabilities={0} "
            "prediction=0 label=0".format(values)
        )
        number = r"-?\d+\.\d{9}"
        ten = number + ("," + number) * 9
        pattern = (
            r"event=infer index=\d+ logits=" + ten +
            r" probabilities=" + ten + r" prediction=\d+ label=\d+"
        )
        self.assert_documented_record(record, pattern)

    def test_exit_code_contract_is_documented(self):
        for text in (
            "`0`: success",
            "`1`: runtime error",
            "`2`: command-line usage error",
            "`3`: evaluation accuracy below the requested threshold",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.readme)


if __name__ == "__main__":
    unittest.main()
