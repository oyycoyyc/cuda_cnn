#include "reporting.h"
#include "test_harness.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

namespace {

class CommaDecimalPoint : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

}  // namespace

TEST_CASE(epoch_summary_has_exact_stable_output) {
  std::ostringstream output;
  PrintEpochSummary(output, 1, 0.123456F, 0.9876F, 1234.567);
  EXPECT_EQ(
      std::string("event=epoch epoch=1 train_loss=0.123456 "
                  "validation_accuracy=0.987600 elapsed_ms=1234.567\n"),
      output.str());
}

TEST_CASE(device_summary_has_exact_stable_output) {
  std::ostringstream output;
  PrintDeviceSummary(output, 0, "NVIDIA H20", 9, 0);
  EXPECT_EQ(std::string("event=device index=0 name=NVIDIA H20 "
                        "compute_capability=9.0\n"),
            output.str());
}

TEST_CASE(final_test_summary_has_exact_stable_output) {
  std::ostringstream output;
  PrintFinalTestSummary(output, 10000, 0.9912F, 17, 0.992F);
  EXPECT_EQ(std::string("event=final_test samples=10000 "
                        "final_test_accuracy=0.991200 best_epoch=17 "
                        "validation_accuracy=0.992000\n"),
            output.str());
}

TEST_CASE(inference_summary_has_exact_stable_output) {
  const std::array<float, 10> logits{{
      -1.0F, 0.0F, 1.0F, 2.0F, 3.0F,
      4.0F, 5.0F, 6.0F, 7.0F, 8.0F}};
  const std::array<float, 10> probabilities{{
      0.01F, 0.02F, 0.03F, 0.04F, 0.05F,
      0.06F, 0.07F, 0.08F, 0.09F, 0.55F}};
  std::ostringstream output;
  PrintInferenceSummary(output, 42, logits, probabilities, 9, 7);
  EXPECT_EQ(
      std::string(
          "event=infer index=42 "
          "logits=-1.000000000,0.000000000,1.000000000,2.000000000,"
          "3.000000000,4.000000000,5.000000000,6.000000000,7.000000000,"
          "8.000000000 "
          "probabilities=0.010000000,0.020000000,0.029999999,0.039999999,"
          "0.050000001,0.059999999,0.070000000,0.079999998,0.090000004,"
          "0.550000012 prediction=9 label=7\n"),
      output.str());
}

TEST_CASE(evaluation_summary_has_exact_pass_and_fail_output) {
  std::ostringstream output;
  PrintEvaluationSummary(output, 10000, 0.9912F, 0.321F, 398753.875F,
                         0.99F, true);
  EXPECT_EQ(
      std::string("event=evaluate samples=10000 accuracy=0.991200 "
                  "mean_forward_ms=0.321 images_per_second=398753.875 "
                  "min_accuracy=0.990000 status=pass\n"),
      output.str());

  output.str(std::string());
  output.clear();
  PrintEvaluationSummary(output, 2, 0.5F, 1.25F, 1600.0F, 0.75F, false);
  EXPECT_EQ(
      std::string("event=evaluate samples=2 accuracy=0.500000 "
                  "mean_forward_ms=1.250 images_per_second=1600.000 "
                  "min_accuracy=0.750000 status=fail\n"),
      output.str());
}

TEST_CASE(reporting_ignores_and_restores_stream_locale_and_formatting) {
  std::ostringstream output;
  const std::locale comma_locale(std::locale::classic(),
                                 new CommaDecimalPoint());
  output.imbue(comma_locale);
  output << std::hex << std::showbase << std::showpos << std::uppercase
         << std::scientific;
  output.precision(2);
  output.fill('*');
  output.width(11);
  const std::ios::fmtflags original_flags = output.flags();
  const std::streamsize original_precision = output.precision();
  const char original_fill = output.fill();
  const std::streamsize original_width = output.width();
  const std::locale original_locale = output.getloc();

  PrintEpochSummary(output, 1, 0.5F, 0.25F, 2.5);
  PrintEvaluationSummary(output, 10000, 0.9912F, 0.321F, 398753.875F,
                         0.99F, true);

  EXPECT_EQ(std::string("event=epoch epoch=1 train_loss=0.500000 "
                        "validation_accuracy=0.250000 elapsed_ms=2.500\n"
                        "event=evaluate samples=10000 accuracy=0.991200 "
                        "mean_forward_ms=0.321 "
                        "images_per_second=398753.875 "
                        "min_accuracy=0.990000 status=pass\n"),
            output.str());
  EXPECT_TRUE(output.flags() == original_flags);
  EXPECT_EQ(original_precision, output.precision());
  EXPECT_EQ(original_fill, output.fill());
  EXPECT_EQ(original_width, output.width());
  EXPECT_TRUE(output.getloc() == original_locale);
}

TEST_CASE(epoch_summary_rejects_invalid_contract_values) {
  std::ostringstream output;
  EXPECT_THROW_CONTAINS(PrintEpochSummary(output, 0, 0.1F, 0.9F, 1.0),
                        "epoch");
  EXPECT_THROW_CONTAINS(PrintEpochSummary(output, 1, -0.1F, 0.9F, 1.0),
                        "training_loss");
  EXPECT_THROW_CONTAINS(
      PrintEpochSummary(output, 1, std::nanf(""), 0.9F, 1.0),
      "training_loss");
  EXPECT_THROW_CONTAINS(PrintEpochSummary(output, 1, 0.1F, 1.1F, 1.0),
                        "validation_accuracy");
  EXPECT_THROW_CONTAINS(
      PrintEpochSummary(output, 1, 0.1F, 0.9F,
                        std::numeric_limits<double>::infinity()),
      "elapsed_ms");
  EXPECT_TRUE(output.str().empty());
}

TEST_CASE(evaluation_summary_rejects_invalid_contract_values) {
  std::ostringstream output;
  EXPECT_THROW_CONTAINS(
      PrintEvaluationSummary(output, 0, 0.9F, 1.0F, 2.0F, 0.8F, true),
      "samples");
  EXPECT_THROW_CONTAINS(
      PrintEvaluationSummary(output, 1, -0.1F, 1.0F, 2.0F, 0.8F, false),
      "accuracy");
  EXPECT_THROW_CONTAINS(
      PrintEvaluationSummary(output, 1, 0.9F, -1.0F, 2.0F, 0.8F, true),
      "mean_forward_ms");
  EXPECT_THROW_CONTAINS(
      PrintEvaluationSummary(output, 1, 0.9F, 1.0F,
                             std::numeric_limits<float>::infinity(), 0.8F,
                             true),
      "images_per_second");
  EXPECT_THROW_CONTAINS(
      PrintEvaluationSummary(output, 1, 0.9F, 1.0F, 2.0F,
                             std::nanf(""), false),
      "minimum_accuracy");
  EXPECT_TRUE(output.str().empty());
}
