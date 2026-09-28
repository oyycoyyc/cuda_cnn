#include "cpu_reference.h"
#include "test_harness.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

void ExpectVectorNear(const std::vector<float>& expected,
                      const std::vector<float>& actual, float tolerance) {
  EXPECT_EQ(expected.size(), actual.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(expected[index], actual[index], tolerance);
  }
}

}  // namespace

TEST_CASE(convolution_forward_matches_hand_calculated_valid_result) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<float> weight{1, 2, 3, 4};
  const std::vector<float> bias{0.5F};

  const std::vector<float> output = cpu_reference::ConvolutionForward(
      input, weight, bias, 1, 1, 3, 3, 1, 2, 2);

  ExpectVectorNear({37.5F, 47.5F, 67.5F, 77.5F}, output, 0.0F);
}

TEST_CASE(convolution_backward_matches_hand_calculated_gather_gradients) {
  const std::vector<float> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::vector<float> weight{1, 0, 0, -1};
  const std::vector<float> output_gradient{1, 2, 3, 4};

  const cpu_reference::ConvolutionGradients gradients =
      cpu_reference::ConvolutionBackward(input, weight, output_gradient, 1, 1,
                                         3, 3, 1, 2, 2);

  ExpectVectorNear({1, 2, 0, 3, 3, -2, 0, -3, -4}, gradients.input, 0.0F);
  ExpectVectorNear({37, 47, 67, 77}, gradients.weight, 0.0F);
  ExpectVectorNear({10}, gradients.bias, 0.0F);
}

TEST_CASE(relu_backward_has_zero_derivative_at_both_signed_zeros) {
  const std::vector<float> input{-2.0F, -0.0F, 0.0F, 3.0F};
  const std::vector<float> output_gradient{5.0F, 6.0F, 7.0F, 8.0F};

  const std::vector<float> output = cpu_reference::ReluForward(input);
  const std::vector<float> input_gradient =
      cpu_reference::ReluBackward(input, output_gradient);

  ExpectVectorNear({0, 0, 0, 3}, output, 0.0F);
  ExpectVectorNear({0, 0, 0, 8}, input_gradient, 0.0F);
  EXPECT_TRUE(!std::signbit(output[1]));
}

TEST_CASE(maxpool_ties_choose_first_row_major_value_and_scatter_back) {
  const std::vector<float> input{5, 5, 1, 2, 1, 0, 2, 2};
  const cpu_reference::MaxPoolResult pooled =
      cpu_reference::MaxPoolForward(input, 1, 1, 2, 4);

  ExpectVectorNear({5, 2}, pooled.output, 0.0F);
  EXPECT_EQ(std::vector<std::uint8_t>({0, 1}), pooled.winner_offsets);

  const std::vector<float> input_gradient = cpu_reference::MaxPoolBackward(
      {3, 4}, pooled.winner_offsets, 1, 1, 2, 4);
  ExpectVectorNear({3, 0, 0, 4, 0, 0, 0, 0}, input_gradient, 0.0F);
}

TEST_CASE(linear_forward_and_backward_match_hand_calculated_results) {
  const std::vector<float> input{1, 2, 3, 4};
  const std::vector<float> weight{1, 2, -1, 3};
  const std::vector<float> bias{0.5F, -0.5F};
  const std::vector<float> output_gradient{1, 2, 3, 4};

  const std::vector<float> output =
      cpu_reference::LinearForward(input, weight, bias, 2, 2, 2);
  const cpu_reference::LinearGradients gradients =
      cpu_reference::LinearBackward(input, weight, output_gradient, 2, 2, 2);

  ExpectVectorNear({5.5F, 4.5F, 11.5F, 8.5F}, output, 0.0F);
  ExpectVectorNear({-1, 8, -1, 18}, gradients.input, 0.0F);
  ExpectVectorNear({10, 14, 14, 20}, gradients.weight, 0.0F);
  ExpectVectorNear({4, 6}, gradients.bias, 0.0F);
}

TEST_CASE(softmax_cross_entropy_is_stable_and_uses_actual_batch_mean) {
  const std::vector<float> logits{1000, 1000, -1000, -1000, 1000, 999};
  const std::vector<std::uint8_t> labels{0, 2};

  const cpu_reference::SoftmaxCrossEntropyResult result =
      cpu_reference::SoftmaxCrossEntropy(logits, labels, 2, 3);

  ExpectVectorNear({0.5F, 0.5F, 0.0F, 0.0F, 0.73105860F, 0.26894143F},
                   result.probabilities, 1e-7F);
  ExpectVectorNear({0.69314718F, 1.31326169F}, result.per_sample_losses,
                   1e-6F);
  EXPECT_NEAR(1.00320444F, result.mean_loss, 1e-6F);
  ExpectVectorNear({-0.25F, 0.25F, 0.0F, 0.0F, 0.36552930F, -0.36552930F},
                   result.logits_gradient, 1e-7F);
  EXPECT_TRUE(std::isfinite(result.mean_loss));
}

TEST_CASE(adamw_first_update_uses_t_one_and_pre_update_parameter_decay) {
  std::vector<double> parameters{1.0};
  const std::vector<double> gradients{0.5};
  std::vector<double> first_moments{0.0};
  std::vector<double> second_moments{0.0};

  cpu_reference::AdamWStep(&parameters, gradients, &first_moments,
                           &second_moments, 1, 0.1, 0.9, 0.999, 1e-8, 0.01);

  EXPECT_NEAR(0.899000002, parameters[0], 1e-12);
  EXPECT_NEAR(0.05, first_moments[0], 1e-15);
  EXPECT_NEAR(0.00025, second_moments[0], 1e-15);
}

TEST_CASE(adamw_ten_updates_match_hand_calculated_fp64_fixture) {
  std::vector<double> parameters{1.0};
  const std::vector<double> gradients{0.5};
  std::vector<double> first_moments{0.0};
  std::vector<double> second_moments{0.0};
  for (std::uint64_t step = 1; step <= 10; ++step) {
    cpu_reference::AdamWStep(&parameters, gradients, &first_moments,
                             &second_moments, step, 0.1, 0.9, 0.999, 1e-8,
                             0.01);
  }

  EXPECT_NEAR(-0.005467078905189868, parameters[0], 1e-12);
  EXPECT_NEAR(0.32566077995, first_moments[0], 1e-15);
  EXPECT_NEAR(0.0024887799475629495, second_moments[0], 1e-15);
}
