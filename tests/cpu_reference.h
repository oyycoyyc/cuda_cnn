#ifndef TESTS_CPU_REFERENCE_H_
#define TESTS_CPU_REFERENCE_H_

#include <cstdint>
#include <vector>

namespace cpu_reference {

// Owns all valid-convolution gradients. input uses contiguous NCHW layout,
// weight uses contiguous OIHW layout, and bias has one value per output
// channel. The three vectors never alias each other or caller storage.
struct ConvolutionGradients {
  std::vector<float> input;
  std::vector<float> weight;
  std::vector<float> bias;
};

// Computes stride-one, unpadded valid convolution. input is NCHW, weight is
// OIHW, bias is [output_channels], and the returned owner stores NCHW output
// with height/width (input-kernel+1). Inputs remain caller-owned and must not
// alias each other. All dimensions must be positive, kernels must fit, and
// vector sizes must exactly match; otherwise std::invalid_argument is thrown.
std::vector<float> ConvolutionForward(
    const std::vector<float>& input, const std::vector<float>& weight,
    const std::vector<float>& bias, int batch_size, int input_channels,
    int input_height, int input_width, int output_channels,
    int kernel_height, int kernel_width);

// Computes gradients for the valid convolution described above. The incoming
// gradient is contiguous NCHW with the derived output shape. Returned buffers
// are newly owned, do not alias inputs, and use input NCHW, weight OIHW, and
// bias [output_channels] layouts. Dimensions and sizes have the same positive,
// fitting preconditions as ConvolutionForward; violations throw
// std::invalid_argument.
ConvolutionGradients ConvolutionBackward(
    const std::vector<float>& input, const std::vector<float>& weight,
    const std::vector<float>& output_gradient, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width);

// Applies max(0,x) elementwise and returns newly owned contiguous storage.
// input may have any size, remains caller-owned, and cannot alias the result.
// This operation has no error cases.
std::vector<float> ReluForward(const std::vector<float>& input);

// Multiplies output_gradient by (forward_input > 0) elementwise, including a
// zero derivative for positive or negative zero. Inputs remain caller-owned,
// may alias each other, and the returned vector aliases neither. Sizes must
// match or std::invalid_argument is thrown.
std::vector<float> ReluBackward(
    const std::vector<float>& forward_input,
    const std::vector<float>& output_gradient);

// Owns 2x2 stride-two max-pool output in NCHW layout and one row-major winner
// offset in [0,3] per output value. The vectors do not alias caller storage or
// each other.
struct MaxPoolResult {
  std::vector<float> output;
  std::vector<std::uint8_t> winner_offsets;
};

// Computes non-overlapping 2x2 stride-two max pooling for contiguous NCHW
// input, choosing the first row-major element on ties. The input remains
// caller-owned and the result owns its storage. Batch, channels, height, and
// width must be positive, spatial dimensions must be even, and input size must
// match exactly; otherwise std::invalid_argument is thrown.
MaxPoolResult MaxPoolForward(const std::vector<float>& input, int batch_size,
                             int channels, int input_height, int input_width);

// Scatters contiguous NCHW output gradients to recorded winners and returns a
// newly owned NCHW input-gradient buffer, with non-winners set to zero.
// Inputs remain caller-owned and must not alias the result. Dimensions must be
// positive and even, sizes must match the derived shapes, and offsets must be
// in [0,3]; violations throw std::invalid_argument.
std::vector<float> MaxPoolBackward(
    const std::vector<float>& output_gradient,
    const std::vector<std::uint8_t>& winner_offsets, int batch_size,
    int channels, int input_height, int input_width);

// Owns all fully connected gradients. input is [batch_size][input_features],
// weight is row-major [output_features][input_features], and bias is
// [output_features]. The vectors do not alias each other or caller storage.
struct LinearGradients {
  std::vector<float> input;
  std::vector<float> weight;
  std::vector<float> bias;
};

// Computes a fully connected forward pass. input is row-major [batch][in],
// weight is [out][in], bias is [out], and the newly owned result is
// [batch][out]. Inputs remain caller-owned and must not alias each other.
// Dimensions must be positive and sizes exact or std::invalid_argument is
// thrown.
std::vector<float> LinearForward(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 const std::vector<float>& bias,
                                 int batch_size, int input_features,
                                 int output_features);

// Computes fully connected input, weight, and bias gradients from row-major
// [batch][out] output_gradient. Returned vectors own their storage and do not
// alias caller buffers. Inputs use the LinearForward layouts; dimensions must
// be positive and sizes exact or std::invalid_argument is thrown.
LinearGradients LinearBackward(const std::vector<float>& input,
                               const std::vector<float>& weight,
                               const std::vector<float>& output_gradient,
                               int batch_size, int input_features,
                               int output_features);

// Owns stable softmax/cross-entropy outputs. probabilities and logits_gradient
// are row-major [batch_size][class_count], per_sample_losses is [batch_size],
// and mean_loss is the actual-batch arithmetic mean. Owned buffers do not
// alias inputs or each other.
struct SoftmaxCrossEntropyResult {
  std::vector<float> probabilities;
  std::vector<float> per_sample_losses;
  float mean_loss;
  std::vector<float> logits_gradient;
};

// Computes max-subtracted softmax, per-sample negative log likelihood, mean
// loss, and (probability-one_hot)/actual_batch_size. logits is row-major
// [batch][class] and labels is [batch]. Inputs remain caller-owned and may not
// alias returned storage. Dimensions must be positive, sizes exact, and every
// label less than class_count; violations throw std::invalid_argument.
SoftmaxCrossEntropyResult SoftmaxCrossEntropy(
    const std::vector<float>& logits, const std::vector<std::uint8_t>& labels,
    int batch_size, int class_count);

// Applies one FP64 decoupled AdamW update in place using global_step >= 1.
// parameters, first_moments, and second_moments are equal-sized caller-owned
// vectors; gradients is a read-only equal-sized vector and must not be one of
// the mutated vectors. Decay uses each pre-update parameter. Hyperparameters
// require learning_rate >= 0, beta values in [0,1), epsilon > 0, and
// weight_decay >= 0. Invalid pointers, sizes, step, or hyperparameters throw
// std::invalid_argument; no storage or synchronization is retained.
void AdamWStep(std::vector<double>* parameters,
               const std::vector<double>& gradients,
               std::vector<double>* first_moments,
               std::vector<double>* second_moments,
               std::uint64_t global_step, double learning_rate, double beta1,
               double beta2, double epsilon, double weight_decay);

}  // namespace cpu_reference

#endif  // TESTS_CPU_REFERENCE_H_
