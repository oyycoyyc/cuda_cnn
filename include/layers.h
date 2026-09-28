#ifndef INCLUDE_LAYERS_H_
#define INCLUDE_LAYERS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Converts batch_size MNIST images from contiguous [N][28][28] uint8 storage
// to normalized FP32 NCHW output and optionally translates each image by a
// deterministic seed/epoch/original-index offset with zero fill. All pointers
// reference caller-owned device storage valid through stream completion;
// buffers may not alias. batch_size must be positive, indices must address the
// original dataset, and one_based_epoch must be nonzero when augment is true.
// Launch errors are reported via CUDA_KERNEL_CHECK; the call does not sync.
void LaunchNormalizeTranslate(const std::uint8_t* images,
    const std::uint32_t* original_indices, float* output, int batch_size,
    std::uint64_t seed, std::uint32_t one_based_epoch, bool augment,
    cudaStream_t stream);

// Writes zero to count caller-owned contiguous device floats. values remains
// valid through stream completion and may be null only when count is zero.
// There are no input buffers to alias. Launch errors are reported via
// CUDA_KERNEL_CHECK; work is ordered on stream without implicit synchronization.
void LaunchZero(float* values, std::size_t count, cudaStream_t stream);

// Applies max(0,x) to count contiguous device floats. Buffers are caller-owned
// through stream completion and may alias exactly for in-place operation;
// partial overlap is unsupported. Null is allowed only for count zero. Launch
// errors are reported via CUDA_KERNEL_CHECK and no synchronization is added.
void LaunchReluForward(const float* input, float* output,
    std::size_t count, cudaStream_t stream);

// Computes input_gradient[i]=(forward_input[i]>0?output_gradient[i]:0) for
// count contiguous floats. Caller-owned buffers live through stream completion;
// output_gradient may alias input_gradient exactly, while forward_input may not
// alias either writable buffer. Null is allowed only for zero count. Launch
// errors are reported via CUDA_KERNEL_CHECK without synchronization.
void LaunchReluBackward(const float* forward_input,
    const float* output_gradient, float* input_gradient,
    std::size_t count, cudaStream_t stream);

// Computes non-overlapping 2x2 stride-two NCHW max pooling and writes one
// row-major winner offset [0,3] per output. Input/output/index device buffers
// are caller-owned through stream completion and may not alias. Positive batch,
// channel, and even spatial dimensions are required. Launch errors are reported
// via CUDA_KERNEL_CHECK; the function does not synchronize.
void LaunchMaxPoolForward(const float* input, float* output,
    std::uint8_t* winner_offsets, int batch_size, int channels,
    int input_height, int input_width, cudaStream_t stream);

// Scatters NCHW pooled gradients through [0,3] winner offsets into an NCHW
// input-gradient buffer, zeroing non-winners first. Device buffers are caller-
// owned through stream completion and may not alias. Dimensions must be
// positive/even and offsets valid. Launch errors are reported via
// CUDA_KERNEL_CHECK; stream ordering is used without implicit synchronization.
void LaunchMaxPoolBackward(const float* output_gradient,
    const std::uint8_t* winner_offsets, float* input_gradient,
    int batch_size, int channels, int input_height, int input_width,
    cudaStream_t stream);

// Computes output=input*weight^T+bias for row-major [batch][input_features]
// input, [output_features][input_features] weight, [output_features] bias, and
// [batch][output_features] output. Caller-owned device buffers remain valid
// through stream completion and may not alias. Dimensions must be positive.
// Launch errors are reported via CUDA_KERNEL_CHECK without synchronization.
void LaunchLinearForward(const float* input, const float* weight,
    const float* bias, float* output, int batch_size, int input_features,
    int output_features, cudaStream_t stream);

// Computes fully connected input, weight, and bias gradients from row-major
// [batch][output_features] output_gradient. All device buffers use the forward
// layouts, are caller-owned through stream completion, and may not alias.
// Dimensions must be positive. Launch errors are reported via CUDA_KERNEL_CHECK;
// accumulation is stream-ordered and the function does not synchronize.
void LaunchLinearBackward(const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient, int batch_size,
    int input_features, int output_features, cudaStream_t stream);

// Computes stride-one unpadded convolution from NCHW input, OIHW weight, and
// [output_channels] bias into derived-shape NCHW output. Caller-owned device
// buffers remain valid through stream completion and may not alias. Dimensions
// must be positive and kernels must fit the input. Launch errors are reported
// via CUDA_KERNEL_CHECK; no implicit synchronization occurs.
void LaunchConvolutionForward(const float* input, const float* weight,
    const float* bias, float* output, int batch_size, int input_channels,
    int input_height, int input_width, int output_channels,
    int kernel_height, int kernel_width, cudaStream_t stream);

// Computes NCHW input, OIHW weight, and [output_channels] bias gradients for
// valid convolution from derived-shape NCHW output_gradient. Caller-owned
// device buffers remain valid through stream completion and may not alias.
// Positive fitting dimensions are required. Launch errors are reported via
// CUDA_KERNEL_CHECK and the function does not synchronize.
void LaunchConvolutionBackward(const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width,
    cudaStream_t stream);

// Computes stable softmax probabilities, per-sample losses, their actual-batch
// mean, and (probability-one_hot)/batch_size gradients from row-major
// [batch][class] logits and [batch] labels. Caller-owned device buffers live
// through stream completion and may not alias; labels must be in range and
// dimensions positive. Launch errors use CUDA_KERNEL_CHECK; no sync is added.
void LaunchSoftmaxCrossEntropy(const float* logits,
    const std::uint8_t* labels, float* probabilities,
    float* per_sample_losses, float* mean_loss, float* logits_gradient,
    int batch_size, int class_count, cudaStream_t stream);

// Computes max-subtracted softmax from row-major [batch][class] logits into an
// equal-shaped probability buffer. Caller-owned device buffers remain valid
// through stream completion and may not alias. Dimensions must be positive.
// Launch errors are reported via CUDA_KERNEL_CHECK without synchronization.
void LaunchSoftmax(const float* logits, float* probabilities,
    int batch_size, int class_count, cudaStream_t stream);

// Writes smallest-index argmax predictions and per-sample correctness, then a
// deterministic total, from [batch][class] logits and [batch] labels.
// predictions/correct_flags are [batch] and correct_count is scalar. Caller-
// owned device buffers live through stream completion and may not alias;
// dimensions/labels must be valid. Launch errors are checked without syncing.
void LaunchArgmaxAndCountCorrect(const float* logits,
    const std::uint8_t* labels, std::uint8_t* predictions,
    int* correct_flags, int* correct_count, int batch_size,
    int class_count, cudaStream_t stream);

// Applies one decoupled AdamW update to count contiguous parameters and moment
// values using caller-supplied inverse bias corrections for a nonzero global
// step; decay uses the pre-update parameter. Device arrays are caller-owned
// through stream completion, equal-sized, and may not alias. Hyperparameters
// must be finite with standard Adam ranges. Launch errors are checked without
// implicit synchronization.
void LaunchAdamW(float* parameters, const float* gradients,
    float* first_moments, float* second_moments, std::size_t count,
    float learning_rate, float beta1, float beta2, float epsilon,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float weight_decay, cudaStream_t stream);

// Finds the smallest index in count contiguous floats that is NaN or infinity,
// leaving count when all values are finite. values and scalar first_bad_index
// are caller-owned device buffers valid through stream completion and may not
// alias; values may be null only at count zero. Launch errors are reported via
// CUDA_KERNEL_CHECK and results require caller synchronization before host use.
void LaunchFindFirstNonFinite(const float* values, std::size_t count,
    int* first_bad_index, cudaStream_t stream);

#endif  // INCLUDE_LAYERS_H_
