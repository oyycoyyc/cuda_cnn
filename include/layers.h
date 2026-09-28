#ifndef INCLUDE_LAYERS_H_
#define INCLUDE_LAYERS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Converts images from contiguous uint8 [batch_size][28][28] storage to FP32
// NCHW [batch_size][1][28][28], optionally applying deterministic translation
// from seed, one_based_epoch, and original_indices[batch]. Positive dx/dy moves
// content right/down: output (x,y) samples source (x-dx,y-dy). Out-of-bounds
// source pixels are uint8 zero and are then normalized by
// (pixel/255-0.1307)/0.3081. images, indices, and output are non-null caller-
// owned device buffers of at least batch_size*784, batch_size, and
// batch_size*784 elements, remain live through stream work, and do not overlap.
// batch_size >= 1 and must fit one CUDA x-grid; seed/indices may hold any values;
// one_based_epoch >= 1 when augment is true and is ignored otherwise. stream is
// valid or null. Invalid batch/grid dimensions or augmented epoch throw
// std::invalid_argument/std::overflow_error before launch; pointer/extents are
// caller preconditions. Immediate launch failure throws std::runtime_error; no
// synchronization occurs, so asynchronous errors surface at a later boundary.
void LaunchNormalizeTranslate(const std::uint8_t* images,
    const std::uint32_t* original_indices, float* output, int batch_size,
    std::uint64_t seed, std::uint32_t one_based_epoch, bool augment,
    cudaStream_t stream);

// Writes zero to count contiguous FP32 values. values is a caller-owned device
// buffer of at least count elements, remains live through stream work, and may
// be null exactly when count == 0; zero count is a no-op. count <=
// SIZE_MAX/sizeof(float), and stream is a valid stream or null default stream.
// An overflowing count throws std::overflow_error before launch; pointer extent
// remains a caller precondition. Immediate launch failure throws
// std::runtime_error; no synchronization occurs and async errors surface later.
void LaunchZero(float* values, std::size_t count, cudaStream_t stream);

// Applies output[i]=max(0,input[i]) to count contiguous FP32 values. Caller-
// owned device buffers contain at least count elements and remain live through
// stream work. Both may be null exactly when count == 0; otherwise both are
// non-null. They may be identical for in-place use or disjoint, but may not
// partially overlap. count <= SIZE_MAX/sizeof(float); stream is valid or null.
// An overflowing count throws std::overflow_error before launch; pointer and
// aliasing preconditions are caller-validated. Immediate launch failure throws
// std::runtime_error; no sync occurs and async errors surface later.
void LaunchReluForward(const float* input, float* output,
    std::size_t count, cudaStream_t stream);

// Computes input_gradient[i]=(forward_input[i]>0?output_gradient[i]:0) for count
// contiguous FP32 values. Each caller-owned device buffer has at least count
// elements and remains live through stream work; all may be null exactly for
// count == 0. output_gradient and input_gradient may be identical or disjoint;
// forward_input is disjoint from both, and partial overlaps are forbidden.
// count <= SIZE_MAX/sizeof(float); stream is valid or null.
// An overflowing count throws std::overflow_error before launch; pointer and
// aliasing preconditions are caller-validated. Immediate launch failure throws
// std::runtime_error; no sync occurs and async errors surface later.
void LaunchReluBackward(const float* forward_input,
    const float* output_gradient, float* input_gradient,
    std::size_t count, cudaStream_t stream);

// Computes non-overlapping 2x2 stride-two max pooling from FP32 NCHW
// [batch_size][channels][input_height][input_width] to NCHW with half spatial
// dimensions, writing one uint8 row-major winner offset in [0,3] per output;
// equal values choose the smallest offset. All pointers are non-null, disjoint,
// caller-owned device buffers with extents implied by these shapes and remain
// live through stream work. batch_size, channels >= 1; input_height,input_width
// are even and >= 2; element/byte extents fit size_t; stream is valid or null.
// Invalid/overflowing dimensions throw std::invalid_argument or
// std::overflow_error before launch; pointer extents remain caller preconditions.
// Immediate launch failure throws std::runtime_error; no sync occurs and async
// errors surface later.
void LaunchMaxPoolForward(const float* input, float* output,
    std::uint8_t* winner_offsets, int batch_size, int channels,
    int input_height, int input_width, cudaStream_t stream);

// Zeroes an FP32 NCHW [batch_size][channels][input_height][input_width] input
// gradient, then scatters each half-spatial NCHW output gradient through its
// uint8 winner offset. All pointers are non-null, disjoint caller-owned device
// buffers with implied extents and remain live through stream work; every
// winner offset is in [0,3]. batch_size,channels >= 1; spatial dimensions are
// even and >= 2; element/byte extents fit size_t; stream is valid or null.
// Invalid/overflowing dimensions throw std::invalid_argument or
// std::overflow_error before launch. Pointer extents and device offset values
// remain caller preconditions. Immediate launch failure throws
// std::runtime_error; no sync occurs and async errors surface later.
void LaunchMaxPoolBackward(const float* output_gradient,
    const std::uint8_t* winner_offsets, float* input_gradient,
    int batch_size, int channels, int input_height, int input_width,
    cudaStream_t stream);

// Computes output=input*weight^T+bias for row-major FP32 [batch_size]
// [input_features] input, [output_features][input_features] weight,
// [output_features] bias, and [batch_size][output_features] output. All pointers
// are non-null, pairwise-disjoint caller-owned device buffers with those minimum
// extents and remain live through stream work. Every dimension >= 1, all
// element/byte extents fit size_t, and stream is valid or null. Invalid
// dimensions, null pointers, overlap, or overflowing extents throw
// std::invalid_argument/std::overflow_error before launch; allocation extents
// remain caller preconditions. Immediate launch failure throws
// std::runtime_error; no sync occurs and asynchronous errors surface later.
void LaunchLinearForward(const float* input, const float* weight,
    const float* bias, float* output, int batch_size, int input_features,
    int output_features, cudaStream_t stream);

// Computes fully connected gradients from row-major FP32 input [batch_size]
// [input_features], weight [output_features][input_features], and incoming
// gradient [batch_size][output_features], writing equal-layout input/weight
// gradients and [output_features] bias gradient. All six pointers are non-null,
// pairwise-disjoint caller-owned device buffers of the implied extents and live
// through stream work. Dimensions >= 1, element/byte extents fit size_t, and
// stream is valid or null. Invalid dimensions, null pointers, overlap, or
// overflowing extents throw std::invalid_argument/std::overflow_error before
// launch; allocation extents remain caller preconditions. Immediate launch
// failure throws std::runtime_error; no sync occurs and async errors surface
// later.
void LaunchLinearBackward(const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient, int batch_size,
    int input_features, int output_features, cudaStream_t stream);

// Computes stride-one unpadded FP32 convolution from NCHW [batch_size]
// [input_channels][input_height][input_width], OIHW [output_channels]
// [input_channels][kernel_height][kernel_width], and [output_channels] bias to
// NCHW output spatial size input-kernel+1. All four pointers are non-null,
// pairwise-disjoint caller-owned device buffers of the implied extents and live
// through stream work. All dimensions >= 1; each kernel dimension is no larger
// than its input dimension; element/byte extents fit size_t; stream is valid or
// null. Invalid dimensions, kernels, null pointers, overlap, or overflowing
// extents throw std::invalid_argument/std::overflow_error before launch;
// allocation extents remain caller preconditions. Immediate launch failure
// throws std::runtime_error; no sync occurs and async errors surface later.
void LaunchConvolutionForward(const float* input, const float* weight,
    const float* bias, float* output, int batch_size, int input_channels,
    int input_height, int input_width, int output_channels,
    int kernel_height, int kernel_width, cudaStream_t stream);

// Computes FP32 valid-convolution gradients from NCHW input, OIHW weight, and
// derived-spatial NCHW output_gradient, writing NCHW input, OIHW weight, and
// [output_channels] bias gradients. All six pointers are non-null, pairwise-
// disjoint caller-owned device buffers of the forward/implied extents and live
// through stream work. Every dimension >= 1; kernel dimensions fit input;
// element/byte extents fit size_t; stream is valid or null. Invalid dimensions,
// kernels, null pointers, overlap, or overflowing extents throw
// std::invalid_argument/std::overflow_error before launch; allocation extents
// remain caller preconditions. Immediate launch failure throws
// std::runtime_error; no sync occurs and asynchronous errors surface later.
void LaunchConvolutionBackward(const float* input, const float* weight,
    const float* output_gradient, float* input_gradient,
    float* weight_gradient, float* bias_gradient, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width,
    cudaStream_t stream);

// Computes max-subtracted softmax, [batch_size] losses, scalar actual-batch mean,
// and (probability-one_hot)/batch_size gradient from finite FP32 row-major
// logits [batch_size][class_count] and uint8 labels [batch_size], writing
// probabilities/gradients of logits shape. All six pointers are non-null,
// pairwise-disjoint caller-owned device buffers of these extents and live
// through stream work. batch_size >= 1; 1 <= class_count <= 256; each label is
// less than class_count; element/byte extents fit size_t; stream is valid or
// null. Invalid dimensions, null pointers, overlap, or overflowing pointer/
// element extents throw std::invalid_argument/std::overflow_error before
// launch; labels and other device values are not host-validated. Immediate
// runtime/launch failure throws std::runtime_error; no sync occurs and async
// errors surface later.
void LaunchSoftmaxCrossEntropy(const float* logits,
    const std::uint8_t* labels, float* probabilities,
    float* per_sample_losses, float* mean_loss, float* logits_gradient,
    int batch_size, int class_count, cudaStream_t stream);

// Computes max-subtracted softmax from finite FP32 row-major logits
// [batch_size][class_count] into an equal-shaped probability buffer. Both
// pointers are non-null, disjoint caller-owned device buffers of at least
// batch_size*class_count elements and live through stream work. batch_size >= 1,
// 1 <= class_count <= 256, element/byte extents fit size_t, and stream is valid
// or null. Invalid dimensions, null pointers, overlap, or overflowing pointer/
// element extents throw std::invalid_argument/std::overflow_error before
// launch; device values are not host-validated. Immediate launch failure throws
// std::runtime_error; no sync occurs and async errors surface later.
void LaunchSoftmax(const float* logits, float* probabilities,
    int batch_size, int class_count, cudaStream_t stream);

// Finds the smallest-index argmax of each finite FP32 row-major logits row
// [batch_size][class_count], writes uint8 predictions and int correct_flags of
// length batch_size, then deterministically writes scalar int correct_count by
// comparing uint8 labels [batch_size]. All five pointers are non-null, pairwise-
// disjoint caller-owned device buffers of these extents and live through stream
// work. batch_size >= 1; 1 <= class_count <= 256; each label < class_count;
// element/byte extents fit size_t; stream is valid or null. Device values are
// not host-validated. Invalid dimensions, null pointers, overlap, or overflowing
// pointer/element extents throw std::invalid_argument/std::overflow_error before
// launch. Immediate runtime/launch failure throws std::runtime_error; no sync
// occurs and async errors surface at the caller's later sync boundary.
void LaunchArgmaxAndCountCorrect(const float* logits,
    const std::uint8_t* labels, std::uint8_t* predictions,
    int* correct_flags, int* correct_count, int batch_size,
    int class_count, cudaStream_t stream);

// Applies one FP32 decoupled AdamW update to count-element parameter, gradient,
// first-moment, and second-moment arrays; decay uses the pre-update parameter.
// For count > 0 all pointers are non-null, pairwise disjoint caller-owned device
// buffers of at least count elements and live through stream work; all may be
// null when count == 0, which is a no-op. Array values are finite and second
// moments are nonnegative. learning_rate and weight_decay are finite and >= 0;
// beta1,beta2 are finite in [0,1); epsilon is finite and > 0. Each inverse bias
// correction is finite, >= 1, and is the FP32 host-computed value of
// 1/(1-beta^t) for its matching beta and the same integer t >= 1. The update is
// m'=beta1*m+(1-beta1)*g, v'=beta2*v+(1-beta2)*g*g, and
// p'=p-lr*((m'*inverse1)/(sqrt(v'*inverse2)+epsilon)+weight_decay*p), where p
// in the decay term is pre-update. count <= SIZE_MAX/sizeof(float); stream is
// valid or null. Invalid scalars, null pointers, overlap, or overflowing
// pointer/element extents throw std::invalid_argument/std::overflow_error
// before launch. Immediate launch failure throws std::runtime_error; no sync
// occurs and async errors surface later.
void LaunchAdamW(float* parameters, const float* gradients,
    float* first_moments, float* second_moments, std::size_t count,
    float learning_rate, float beta1, float beta2, float epsilon,
    float inverse_bias_correction1, float inverse_bias_correction2,
    float weight_decay, cudaStream_t stream);

// Writes the smallest index of a NaN or infinity in count contiguous FP32 values
// to scalar int first_bad_index, or writes static_cast<int>(count) when all are
// finite. first_bad_index is always a non-null caller-owned device pointer;
// values is a disjoint caller-owned device buffer of at least count elements and
// may be null exactly when count == 0. Buffers remain live through stream work.
// The exact representability preconditions are count <= INT_MAX and the byte
// extent fits size_t; stream is valid or null. Invalid count, null pointers,
// overlap, or overflowing pointer/element extents throw
// std::invalid_argument/std::overflow_error before launch. Immediate
// runtime/launch failure throws std::runtime_error; no sync occurs, and the
// caller must synchronize before reading the result or attributing asynchronous
// errors.
void LaunchFindFirstNonFinite(const float* values, std::size_t count,
    int* first_bad_index, cudaStream_t stream);

#endif  // INCLUDE_LAYERS_H_
