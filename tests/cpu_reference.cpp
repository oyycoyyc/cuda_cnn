#include "cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace cpu_reference {
namespace {

std::size_t CheckedProduct(std::initializer_list<int> dimensions,
                           const char* operation) {
  std::size_t product = 1;
  for (int dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(std::string(operation) +
                                  " dimensions must be positive");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (product > std::numeric_limits<std::size_t>::max() / value) {
      throw std::invalid_argument(std::string(operation) +
                                  " dimensions overflow size_t");
    }
    product *= value;
  }
  return product;
}

void RequireSize(std::size_t actual, std::size_t expected, const char* name) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(name) + " size mismatch");
  }
}

void ValidateConvolution(const std::vector<float>& input,
                         const std::vector<float>& weight, int batch_size,
                         int input_channels, int input_height, int input_width,
                         int output_channels, int kernel_height,
                         int kernel_width) {
  CheckedProduct({batch_size, input_channels, input_height, input_width},
                 "convolution");
  CheckedProduct({output_channels, kernel_height, kernel_width},
                 "convolution");
  if (kernel_height > input_height || kernel_width > input_width) {
    throw std::invalid_argument("convolution kernel must fit input");
  }
  RequireSize(input.size(),
              CheckedProduct(
                  {batch_size, input_channels, input_height, input_width},
                  "convolution"),
              "convolution input");
  RequireSize(weight.size(),
              CheckedProduct({output_channels, input_channels, kernel_height,
                              kernel_width},
                             "convolution"),
              "convolution weight");
}

std::size_t InputIndex(int n, int channel, int row, int column,
                       int channels, int height, int width) {
  return ((static_cast<std::size_t>(n) * channels + channel) * height + row) *
             width +
         column;
}

std::size_t WeightIndex(int output_channel, int input_channel, int kernel_row,
                        int kernel_column, int input_channels,
                        int kernel_height, int kernel_width) {
  return ((static_cast<std::size_t>(output_channel) * input_channels +
           input_channel) *
              kernel_height +
          kernel_row) *
             kernel_width +
         kernel_column;
}

std::size_t OutputIndex(int n, int channel, int row, int column, int channels,
                        int height, int width) {
  return ((static_cast<std::size_t>(n) * channels + channel) * height + row) *
             width +
         column;
}

void ValidateLinear(const std::vector<float>& input,
                    const std::vector<float>& weight, int batch_size,
                    int input_features, int output_features) {
  RequireSize(input.size(), CheckedProduct({batch_size, input_features},
                                           "linear"),
              "linear input");
  RequireSize(weight.size(), CheckedProduct({output_features, input_features},
                                            "linear"),
              "linear weight");
}

}  // namespace

std::vector<float> ConvolutionForward(
    const std::vector<float>& input, const std::vector<float>& weight,
    const std::vector<float>& bias, int batch_size, int input_channels,
    int input_height, int input_width, int output_channels,
    int kernel_height, int kernel_width) {
  ValidateConvolution(input, weight, batch_size, input_channels, input_height,
                      input_width, output_channels, kernel_height,
                      kernel_width);
  RequireSize(bias.size(), static_cast<std::size_t>(output_channels),
              "convolution bias");
  const int output_height = input_height - kernel_height + 1;
  const int output_width = input_width - kernel_width + 1;
  std::vector<float> output(
      CheckedProduct({batch_size, output_channels, output_height, output_width},
                     "convolution output"));

  for (int n = 0; n < batch_size; ++n) {
    for (int output_channel = 0; output_channel < output_channels;
         ++output_channel) {
      for (int output_row = 0; output_row < output_height; ++output_row) {
        for (int output_column = 0; output_column < output_width;
             ++output_column) {
          float sum = 0.0F;
          for (int input_channel = 0; input_channel < input_channels;
               ++input_channel) {
            for (int kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
              for (int kernel_column = 0; kernel_column < kernel_width;
                   ++kernel_column) {
                sum += input[InputIndex(
                           n, input_channel, output_row + kernel_row,
                           output_column + kernel_column, input_channels,
                           input_height, input_width)] *
                       weight[WeightIndex(output_channel, input_channel,
                                          kernel_row, kernel_column,
                                          input_channels, kernel_height,
                                          kernel_width)];
              }
            }
          }
          output[OutputIndex(n, output_channel, output_row, output_column,
                             output_channels, output_height, output_width)] =
              sum + bias[output_channel];
        }
      }
    }
  }
  return output;
}

ConvolutionGradients ConvolutionBackward(
    const std::vector<float>& input, const std::vector<float>& weight,
    const std::vector<float>& output_gradient, int batch_size,
    int input_channels, int input_height, int input_width,
    int output_channels, int kernel_height, int kernel_width) {
  ValidateConvolution(input, weight, batch_size, input_channels, input_height,
                      input_width, output_channels, kernel_height,
                      kernel_width);
  const int output_height = input_height - kernel_height + 1;
  const int output_width = input_width - kernel_width + 1;
  RequireSize(output_gradient.size(),
              CheckedProduct({batch_size, output_channels, output_height,
                              output_width},
                             "convolution output gradient"),
              "convolution output gradient");
  ConvolutionGradients gradients{
      std::vector<float>(input.size(), 0.0F),
      std::vector<float>(weight.size(), 0.0F),
      std::vector<float>(static_cast<std::size_t>(output_channels), 0.0F)};

  for (int n = 0; n < batch_size; ++n) {
    for (int input_channel = 0; input_channel < input_channels;
         ++input_channel) {
      for (int input_row = 0; input_row < input_height; ++input_row) {
        for (int input_column = 0; input_column < input_width; ++input_column) {
          float sum = 0.0F;
          for (int output_channel = 0; output_channel < output_channels;
               ++output_channel) {
            for (int kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
              const int output_row = input_row - kernel_row;
              if (output_row < 0 || output_row >= output_height) {
                continue;
              }
              for (int kernel_column = 0; kernel_column < kernel_width;
                   ++kernel_column) {
                const int output_column = input_column - kernel_column;
                if (output_column < 0 || output_column >= output_width) {
                  continue;
                }
                sum += output_gradient[OutputIndex(
                           n, output_channel, output_row, output_column,
                           output_channels, output_height, output_width)] *
                       weight[WeightIndex(output_channel, input_channel,
                                          kernel_row, kernel_column,
                                          input_channels, kernel_height,
                                          kernel_width)];
              }
            }
          }
          gradients.input[InputIndex(n, input_channel, input_row, input_column,
                                     input_channels, input_height,
                                     input_width)] = sum;
        }
      }
    }
  }

  for (int output_channel = 0; output_channel < output_channels;
       ++output_channel) {
    for (int input_channel = 0; input_channel < input_channels;
         ++input_channel) {
      for (int kernel_row = 0; kernel_row < kernel_height; ++kernel_row) {
        for (int kernel_column = 0; kernel_column < kernel_width;
             ++kernel_column) {
          float sum = 0.0F;
          for (int n = 0; n < batch_size; ++n) {
            for (int output_row = 0; output_row < output_height; ++output_row) {
              for (int output_column = 0; output_column < output_width;
                   ++output_column) {
                sum += input[InputIndex(
                           n, input_channel, output_row + kernel_row,
                           output_column + kernel_column, input_channels,
                           input_height, input_width)] *
                       output_gradient[OutputIndex(
                           n, output_channel, output_row, output_column,
                           output_channels, output_height, output_width)];
              }
            }
          }
          gradients.weight[WeightIndex(
              output_channel, input_channel, kernel_row, kernel_column,
              input_channels, kernel_height, kernel_width)] = sum;
        }
      }
    }

    float sum = 0.0F;
    for (int n = 0; n < batch_size; ++n) {
      for (int output_row = 0; output_row < output_height; ++output_row) {
        for (int output_column = 0; output_column < output_width;
             ++output_column) {
          sum += output_gradient[OutputIndex(
              n, output_channel, output_row, output_column, output_channels,
              output_height, output_width)];
        }
      }
    }
    gradients.bias[output_channel] = sum;
  }
  return gradients;
}

std::vector<float> ReluForward(const std::vector<float>& input) {
  std::vector<float> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] > 0.0F ? input[index] : 0.0F;
  }
  return output;
}

std::vector<float> ReluBackward(
    const std::vector<float>& forward_input,
    const std::vector<float>& output_gradient) {
  RequireSize(output_gradient.size(), forward_input.size(),
              "ReLU output gradient");
  std::vector<float> input_gradient(forward_input.size());
  for (std::size_t index = 0; index < forward_input.size(); ++index) {
    input_gradient[index] =
        forward_input[index] > 0.0F ? output_gradient[index] : 0.0F;
  }
  return input_gradient;
}

MaxPoolResult MaxPoolForward(const std::vector<float>& input, int batch_size,
                             int channels, int input_height, int input_width) {
  RequireSize(input.size(),
              CheckedProduct({batch_size, channels, input_height, input_width},
                             "max pool"),
              "max pool input");
  if (input_height % 2 != 0 || input_width % 2 != 0) {
    throw std::invalid_argument("max pool spatial dimensions must be even");
  }
  const int output_height = input_height / 2;
  const int output_width = input_width / 2;
  const std::size_t output_size = CheckedProduct(
      {batch_size, channels, output_height, output_width}, "max pool output");
  MaxPoolResult result{std::vector<float>(output_size),
                       std::vector<std::uint8_t>(output_size)};

  for (int n = 0; n < batch_size; ++n) {
    for (int channel = 0; channel < channels; ++channel) {
      for (int output_row = 0; output_row < output_height; ++output_row) {
        for (int output_column = 0; output_column < output_width;
             ++output_column) {
          const int input_row = output_row * 2;
          const int input_column = output_column * 2;
          std::uint8_t winner = 0;
          float maximum = input[InputIndex(n, channel, input_row, input_column,
                                           channels, input_height,
                                           input_width)];
          for (std::uint8_t offset = 1; offset < 4; ++offset) {
            const float candidate = input[InputIndex(
                n, channel, input_row + offset / 2,
                input_column + offset % 2, channels, input_height,
                input_width)];
            if (candidate > maximum) {
              maximum = candidate;
              winner = offset;
            }
          }
          const std::size_t output_index = OutputIndex(
              n, channel, output_row, output_column, channels, output_height,
              output_width);
          result.output[output_index] = maximum;
          result.winner_offsets[output_index] = winner;
        }
      }
    }
  }
  return result;
}

std::vector<float> MaxPoolBackward(
    const std::vector<float>& output_gradient,
    const std::vector<std::uint8_t>& winner_offsets, int batch_size,
    int channels, int input_height, int input_width) {
  const std::size_t input_size = CheckedProduct(
      {batch_size, channels, input_height, input_width}, "max pool backward");
  if (input_height % 2 != 0 || input_width % 2 != 0) {
    throw std::invalid_argument("max pool spatial dimensions must be even");
  }
  const int output_height = input_height / 2;
  const int output_width = input_width / 2;
  const std::size_t output_size = CheckedProduct(
      {batch_size, channels, output_height, output_width}, "max pool backward");
  RequireSize(output_gradient.size(), output_size,
              "max pool output gradient");
  RequireSize(winner_offsets.size(), output_size, "max pool winner offsets");
  std::vector<float> input_gradient(input_size, 0.0F);

  for (int n = 0; n < batch_size; ++n) {
    for (int channel = 0; channel < channels; ++channel) {
      for (int output_row = 0; output_row < output_height; ++output_row) {
        for (int output_column = 0; output_column < output_width;
             ++output_column) {
          const std::size_t output_index = OutputIndex(
              n, channel, output_row, output_column, channels, output_height,
              output_width);
          const std::uint8_t offset = winner_offsets[output_index];
          if (offset >= 4) {
            throw std::invalid_argument("max pool winner offset out of range");
          }
          input_gradient[InputIndex(n, channel, output_row * 2 + offset / 2,
                                    output_column * 2 + offset % 2, channels,
                                    input_height, input_width)] =
              output_gradient[output_index];
        }
      }
    }
  }
  return input_gradient;
}

std::vector<float> LinearForward(const std::vector<float>& input,
                                 const std::vector<float>& weight,
                                 const std::vector<float>& bias,
                                 int batch_size, int input_features,
                                 int output_features) {
  ValidateLinear(input, weight, batch_size, input_features, output_features);
  RequireSize(bias.size(), static_cast<std::size_t>(output_features),
              "linear bias");
  std::vector<float> output(
      CheckedProduct({batch_size, output_features}, "linear output"));
  for (int n = 0; n < batch_size; ++n) {
    for (int output_feature = 0; output_feature < output_features;
         ++output_feature) {
      float sum = 0.0F;
      for (int input_feature = 0; input_feature < input_features;
           ++input_feature) {
        sum += input[static_cast<std::size_t>(n) * input_features +
                     input_feature] *
               weight[static_cast<std::size_t>(output_feature) *
                          input_features +
                      input_feature];
      }
      output[static_cast<std::size_t>(n) * output_features + output_feature] =
          sum + bias[output_feature];
    }
  }
  return output;
}

LinearGradients LinearBackward(const std::vector<float>& input,
                               const std::vector<float>& weight,
                               const std::vector<float>& output_gradient,
                               int batch_size, int input_features,
                               int output_features) {
  ValidateLinear(input, weight, batch_size, input_features, output_features);
  RequireSize(output_gradient.size(),
              CheckedProduct({batch_size, output_features},
                             "linear output gradient"),
              "linear output gradient");
  LinearGradients gradients{std::vector<float>(input.size()),
                            std::vector<float>(weight.size()),
                            std::vector<float>(output_features)};

  for (int n = 0; n < batch_size; ++n) {
    for (int input_feature = 0; input_feature < input_features;
         ++input_feature) {
      float sum = 0.0F;
      for (int output_feature = 0; output_feature < output_features;
           ++output_feature) {
        sum += output_gradient[static_cast<std::size_t>(n) * output_features +
                               output_feature] *
               weight[static_cast<std::size_t>(output_feature) *
                          input_features +
                      input_feature];
      }
      gradients.input[static_cast<std::size_t>(n) * input_features +
                      input_feature] = sum;
    }
  }

  for (int output_feature = 0; output_feature < output_features;
       ++output_feature) {
    for (int input_feature = 0; input_feature < input_features;
         ++input_feature) {
      float sum = 0.0F;
      for (int n = 0; n < batch_size; ++n) {
        sum += input[static_cast<std::size_t>(n) * input_features +
                     input_feature] *
               output_gradient[static_cast<std::size_t>(n) * output_features +
                               output_feature];
      }
      gradients.weight[static_cast<std::size_t>(output_feature) *
                           input_features +
                       input_feature] = sum;
    }

    float sum = 0.0F;
    for (int n = 0; n < batch_size; ++n) {
      sum += output_gradient[static_cast<std::size_t>(n) * output_features +
                             output_feature];
    }
    gradients.bias[output_feature] = sum;
  }
  return gradients;
}

SoftmaxCrossEntropyResult SoftmaxCrossEntropy(
    const std::vector<float>& logits, const std::vector<std::uint8_t>& labels,
    int batch_size, int class_count) {
  RequireSize(logits.size(),
              CheckedProduct({batch_size, class_count}, "softmax"),
              "softmax logits");
  RequireSize(labels.size(), static_cast<std::size_t>(batch_size),
              "softmax labels");
  SoftmaxCrossEntropyResult result{
      std::vector<float>(logits.size()), std::vector<float>(batch_size), 0.0F,
      std::vector<float>(logits.size())};
  const float inverse_batch = 1.0F / static_cast<float>(batch_size);

  for (int n = 0; n < batch_size; ++n) {
    if (labels[n] >= class_count) {
      throw std::invalid_argument("softmax label out of range");
    }
    const std::size_t row = static_cast<std::size_t>(n) * class_count;
    float maximum = logits[row];
    for (int class_index = 1; class_index < class_count; ++class_index) {
      maximum = std::max(maximum, logits[row + class_index]);
    }
    float exponential_sum = 0.0F;
    for (int class_index = 0; class_index < class_count; ++class_index) {
      const float exponential =
          std::exp(logits[row + class_index] - maximum);
      result.probabilities[row + class_index] = exponential;
      exponential_sum += exponential;
    }
    for (int class_index = 0; class_index < class_count; ++class_index) {
      const float probability =
          result.probabilities[row + class_index] / exponential_sum;
      result.probabilities[row + class_index] = probability;
      result.logits_gradient[row + class_index] =
          (probability - (class_index == labels[n] ? 1.0F : 0.0F)) *
          inverse_batch;
    }
    result.per_sample_losses[n] =
        std::log(exponential_sum) - (logits[row + labels[n]] - maximum);
  }
  float loss_sum = 0.0F;
  for (float loss : result.per_sample_losses) {
    loss_sum += loss;
  }
  result.mean_loss = loss_sum * inverse_batch;
  return result;
}

void AdamWStep(std::vector<double>* parameters,
               const std::vector<double>& gradients,
               std::vector<double>* first_moments,
               std::vector<double>* second_moments,
               std::uint64_t global_step, double learning_rate, double beta1,
               double beta2, double epsilon, double weight_decay) {
  if (parameters == nullptr || first_moments == nullptr ||
      second_moments == nullptr) {
    throw std::invalid_argument("AdamW vectors must be non-null");
  }
  if (parameters == &gradients || first_moments == &gradients ||
      second_moments == &gradients || parameters == first_moments ||
      parameters == second_moments || first_moments == second_moments) {
    throw std::invalid_argument("AdamW vectors must not alias");
  }
  RequireSize(gradients.size(), parameters->size(), "AdamW gradients");
  RequireSize(first_moments->size(), parameters->size(),
              "AdamW first moments");
  RequireSize(second_moments->size(), parameters->size(),
              "AdamW second moments");
  if (global_step == 0 || !std::isfinite(learning_rate) ||
      learning_rate < 0.0 || !std::isfinite(beta1) || beta1 < 0.0 ||
      beta1 >= 1.0 || !std::isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0 ||
      !std::isfinite(epsilon) || epsilon <= 0.0 ||
      !std::isfinite(weight_decay) || weight_decay < 0.0) {
    throw std::invalid_argument("invalid AdamW step or hyperparameter");
  }

  const double inverse_bias_correction1 =
      1.0 / (1.0 - std::pow(beta1, static_cast<double>(global_step)));
  const double inverse_bias_correction2 =
      1.0 / (1.0 - std::pow(beta2, static_cast<double>(global_step)));
  for (std::size_t index = 0; index < parameters->size(); ++index) {
    const double gradient = gradients[index];
    (*first_moments)[index] =
        beta1 * (*first_moments)[index] + (1.0 - beta1) * gradient;
    (*second_moments)[index] = beta2 * (*second_moments)[index] +
                               (1.0 - beta2) * gradient * gradient;
    const double corrected_first =
        (*first_moments)[index] * inverse_bias_correction1;
    const double corrected_second =
        (*second_moments)[index] * inverse_bias_correction2;
    const double previous_parameter = (*parameters)[index];
    (*parameters)[index] =
        previous_parameter -
        learning_rate *
            (corrected_first / (std::sqrt(corrected_second) + epsilon) +
             weight_decay * previous_parameter);
  }
}

}  // namespace cpu_reference
