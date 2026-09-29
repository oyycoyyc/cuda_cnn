#define CUDA_LENET_ENABLE_TEST_HOOKS

#include "train.h"

#include "checkpoint.h"
#include "cpu_reference.h"
#include "cuda_check.h"
#include "layers.h"
#include "lenet.h"
#include "random.h"
#include "tensor.h"
#include "training_data.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::size_t kImagePixels = 28 * 28;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void PutU32(std::ostream& output, std::uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) {
    output.put(static_cast<char>((value >> (byte * 8)) & 0xffU));
  }
}

struct FixtureFiles {
  std::vector<std::string> paths;

  ~FixtureFiles() {
    for (const std::string& path : paths) {
      std::remove(path.c_str());
    }
  }

  std::string Path(const std::string& name) {
    const std::string path = "build/workflow_" + name;
    paths.push_back(path);
    return path;
  }
};

void WriteDataset(const std::string& path,
                  const std::vector<std::uint8_t>& images,
                  const std::vector<std::uint8_t>& labels) {
  Require(!labels.empty(), "dataset fixture must not be empty");
  Require(images.size() == labels.size() * kImagePixels,
          "dataset fixture image size mismatch");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  Require(static_cast<bool>(output), "could not create dataset fixture");
  const char magic[8] = {'M', 'N', 'I', 'S', 'T', 'C', '1', '\0'};
  output.write(magic, sizeof(magic));
  PutU32(output, 1);
  PutU32(output, static_cast<std::uint32_t>(labels.size()));
  PutU32(output, 28);
  PutU32(output, 28);
  output.write(reinterpret_cast<const char*>(images.data()),
               static_cast<std::streamsize>(images.size()));
  output.write(reinterpret_cast<const char*>(labels.data()),
               static_cast<std::streamsize>(labels.size()));
  Require(static_cast<bool>(output), "could not write dataset fixture");
}

std::vector<std::uint8_t> PatternImages(std::uint32_t count) {
  std::vector<std::uint8_t> images(static_cast<std::size_t>(count) *
                                   kImagePixels);
  for (std::uint32_t sample = 0; sample < count; ++sample) {
    for (std::size_t pixel = 0; pixel < kImagePixels; ++pixel) {
      images[static_cast<std::size_t>(sample) * kImagePixels + pixel] =
          static_cast<std::uint8_t>((sample * 29 + pixel * 17 + 11) % 256);
    }
  }
  return images;
}

std::vector<std::uint8_t> CyclicLabels(std::uint32_t count) {
  std::vector<std::uint8_t> labels(count);
  for (std::uint32_t sample = 0; sample < count; ++sample) {
    labels[sample] = static_cast<std::uint8_t>(sample % 10);
  }
  return labels;
}

Checkpoint ZeroCheckpoint(float accuracy = 0.0F,
                          std::uint32_t epoch = 1) {
  Checkpoint checkpoint{{epoch, accuracy, 0.1307F, 0.3081F},
                        CreateLenetParameters()};
  for (ParameterTensor& tensor : checkpoint.parameters) {
    std::fill(tensor.values.begin(), tensor.values.end(), 0.0F);
  }
  return checkpoint;
}

std::vector<float> CpuLogits(const std::vector<std::uint8_t>& image,
                             const ParameterSet& parameters) {
  std::vector<float> normalized(kImagePixels);
  for (std::size_t pixel = 0; pixel < image.size(); ++pixel) {
    normalized[pixel] =
        (static_cast<float>(image[pixel]) / 255.0F - 0.1307F) / 0.3081F;
  }
  const std::vector<float> conv1 = cpu_reference::ConvolutionForward(
      normalized, parameters[0].values, parameters[1].values, 1, 1, 28, 28,
      6, 5, 5);
  const std::vector<float> relu1 = cpu_reference::ReluForward(conv1);
  const cpu_reference::MaxPoolResult pool1 =
      cpu_reference::MaxPoolForward(relu1, 1, 6, 24, 24);
  const std::vector<float> conv2 = cpu_reference::ConvolutionForward(
      pool1.output, parameters[2].values, parameters[3].values, 1, 6, 12, 12,
      16, 5, 5);
  const std::vector<float> relu2 = cpu_reference::ReluForward(conv2);
  const cpu_reference::MaxPoolResult pool2 =
      cpu_reference::MaxPoolForward(relu2, 1, 16, 8, 8);
  const std::vector<float> fc1 = cpu_reference::LinearForward(
      pool2.output, parameters[4].values, parameters[5].values, 1, 256, 120);
  const std::vector<float> relu3 = cpu_reference::ReluForward(fc1);
  const std::vector<float> fc2 = cpu_reference::LinearForward(
      relu3, parameters[6].values, parameters[7].values, 1, 120, 84);
  const std::vector<float> relu4 = cpu_reference::ReluForward(fc2);
  return cpu_reference::LinearForward(relu4, parameters[8].values,
                                      parameters[9].values, 1, 84, 10);
}

double Field(const std::string& record, const std::string& name) {
  const std::string marker = name + "=";
  const std::string::size_type begin = record.find(marker);
  Require(begin != std::string::npos, "missing output field " + name);
  const std::string::size_type value_begin = begin + marker.size();
  const std::string::size_type end = record.find(' ', value_begin);
  return std::stod(record.substr(value_begin, end - value_begin));
}

std::vector<double> CsvField(const std::string& record,
                             const std::string& name) {
  const std::string marker = name + "=";
  const std::string::size_type begin = record.find(marker);
  Require(begin != std::string::npos, "missing output field " + name);
  const std::string::size_type value_begin = begin + marker.size();
  const std::string::size_type end = record.find(' ', value_begin);
  std::istringstream input(record.substr(value_begin, end - value_begin));
  std::vector<double> values;
  std::string value;
  while (std::getline(input, value, ',')) {
    values.push_back(std::stod(value));
  }
  return values;
}

template <typename Function>
void RequireThrowsContaining(Function function, const std::string& text) {
  try {
    function();
  } catch (const std::exception& error) {
    Require(std::string(error.what()).find(text) != std::string::npos,
            "exception did not contain " + text + ": " + error.what());
    return;
  }
  throw std::runtime_error("expected exception containing " + text);
}

TrainOptions TrainFixtureOptions(const std::string& train,
                                 const std::string& test,
                                 const std::string& output,
                                 std::uint32_t epochs,
                                 std::uint32_t batch_size) {
  TrainOptions options;
  options.train_path = train;
  options.test_path = test;
  options.output_path = output;
  options.epochs = epochs;
  options.batch_size = batch_size;
  options.seed = UINT64_C(1337);
  options.device = 0;
  options.allow_nonstandard_count = true;
  return options;
}

EvaluateOptions EvaluateFixtureOptions(const std::string& data,
                                       const std::string& weights,
                                       float minimum_accuracy) {
  EvaluateOptions options;
  options.data_path = data;
  options.weights_path = weights;
  options.minimum_accuracy = minimum_accuracy;
  options.device = 0;
  options.allow_nonstandard_count = true;
  return options;
}

void CaseSchedule() {
  Require(LearningRateForEpoch(1) == 1.0e-3F, "epoch 1 learning rate");
  Require(LearningRateForEpoch(12) == 1.0e-3F, "epoch 12 learning rate");
  Require(LearningRateForEpoch(13) == 1.0e-4F, "epoch 13 learning rate");
  Require(LearningRateForEpoch(17) == 1.0e-4F, "epoch 17 learning rate");
  Require(LearningRateForEpoch(18) == 1.0e-5F, "epoch 18 learning rate");
  Require(LearningRateForEpoch(20) == 1.0e-5F, "epoch 20 learning rate");
  RequireThrowsContaining([] { LearningRateForEpoch(0); }, "one-based");
}

void CaseValidation() {
  FixtureFiles files;
  std::ostringstream output;
  std::ostringstream error;
  EvaluateOptions missing =
      EvaluateFixtureOptions("build/does_not_exist.bin", "also_missing.bin", 0);
  RequireThrowsContaining(
      [&] { RunEvaluate(missing, output, error); }, "does_not_exist.bin");

  const std::string data = files.Path("validation_data.bin");
  const std::string weights = files.Path("validation_weights.bin");
  WriteDataset(data, PatternImages(2), CyclicLabels(2));
  SaveCheckpoint(weights, ZeroCheckpoint());

  const std::string malformed_data = files.Path("malformed_data.bin");
  {
    std::ofstream malformed(malformed_data,
                            std::ios::binary | std::ios::trunc);
    malformed << "not a dataset";
  }
  EvaluateOptions malformed_dataset =
      EvaluateFixtureOptions(malformed_data, weights, 0);
  RequireThrowsContaining(
      [&] { RunEvaluate(malformed_dataset, output, error); }, malformed_data);

  const std::string malformed_weights = files.Path("malformed_weights.bin");
  {
    std::ofstream malformed(malformed_weights,
                            std::ios::binary | std::ios::trunc);
    malformed << "not a checkpoint";
  }
  EvaluateOptions malformed_checkpoint =
      EvaluateFixtureOptions(data, malformed_weights, 0);
  RequireThrowsContaining(
      [&] { RunEvaluate(malformed_checkpoint, output, error); },
      malformed_weights);

  EvaluateOptions wrong_count = EvaluateFixtureOptions(data, weights, 0);
  wrong_count.allow_nonstandard_count = false;
  RequireThrowsContaining(
      [&] { RunEvaluate(wrong_count, output, error); }, "10000");

  InferOptions bad_index{data, weights, 2, 0};
  RequireThrowsContaining(
      [&] { RunInfer(bad_index, output, error); }, "index");

  TrainOptions missing_parent = TrainFixtureOptions(
      data, data, "build/workflow_missing_parent/model.bin", 1, 1);
  missing_parent.device = std::numeric_limits<int>::max();
  RequireThrowsContaining(
      [&] { RunTrain(missing_parent, output, error); }, "parent");

  InferOptions unavailable{data, weights, 0, std::numeric_limits<int>::max()};
  RequireThrowsContaining(
      [&] { RunInfer(unavailable, output, error); }, "device");
}

void CaseBatch17() {
  FixtureFiles files;
  const std::string train = files.Path("batch17_train.bin");
  const std::string test = files.Path("batch17_test.bin");
  const std::string weights = files.Path("batch17_weights.bin");
  WriteDataset(train, PatternImages(22), CyclicLabels(22));
  WriteDataset(test, PatternImages(19), CyclicLabels(19));
  const TrainOptions options =
      TrainFixtureOptions(train, test, weights, 1, 17);
  std::ostringstream output;
  std::ostringstream error;
  Require(RunTrain(options, output, error) == kSuccess,
          "batch17 training failed: " + error.str());
  const std::string records = output.str();
  Require(records.find("event=epoch epoch=1") != std::string::npos,
          "missing epoch summary");
  Require(std::isfinite(Field(records, "train_loss")),
          "training loss is not finite");
  const Checkpoint checkpoint = LoadCheckpoint(weights);
  Require(checkpoint.metadata.best_epoch == 1,
          "first update/checkpoint epoch must be one-based");
  for (const ParameterTensor& tensor : checkpoint.parameters) {
    for (float value : tensor.values) {
      Require(std::isfinite(value), "checkpoint parameter is non-finite");
    }
  }
}

void CaseGlobalStepStartsAtOne() {
  FixtureFiles files;
  const std::vector<std::uint8_t> images = PatternImages(2);
  const std::vector<std::uint8_t> labels{{3, 8}};
  const std::string train = files.Path("global_step_train.bin");
  const std::string test = files.Path("global_step_test.bin");
  const std::string weights = files.Path("global_step_weights.bin");
  WriteDataset(train, images, labels);
  WriteDataset(test, images, labels);
  std::ostringstream output;
  std::ostringstream error;
  Require(RunTrain(TrainFixtureOptions(train, test, weights, 1, 1), output,
                   error) == kSuccess,
          "single-update workflow failed: " + error.str());
  const Checkpoint actual = LoadCheckpoint(weights);

  const DatasetSplit split = MakeWorkflowSplit(2, 1337, true);
  const std::uint32_t original = split.training_indices.front();
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  try {
    DeviceBuffer<std::uint8_t> device_image(kImagePixels);
    DeviceBuffer<std::uint8_t> device_label(1);
    DeviceBuffer<std::uint32_t> device_index(1);
    DeviceBuffer<float> normalized(kImagePixels);
    DeviceBuffer<float> probabilities(10);
    DeviceBuffer<float> losses(1);
    DeviceBuffer<float> mean_loss(1);
    DeviceBuffer<float> logits_gradient(10);
    LeNet model(1, 1337, stream);
    CUDA_CHECK(cudaMemcpyAsync(
        device_image.get(),
        images.data() + static_cast<std::size_t>(original) * kImagePixels,
        kImagePixels, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_label.get(), &labels[original], 1,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(device_index.get(), &original, sizeof(original),
                               cudaMemcpyHostToDevice, stream));
    LaunchNormalizeTranslate(device_image.get(), device_index.get(),
                             normalized.get(), 1, 1337, 1, true, stream);
    const float* logits = model.Forward(normalized.get(), 1);
    LaunchSoftmaxCrossEntropy(logits, device_label.get(), probabilities.get(),
                              losses.get(), mean_loss.get(),
                              logits_gradient.get(), 1, 10, stream);
    model.Backward(logits_gradient.get(), 1);
    model.AdamWStep(1, 1.0e-3F,
                    AdamWConfig{0.9F, 0.999F, 1.0e-8F, 1.0e-4F});
    const ParameterSet expected = model.ExportParameters();
    Require(expected.size() == actual.parameters.size(),
            "single-update parameter schema mismatch");
    for (std::size_t tensor = 0; tensor < expected.size(); ++tensor) {
      Require(expected[tensor].values.size() ==
                  actual.parameters[tensor].values.size(),
              "single-update parameter extent mismatch");
      for (std::size_t value = 0; value < expected[tensor].values.size();
           ++value) {
        Require(expected[tensor].values[value] ==
                    actual.parameters[tensor].values[value],
                "workflow did not use global Adam step one");
      }
    }
  } catch (...) {
    static_cast<void>(cudaStreamDestroy(stream));
    throw;
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void CaseStrictCheckpointTie() {
  FixtureFiles files;
  const std::uint32_t count = 52;
  std::vector<std::uint8_t> images(count * kImagePixels, 37);
  std::vector<std::uint8_t> labels(count, 0);
  const DatasetSplit split = MakeWorkflowSplit(count, 1337, true);
  Require(split.validation_indices.size() == 10,
          "tie fixture must have ten validation rows");
  for (std::size_t index = 0; index < split.validation_indices.size(); ++index) {
    labels[split.validation_indices[index]] = static_cast<std::uint8_t>(index);
  }
  const std::string train = files.Path("tie_train.bin");
  const std::string test = files.Path("tie_test.bin");
  const std::string weights = files.Path("tie_weights.bin");
  WriteDataset(train, images, labels);
  WriteDataset(test, std::vector<std::uint8_t>(10 * kImagePixels, 37),
               CyclicLabels(10));
  std::ostringstream output;
  std::ostringstream error;
  Require(RunTrain(TrainFixtureOptions(train, test, weights, 2, 17), output,
                   error) == kSuccess,
          "tie training failed: " + error.str());
  const Checkpoint checkpoint = LoadCheckpoint(weights);
  Require(checkpoint.metadata.best_epoch == 1,
          "equal validation accuracy replaced the earlier checkpoint");
  Require(std::fabs(checkpoint.metadata.validation_accuracy - 0.1F) < 1.0e-6F,
          "tie fixture validation accuracy must be exact");

  std::ostringstream evaluation;
  Require(RunEvaluate(EvaluateFixtureOptions(test, weights, 0), evaluation,
                      error) == kSuccess,
          "reloaded checkpoint evaluation failed");
  Require(std::fabs(Field(output.str(), "final_test_accuracy") -
                    Field(evaluation.str(), "accuracy")) < 1.0e-6,
          "final test did not use the persisted best checkpoint");
}

void CaseOverfit32() {
  FixtureFiles files;
  const std::uint32_t count = 40;
  std::vector<std::uint8_t> images(count * kImagePixels, 0);
  std::vector<std::uint8_t> labels(count);
  for (std::uint32_t sample = 0; sample < count; ++sample) {
    labels[sample] = static_cast<std::uint8_t>(sample % 4);
    const std::size_t base = static_cast<std::size_t>(sample) * kImagePixels;
    const int quadrant = labels[sample];
    const int start_y = quadrant >= 2 ? 14 : 2;
    const int start_x = (quadrant % 2) != 0 ? 14 : 2;
    for (int y = start_y; y < start_y + 10; ++y) {
      for (int x = start_x; x < start_x + 10; ++x) {
        images[base + static_cast<std::size_t>(y) * 28 + x] = 255;
      }
    }
  }
  const DatasetSplit split = MakeWorkflowSplit(count, 1337, true);
  Require(split.training_indices.size() == 32,
          "overfit fixture must contain 32 training rows");
  std::vector<std::uint8_t> training_images(32 * kImagePixels);
  std::vector<std::uint8_t> training_labels(32);
  for (std::size_t packed = 0; packed < split.training_indices.size(); ++packed) {
    const std::uint32_t original = split.training_indices[packed];
    std::copy_n(images.begin() + static_cast<std::size_t>(original) *
                                   kImagePixels,
                kImagePixels,
                training_images.begin() + packed * kImagePixels);
    training_labels[packed] = labels[original];
  }
  const std::string train = files.Path("overfit_train.bin");
  const std::string test = files.Path("overfit_test.bin");
  const std::string subset = files.Path("overfit_subset.bin");
  const std::string weights = files.Path("overfit_weights.bin");
  WriteDataset(train, images, labels);
  WriteDataset(test, images, labels);
  WriteDataset(subset, training_images, training_labels);
  std::ostringstream output;
  std::ostringstream error;
  Require(RunTrain(TrainFixtureOptions(train, test, weights, 200, 32), output,
                   error) == kSuccess,
          "overfit training failed: " + error.str());
  const std::string records = output.str();
  const std::string first_marker = "event=epoch epoch=1 ";
  const std::string last_marker = "event=epoch epoch=200 ";
  const std::size_t first = records.find(first_marker);
  const std::size_t last = records.find(last_marker);
  Require(first != std::string::npos && last != std::string::npos,
          "missing overfit epoch records");
  const float initial_loss = static_cast<float>(Field(records.substr(first),
                                                       "train_loss"));
  const float final_loss = static_cast<float>(Field(records.substr(last),
                                                     "train_loss"));
  Require(final_loss <= initial_loss * 0.2F,
          "overfit loss did not fall to 20 percent");
  std::ostringstream evaluation;
  Require(RunEvaluate(EvaluateFixtureOptions(subset, weights, 0.95F),
                      evaluation, error) == kSuccess,
          "overfit training accuracy was below 95 percent");
}

void CaseOneEpochCheckpoint() {
  FixtureFiles files;
  const std::string train = files.Path("one_epoch_train.bin");
  const std::string test = files.Path("one_epoch_test.bin");
  const std::string weights = files.Path("one_epoch_weights.bin");
  WriteDataset(train, PatternImages(1280), CyclicLabels(1280));
  WriteDataset(test, PatternImages(257), CyclicLabels(257));
  const DatasetSplit split = MakeWorkflowSplit(1280, 1337, true);
  Require(split.training_indices.size() == 1024 &&
              split.validation_indices.size() == 256,
          "one-epoch fixture did not produce the required 1024/256 split");
  std::ostringstream output;
  std::ostringstream error;
  Require(RunTrain(TrainFixtureOptions(train, test, weights, 1, 128), output,
                   error) == kSuccess,
          "one-epoch workflow failed: " + error.str());
  const Checkpoint checkpoint = LoadCheckpoint(weights);
  Require(checkpoint.metadata.best_epoch == 1,
          "one-epoch checkpoint metadata is invalid");
  std::ostringstream evaluation;
  Require(RunEvaluate(EvaluateFixtureOptions(test, weights, 0), evaluation,
                      error) == kSuccess,
          "one-epoch checkpoint could not be reloaded");
  Require(Field(evaluation.str(), "samples") == 257,
          "partial evaluation batch lost samples");
}

void CaseEvaluate() {
  FixtureFiles files;
  const std::string one_data = files.Path("evaluate_one.bin");
  const std::string many_data = files.Path("evaluate_many.bin");
  const std::string weights = files.Path("evaluate_weights.bin");
  WriteDataset(one_data, PatternImages(1), std::vector<std::uint8_t>(1, 0));
  std::vector<std::uint8_t> labels(257, 1);
  std::fill(labels.begin(), labels.begin() + 129, 0);
  WriteDataset(many_data, PatternImages(257), labels);
  SaveCheckpoint(weights, ZeroCheckpoint(0.5F));
  std::ostringstream error;

  const std::size_t before_one =
      DeviceBufferSuccessfulAllocationEventCountForTests();
  std::ostringstream one_output;
  Require(RunEvaluate(EvaluateFixtureOptions(one_data, weights, 1.0F),
                      one_output, error) == kSuccess,
          "one-batch threshold equality failed");
  const std::size_t one_allocations =
      DeviceBufferSuccessfulAllocationEventCountForTests() - before_one;
  Require(Field(one_output.str(), "accuracy") == 1.0,
          "warm-up sample was not counted exactly once");
  Require(Field(one_output.str(), "mean_forward_ms") == 0.0,
          "one-batch warm-up timing must be explicit zero");
  Require(Field(one_output.str(), "images_per_second") == 0.0,
          "one-batch throughput must be explicit zero");

  const float exact_accuracy = 129.0F / 257.0F;
  const std::size_t before_many =
      DeviceBufferSuccessfulAllocationEventCountForTests();
  std::ostringstream many_output;
  Require(RunEvaluate(EvaluateFixtureOptions(many_data, weights,
                                             exact_accuracy),
                      many_output, error) == kSuccess,
          "exact threshold equality did not pass");
  const std::size_t many_allocations =
      DeviceBufferSuccessfulAllocationEventCountForTests() - before_many;
  Require(one_allocations == many_allocations,
          "device allocation count depends on batch-loop iterations");
  Require(Field(many_output.str(), "samples") == 257,
          "evaluation did not count every partial-batch sample");
  Require(std::fabs(Field(many_output.str(), "accuracy") - exact_accuracy) <
              1.0e-6,
          "evaluation accuracy double-counted or omitted warm-up data");
  Require(std::isfinite(Field(many_output.str(), "mean_forward_ms")) &&
              Field(many_output.str(), "mean_forward_ms") >= 0.0,
          "timing mean is invalid");
  Require(std::isfinite(Field(many_output.str(), "images_per_second")) &&
              Field(many_output.str(), "images_per_second") >= 0.0,
          "throughput is invalid");

  std::ostringstream failed_output;
  Require(RunEvaluate(EvaluateFixtureOptions(
                          many_data, weights,
                          std::nextafter(exact_accuracy, 1.0F)),
                      failed_output, error) == kAcceptanceFailure,
          "below-threshold evaluation did not return exit code 3");
  Require(failed_output.str().find("status=fail") != std::string::npos,
          "failed threshold summary is missing");
}

void CaseInfer() {
  FixtureFiles files;
  const std::string data = files.Path("infer_data.bin");
  const std::string weights = files.Path("infer_weights.bin");
  const std::vector<std::uint8_t> images = PatternImages(3);
  const std::vector<std::uint8_t> labels{{4, 7, 2}};
  WriteDataset(data, images, labels);
  const Checkpoint checkpoint = ZeroCheckpoint();
  SaveCheckpoint(weights, checkpoint);
  InferOptions options{data, weights, 1, 0};
  std::ostringstream output;
  std::ostringstream error;
  Require(RunInfer(options, output, error) == kSuccess,
          "inference workflow failed: " + error.str());
  const std::string record = output.str();
  Require(record.find("event=infer index=1 ") != std::string::npos,
          "inference record prefix is unstable");
  const std::vector<double> logits = CsvField(record, "logits");
  const std::vector<double> probabilities = CsvField(record, "probabilities");
  Require(logits.size() == 10 && probabilities.size() == 10,
          "inference did not print ten logits and probabilities");
  const std::vector<std::uint8_t> image(
      images.begin() + kImagePixels, images.begin() + 2 * kImagePixels);
  const std::vector<float> reference = CpuLogits(image, checkpoint.parameters);
  double probability_sum = 0.0;
  for (std::size_t index = 0; index < 10; ++index) {
    Require(std::isfinite(logits[index]) &&
                std::fabs(logits[index] - reference[index]) <= 1.0e-5,
            "inference logits disagree with CPU reference");
    Require(std::isfinite(probabilities[index]) &&
                std::fabs(probabilities[index] - 0.1) <= 1.0e-6,
            "inference probability is unstable or incorrect");
    probability_sum += probabilities[index];
  }
  Require(std::fabs(probability_sum - 1.0) <= 1.0e-6,
          "inference probabilities do not sum to one");
  Require(Field(record, "prediction") == 0,
          "argmax tie did not select class zero");
  Require(Field(record, "label") == 7, "inference label was not copied");
}

struct NamedCase {
  const char* name;
  void (*function)();
};

const NamedCase kCases[] = {
    {"schedule", &CaseSchedule},
    {"validation", &CaseValidation},
    {"batch17", &CaseBatch17},
    {"global_step", &CaseGlobalStepStartsAtOne},
    {"strict_checkpoint_tie", &CaseStrictCheckpointTie},
    {"overfit32", &CaseOverfit32},
    {"one_epoch_checkpoint", &CaseOneEpochCheckpoint},
    {"evaluate", &CaseEvaluate},
    {"infer", &CaseInfer},
};

}  // namespace

int main(int argc, char** argv) {
  std::string selected;
  if (argc == 3 && std::string(argv[1]) == "--case") {
    selected = argv[2];
  } else if (argc != 1) {
    std::cerr << "usage: " << argv[0] << " [--case NAME]\n";
    return 2;
  }

  int failures = 0;
  bool found = selected.empty();
  for (const NamedCase& test : kCases) {
    if (!selected.empty() && selected != test.name) {
      continue;
    }
    found = true;
    try {
      test.function();
      std::cout << "event=workflow_test case=" << test.name
                << " status=pass\n";
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "workflow test failure: " << test.name << ": "
                << error.what() << '\n';
      std::cout << "event=workflow_test case=" << test.name
                << " status=fail\n";
    }
  }
  if (!found) {
    std::cerr << "unknown workflow test case: " << selected << '\n';
    return 2;
  }
  return failures == 0 ? 0 : 1;
}
