# Comment Review Checklist

Task 13 records the stable inventory below without claiming semantic review.
On the H20 host, inspect each item for every applicable shape/layout,
index/formula, race/synchronization, boundary, numerical-stability, ownership,
lifetime, aliasing, and error-behavior requirement. Change `[ ]` to `[x]` only
after that manual review; Task 14 requires `--require-reviewed`.

## Public Types And Declarations

- [ ] `public:include/checkpoint.h:Checkpoint`
- [ ] `public:include/checkpoint.h:CheckpointMetadata`
- [ ] `public:include/checkpoint.h:LoadCheckpoint`
- [ ] `public:include/checkpoint.h:SaveCheckpoint`
- [ ] `public:include/cli.h:CliOptions`
- [ ] `public:include/cli.h:Command`
- [ ] `public:include/cli.h:EvaluateOptions`
- [ ] `public:include/cli.h:InferOptions`
- [ ] `public:include/cli.h:ParseCli`
- [ ] `public:include/cli.h:TrainOptions`
- [ ] `public:include/cli.h:Usage`
- [ ] `public:include/dataset.h:LoadMnistDataset`
- [ ] `public:include/dataset.h:MnistDataset`
- [ ] `public:include/dataset.h:MnistDataset::Image`
- [ ] `public:include/dataset.h:RequireDatasetCount`
- [ ] `public:include/layers.h:LaunchAdamW`
- [ ] `public:include/layers.h:LaunchArgmaxAndCountCorrect`
- [ ] `public:include/layers.h:LaunchConvolutionBackward`
- [ ] `public:include/layers.h:LaunchConvolutionForward`
- [ ] `public:include/layers.h:LaunchFindFirstNonFinite`
- [ ] `public:include/layers.h:LaunchLinearBackward`
- [ ] `public:include/layers.h:LaunchLinearForward`
- [ ] `public:include/layers.h:LaunchMaxPoolBackward`
- [ ] `public:include/layers.h:LaunchMaxPoolForward`
- [ ] `public:include/layers.h:LaunchNormalizeTranslate`
- [ ] `public:include/layers.h:LaunchReluBackward`
- [ ] `public:include/layers.h:LaunchReluForward`
- [ ] `public:include/layers.h:LaunchSoftmax`
- [ ] `public:include/layers.h:LaunchSoftmaxCrossEntropy`
- [ ] `public:include/layers.h:LaunchZero`
- [ ] `public:include/lenet.h:AdamWConfig`
- [ ] `public:include/lenet.h:LeNet`
- [ ] `public:include/lenet.h:LeNet::AdamWStep`
- [ ] `public:include/lenet.h:LeNet::Backward`
- [ ] `public:include/lenet.h:LeNet::ExportParameters`
- [ ] `public:include/lenet.h:LeNet::Forward`
- [ ] `public:include/lenet.h:LeNet::ImportParameters`
- [ ] `public:include/lenet.h:LeNet::LeNet`
- [ ] `public:include/lenet.h:LeNet::RequireFinite`
- [ ] `public:include/lenet.h:LeNet::RequiredDeviceBytes`
- [ ] `public:include/lenet.h:LeNet::operator=`
- [ ] `public:include/lenet.h:LeNet::~LeNet`
- [ ] `public:include/lenet.h:LeNetTestAccess`
- [ ] `public:include/lenet.h:LeNetTestAccess::Fc1InputAliasesPool2`
- [ ] `public:include/lenet.h:LeNetTestAccess::FiniteDiagnosticNamesPrepared`
- [ ] `public:include/parameters.h:CreateLenetParameters`
- [ ] `public:include/parameters.h:InitializeLenetParameters`
- [ ] `public:include/parameters.h:LenetParameterSpecs`
- [ ] `public:include/parameters.h:ParameterSet`
- [ ] `public:include/parameters.h:ParameterSpec`
- [ ] `public:include/parameters.h:ParameterTensor`
- [ ] `public:include/parameters.h:ValidateLenetParameters`
- [ ] `public:include/random.h:DatasetSplit`
- [ ] `public:include/random.h:MakeTrainValidationSplit`
- [ ] `public:include/random.h:ShuffledTrainingIndices`
- [ ] `public:include/random.h:SplitMix64`
- [ ] `public:include/random.h:SplitMix64::Next`
- [ ] `public:include/random.h:SplitMix64::SplitMix64`
- [ ] `public:include/random.h:SplitMix64::UniformBounded`
- [ ] `public:include/random.h:TranslationOffset`
- [ ] `public:include/random.h:TranslationOffsetForSample`
- [ ] `public:include/reporting.h:PrintEpochSummary`
- [ ] `public:include/reporting.h:PrintEvaluationSummary`
- [ ] `public:include/tensor.h:CudaMemoryInfoQueryCountForTests`
- [ ] `public:include/tensor.h:CudaMemoryInfoQueryInvocationSequenceForTests`
- [ ] `public:include/tensor.h:DeviceBuffer`
- [ ] `public:include/tensor.h:DeviceBuffer::DeviceBuffer`
- [ ] `public:include/tensor.h:DeviceBuffer::get`
- [ ] `public:include/tensor.h:DeviceBuffer::operator=`
- [ ] `public:include/tensor.h:DeviceBuffer::size`
- [ ] `public:include/tensor.h:DeviceBuffer::~DeviceBuffer`
- [ ] `public:include/tensor.h:DeviceBufferAllocationAttemptSequenceForTests`
- [ ] `public:include/tensor.h:DeviceBufferAllocationCountForTests`
- [ ] `public:include/tensor.h:DeviceBufferSuccessfulAllocationEventCountForTests`
- [ ] `public:include/train.h:ExitCode`
- [ ] `public:include/train.h:LearningRateForEpoch`
- [ ] `public:include/train.h:RunEvaluate`
- [ ] `public:include/train.h:RunInfer`
- [ ] `public:include/train.h:RunTrain`
- [ ] `public:include/training_data.h:HostBatch`
- [ ] `public:include/training_data.h:MakeWorkflowSplit`
- [ ] `public:include/training_data.h:PackBatch`

## CUDA Kernels And Reductions

- [ ] `kernel:src/kernels/activation.cu:ReluBackwardKernel`
- [ ] `kernel:src/kernels/activation.cu:ReluForwardKernel`
- [ ] `kernel:src/kernels/activation.cu:ZeroKernel`
- [ ] `kernel:src/kernels/adam.cu:AdamWKernel`
- [ ] `kernel:src/kernels/adam.cu:FindFirstNonFiniteKernel`
- [ ] `kernel:src/kernels/adam.cu:InitializeFirstBadIndexKernel`
- [ ] `kernel:src/kernels/convolution.cu:ConvolutionBiasGradientKernel`
- [ ] `kernel:src/kernels/convolution.cu:ConvolutionForwardKernel`
- [ ] `kernel:src/kernels/convolution.cu:ConvolutionInputGradientKernel`
- [ ] `kernel:src/kernels/convolution.cu:ConvolutionWeightGradientKernel`
- [ ] `kernel:src/kernels/input.cu:NormalizeTranslateKernel`
- [ ] `kernel:src/kernels/linear.cu:LinearBiasGradientKernel`
- [ ] `kernel:src/kernels/linear.cu:LinearForwardKernel`
- [ ] `kernel:src/kernels/linear.cu:LinearInputGradientKernel`
- [ ] `kernel:src/kernels/linear.cu:LinearWeightGradientKernel`
- [ ] `kernel:src/kernels/loss.cu:MeanLossKernel`
- [ ] `kernel:src/kernels/loss.cu:SoftmaxCrossEntropyKernel`
- [ ] `kernel:src/kernels/loss.cu:SoftmaxKernel`
- [ ] `kernel:src/kernels/metrics.cu:ArgmaxFlagsKernel`
- [ ] `kernel:src/kernels/metrics.cu:CorrectCountKernel`
- [ ] `kernel:src/kernels/pooling.cu:MaxPoolBackwardKernel`
- [ ] `kernel:src/kernels/pooling.cu:MaxPoolForwardKernel`

## Binary Serializers And Parsers

- [ ] `manual:include/cuda_check.h:CheckCuda`
- [ ] `manual:scripts/prepare_mnist.py:convert_parquet`
- [ ] `manual:src/binary_io.h:ReadExact`
- [ ] `manual:src/binary_io.h:ReadF32LE`
- [ ] `manual:src/binary_io.h:ReadU32LE`
- [ ] `manual:src/binary_io.h:ReadU64LE`
- [ ] `manual:src/binary_io.h:WriteExact`
- [ ] `manual:src/binary_io.h:WriteF32LE`
- [ ] `manual:src/binary_io.h:WriteU32LE`
- [ ] `manual:src/binary_io.h:WriteU64LE`
- [ ] `manual:src/checkpoint.cpp:LoadCheckpoint`
- [ ] `manual:src/checkpoint.cpp:ReadAndValidateName`
- [ ] `manual:src/checkpoint.cpp:SaveCheckpoint`
- [ ] `manual:src/checkpoint.cpp:WriteCheckpointContents`
- [ ] `manual:src/dataset.cpp:LoadMnistDataset`
