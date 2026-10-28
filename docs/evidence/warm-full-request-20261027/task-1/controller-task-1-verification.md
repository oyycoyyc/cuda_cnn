# Warm Task1 main H20 verification

- Tests-first: NVCC/C++14/O2/arch90 tests-only compile/link0warnings;
 7casesfailonlynamedmissingtoolbeforefixtures/hardware. task-1-red*.txt.
- Productionhandoffchangesonlynewtool,newtest,optionalCMakeinsideCOURSE_BUILD_TESTS.
  Fullfreshconfigure/build0warnings; all5CTestPASS48host/8GPU/9training/15runtime/
  7warm. Warmtest has300sCTesttimeout and240schilddeadline. task-1-build.txt /
  task-1-ctest.txt, sevennewcasesactuallyexecuted/noskips.
- Literalnormalizedzero/onefixture[1,0]atB1/B2W2R3; officialinfercomparisons
  N17/B4and1025/B1024; finitepoisonfailurezerooutput; usage/mode/input/model
  branchesactuallyruninchildGPUprocesses. ParentmakesnoCUDAcallbeforefork.
- Defaultfreshminimal sourcecopy hasnooptionaltests/tools files butbuilds
  train/inferwithnewCMake,0warnings. task-1-default-build.txt.
- Actualwarmtoolandfreshinferlink/nmcontainonlyhost/forwarddependencies;
  noLeNet/trainworkflow/Adam/loss/metrics. AllcommandsC++14/O2/arch90,nofastmath.
  Oldofficialruntime/math untouched, no model/data/rootartifactwrites.
- MainrealOSsinkprobes: W2R3/N2normalexact2stdoutclasses+2warm/3measurerecords;
  stdout=/dev/fullreturns1/no passingcompletion; stderr=/dev/fullreturns1/empty
  stdout. task-1-audit.txt / io-normal-stderr.txt. No boundedtinyfixture timing
  isclaimedastheactualfull10000warmmeasurement.
- Sourceinspection: startbeforeflagreset, eachroundallH2D/prepare/forward/
  argmax/globaloutput offsets,allN+flagD2H/streamsync/checks,endafterchecks.
  Model/stream/storage/allhostvectorsallocatedbeforeloop,setupdrained; no
  devicealloc/events/disk/formattingintheclockwindow. Everyactualround's
  readbackismatchedtothefirstrealwarmupvector, notcopiedfromcachedpredictions.

Remaining: independentTask1review/currenthashproof/checkpoint, thenTask2
collector/rawschemaandrealW5R101. No actualnewwarmmean/medianexistsyet.
