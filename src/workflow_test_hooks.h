#ifndef SRC_WORKFLOW_TEST_HOOKS_H_
#define SRC_WORKFLOW_TEST_HOOKS_H_

class LeNet;

namespace workflow_test_hooks {

// Names the test-only callback invoked immediately before RunTrain reloads the
// persisted best checkpoint. The callback borrows the live model only for the
// call and must not retain it. Normal production execution has a null callback.
using BeforeFinalReloadHook = void (*)(LeNet* model);

// Replaces the process-wide test callback, or disables it when hook is null.
// This narrow internal seam performs no CUDA work itself and is not declared by
// any public header. Tests must restore null before returning. Concurrent
// RunTrain calls while changing the hook are unsupported.
void SetBeforeFinalReloadHookForTests(BeforeFinalReloadHook hook) noexcept;

}  // namespace workflow_test_hooks

#endif  // SRC_WORKFLOW_TEST_HOOKS_H_
