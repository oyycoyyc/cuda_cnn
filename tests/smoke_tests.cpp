// Minimal registration/link sanity check for the shared test harness.
#include "test_harness.h"

// Confirms a case registers, runs, and can assert without extra dependencies.
TEST_CASE(smoke) {
  EXPECT_EQ(4, 2 + 2);
}
