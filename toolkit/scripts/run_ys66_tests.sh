#!/usr/bin/env bash
# Copyright 2026 Martin Fabbri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run YS-66 replay tests via xcodebuild and report any run with a skipped test as "NOT RUN" (exit 2),
# never "passed".

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_PATH="$REPO_ROOT/ios/Yardstick/Yardstick.xcodeproj"
SCHEME="${YS66_SCHEME:-YardstickTests}"
DESTINATION="${YS66_DESTINATION:-platform=macOS,arch=arm64}"

LOG_FILE="$(mktemp -t ys66_xcodebuild.XXXXXX.log)"
trap 'rm -f "$LOG_FILE"' EXIT

export TEST_RUNNER_YARDSTICK_REPLAY_ROOT="${YARDSTICK_REPLAY_ROOT:-${TEST_RUNNER_YARDSTICK_REPLAY_ROOT:-}}"

set +e
xcodebuild test \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME" \
  -destination "$DESTINATION" \
  2>&1 | tee "$LOG_FILE"
XC_STATUS=${PIPESTATUS[0]}
set -e

if [[ $XC_STATUS -ne 0 ]]; then
  echo "YS-66 RESULT: FAILED (xcodebuild exited with $XC_STATUS)"
  exit "$XC_STATUS"
fi

if grep -Eq "YS-66 VERDICT: NOT RUN|Test Case '.*YS66.*' skipped|with [1-9][0-9]* test[s]? skipped" "$LOG_FILE"; then
  echo "YS-66 RESULT: NOT RUN (one or more YS-66 tests were skipped; set YARDSTICK_REPLAY_ROOT for full walks)"
  exit 2
fi

echo "YS-66 RESULT: PASSED (0 skipped, 0 failed)"
exit 0
