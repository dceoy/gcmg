#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export GOOGLE_MODEL="${GOOGLE_MODEL:-gemini-1.5-flash}"
}

teardown_file() {
  :
}

@test "pass with \"gcmg --google-model\"" {
  run poetry run gcmg --google-model="${GOOGLE_MODEL}"
  [[ "${status}" -eq 0 ]]
}
