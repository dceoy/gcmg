#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
}

teardown_file() {
  :
}

@test "pass with \"gcmg --openai-model\"" {
  run poetry run gcmg --openai-model="${OPENAI_MODEL}"
  [[ "${status}" -eq 0 ]]
}
