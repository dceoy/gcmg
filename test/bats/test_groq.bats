#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  export GROQ_MODEL="${GROQ_MODEL:-llama-3.1-70b-versatile}"
}

teardown_file() {
  :
}

@test "pass with \"gcmg --groq-model\"" {
  run poetry run gcmg --groq-model="${GROQ_MODEL}"
  [[ "${status}" -eq 0 ]]
}
