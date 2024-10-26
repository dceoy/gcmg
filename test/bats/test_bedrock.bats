#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
  aws sts get-caller-identity
  export BEDROCK_MODEL="${BEDROCK_MODEL:-anthropic.claude-3-5-sonnet-20240620-v1:0}"
}

teardown_file() {
  :
}

@test "pass with \"gcmg --bedrock-model\"" {
  run poetry run gcmg --bedrock-model="${BEDROCK_MODEL}"
  [[ "${status}" -eq 0 ]]
}
