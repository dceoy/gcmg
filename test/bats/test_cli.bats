#!/usr/bin/env bats

setup_file() {
  set -euo pipefail
  echo "BATS test file: ${BATS_TEST_FILENAME}" >&3
}

teardown_file() {
  :
}

@test "pass with \"gcmg --version\"" {
  run poetry run gcmg --version
  [[ "${status}" -eq 0 ]]
}

@test "pass with \"gcmg --help\"" {
  run poetry run gcmg --help
  [[ "${status}" -eq 0 ]]
}
