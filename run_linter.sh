#!/usr/bin/env bash
set -euo pipefail

docker run -e RUN_LOCAL=true -v "$PWD":/tmp/lint github/super-linter
