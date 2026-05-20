#!/usr/bin/env bash
# Wire git to use the repo-tracked hooks in .githooks/.
# Run once after cloning.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
git config --local core.hooksPath .githooks

echo "hooks installed:"
ls -1 .githooks/

echo
echo "core.hooksPath = $(git config --get core.hooksPath)"
echo "git will now run .githooks/pre-commit and .githooks/pre-push on every commit/push."
echo "bypass with --no-verify if you've already gated CI yourself."
