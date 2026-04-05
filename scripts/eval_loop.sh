#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: ./scripts/eval_loop.sh <language1> [language2 ...]"
    echo "Example: ./scripts/eval_loop.sh indo sun min"
    exit 1
fi

for language in "$@"; do
    echo "Running evaluation for language: $language"
    ./scripts/eval.sh "$language"
    echo "Finished language: $language"
    echo "----------------------------------------"
done

echo "All language evaluations completed."
