#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: ./scripts/inference_loop.sh <language1> [language2 ...]"
    echo "Example: ./scripts/inference_loop.sh indo sun min"
    exit 1
fi

for language in "$@"; do
    echo "Running inference for language: $language"
    ./scripts/inference.sh "$language"
    echo "Finished language: $language"
    echo "----------------------------------------"
done

echo "All language inferences completed."
