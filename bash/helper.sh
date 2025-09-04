#!/bin/bash

extract_last_best_step() {
    # Function to extract the best checkpoint training step from a given log file
    # Example usage:
    # step=$(extract_last_best_step "/path/to/results.txt")

    local file_path="$1"

    # Check if the file exists
    if [[ ! -f "$file_path" ]]; then
        echo "File '$file_path' not found!"
        return 1
    fi

    # Use awk to find the last "Best results:" section and extract the Step value
    local best_step=$(awk '
        /Best results/ {
            in_best = 1
            next
        }
        in_best == 1 && $1 ~ /^[0-9]+$/ {
            last_step = $1
            in_best = 0
        }
      END {
          if (last_step) print last_step
      }
    ' "$file_path")

    # Output the result
    if [[ -n "$best_step" ]]; then
        echo "$best_step"
    else
        echo "No valid 'Best results' section found in '$file_path'."
        return 1
    fi
}

