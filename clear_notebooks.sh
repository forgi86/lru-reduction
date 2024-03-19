#!/bin/bash

# Action to be executed
clean_notebook() {
  jupyter nbconvert \
    --clear-output \
    --inplace \
    "$1"
}

# Recursive function to process directories
process_directory() {
    local directory="$1"

    for file in "$directory"/*; do
        if [ -d "$file" ]; then
            process_directory "$file"   # Recursively process subdirectories
        elif [ "${file##*.}" = "ipynb" ]; then
            clean_notebook "$file"       # Execute bash script
        fi
    done
}

# Start processing from the current directory
process_directory "."