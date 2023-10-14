#!/bin/bash

input_dir="input"
output_dir="output"
my_face_image=""

# Check for the correct number of arguments
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: $0 <input_face_image.jpg> [input_directory]"
  exit 1
fi

# Check if the my_face_image argument is provided
if [ $# -ge 1 ]; then
  my_face_image="$1"
fi

# Check if the input directory is provided
if [ $# -eq 2 ]; then
  input_dir="$2"
fi

# Check if the input directory exists
if [ ! -d "$input_dir" ]; then
  echo "Input directory '$input_dir' not found."
  exit 1
fi

# Create the output directory if it doesn't exist
if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

# Loop through all .mp4 files in the input directory
for input_file in "$input_dir"/*.mp4; do
  if [ -f "$input_file" ]; then
    # Extract the filename without the directory and extension
    filename=$(basename -- "$input_file")

    # Run the swap.py command for each file
    python3 swap.py "$input_file" "$my_face_image" "$output_dir/$filename"
    echo "Processed: $input_file"
  fi
done

echo "All .mp4 files in '$input_dir' have been processed."