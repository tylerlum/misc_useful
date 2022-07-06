#!/usr/bin/env bash

# PARAMETERS START: PLEASE MODIFY
START_TIME="00:00:00"
END_TIME="00:00:12"
# PARAMETERS END: PLEASE MODIFY

# Get path to video file
input_path_to_video_file=$1
echo "Received input_path_to_video_file = $input_path_to_video_file"

# Validate video file exists
if [ ! -f "$input_path_to_video_file" ]; then
  echo "$input_path_to_video_file does not exist."
  echo "Usage: `basename $0` <input_path_to_video_file>"
  exit 1
fi

# Append time to filename
original_filename=${input_path_to_video_file%.*}
extension=${input_path_to_video_file##*.}
output_path_to_short_video_file="${original_filename}_Start-${START_TIME}_End-${END_TIME}.${extension}"
echo "Creating shorter video at output_path_to_short_video_file = $output_path_to_short_video_file"

# Print out command as it is run
set -x
ffmpeg \
  -i "$input_path_to_video_file" \
  -ss "$START_TIME" -to "$END_TIME" \
  -c:v copy \
  -c:a copy \
  "$output_path_to_short_video_file"
set +x
