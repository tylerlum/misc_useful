#!/usr/bin/env bash

# PARAMETERS START: PLEASE MODIFY
REGION_WIDTH=200  # pixels
REGION_HEIGHT=400  # pixels
REGION_UPPER_LEFT_X=100  # pixels
REGION_UPPER_LEFT_Y=300  # pixels
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

# Add parameters to filename
original_filename=${input_path_to_video_file%.*}
extension=${input_path_to_video_file##*.}
output_path_to_short_video_file="_Start-${START_TIME}_End-${END_TIME}.${extension}"
output_path_to_cropped_region_video_file="${original_filename}_WIDTH-${REGION_WIDTH}_HEIGHT-${REGION_HEIGHT}_UPPER-LEFT-X-${REGION_UPPER_LEFT_X}_UPPER-LEFT-Y-${REGION_UPPER_LEFT_Y}.${extension}"
echo "Creating gif at output_path_to_cropped_region_video_file = $output_path_to_cropped_region_video_file"

# Print out command as it is run
set -x
ffmpeg \
  -i "$input_path_to_video_file" \
  -filter:v "crop=${REGION_WIDTH}:${REGION_HEIGHT}:${REGION_UPPER_LEFT_X}:${REGION_UPPER_LEFT_Y}" \
  "$output_path_to_cropped_region_video_file"
set +x
