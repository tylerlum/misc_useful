#!/usr/bin/env bash

# PARAMETERS START: PLEASE MODIFY
START_TIME="00:00:00"
END_TIME="00:00:12"
WIDTH=512  # pixels
HEIGHT=-1  # pixels or -1 to match width aspect ratio
FPS=30
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

# Remove extension to create output gif name
original_filename=${input_path_to_video_file%.*}
output_path_to_gif_file="${original_filename}.gif"
echo "Creating gif at output_path_to_gif_file = $output_path_to_gif_file"

# Print out command as it is run
set -x
ffmpeg \
  -i "$input_path_to_video_file" \
  -r $FPS \
  -vf scale=$WIDTH:$HEIGHT \
  -ss "$START_TIME" -to "$END_TIME" \
  "$output_path_to_gif_file"
set +x
