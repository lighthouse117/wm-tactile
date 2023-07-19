#!/bin/bash

IMAGE_NAME="iiyama/tactile"
CONTAINER_NAME="iiyama-tactile"
GPU_NUMBER=""

CMD="/bin/bash"

# Parse the arguments
while getopts ":g:" opt; do
  case ${opt} in
    g )
      GPU_NUMBER="${OPTARG}"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if the GPU number is provided
if [ -z "$GPU_NUMBER" ]; then
  echo "Please provide the GPU number using the -g option."
  exit 1
fi

# Set the container name
CONTAINER_NAME="$CONTAINER_NAME$GPU_NUMBER"


docker run -it --rm --runtime=nvidia --gpus device=$GPU_NUMBER \
    -v ~/wm-tactile:/workspace \
    --ipc=host \
    --name $CONTAINER_NAME $IMAGE_NAME \
    $CMD