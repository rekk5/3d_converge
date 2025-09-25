#!/bin/bash

# 3D Reconstruction Pipeline Docker Runner Script
# This script provides convenient commands for running the pipeline in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="mast3r-3d-pipeline"
INPUT_DIR="./input"
OUTPUT_DIR="./output"
TEST_DIR="./test_output"
CONTAINER_NAME="mast3r-container"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build                    Build the Docker image"
    echo "  run                      Run the pipeline"
    echo "  dev                      Start development container with shell"
    echo "  viz                      Run visualization"
    echo "  test                     Run tests"
    echo "  clean                    Clean up containers and images"
    echo "  help                     Show this help message"
    echo ""
    echo "Options:"
    echo "  -i, --input DIR          Input directory (default: ./input)"
    echo "  -o, --output DIR         Output directory (default: ./output)"
    echo "  -t, --test DIR           Test directory (default: ./test_output)"
    echo "  -n, --name NAME          Container name (default: mast3r-container)"
    echo "  -g, --gpu                Enable GPU support"
    echo "  -x, --x11                Enable X11 forwarding for visualization"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run -i ./my_images -o ./my_results"
    echo "  $0 viz -x"
    echo "  $0 dev"
}

# Function to build Docker image
build_image() {
    print_info "Building Docker image..."
    docker build -t $IMAGE_NAME .
    print_success "Docker image built successfully!"
}

# Function to run pipeline
run_pipeline() {
    print_info "Running 3D reconstruction pipeline..."
    
    # Create directories if they don't exist
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$TEST_DIR"
    
    # Build base command
    CMD="docker run --rm"
    
    # Add container name
    CMD="$CMD --name $CONTAINER_NAME"
    
    # Add volume mounts
    CMD="$CMD -v $(realpath $INPUT_DIR):/app/input:ro"
    CMD="$CMD -v $(realpath $OUTPUT_DIR):/app/output"
    CMD="$CMD -v $(realpath $TEST_DIR):/app/test_output"
    
    # Add GPU support if requested
    if [[ $GPU == "true" ]]; then
        CMD="$CMD --gpus all"
        CMD="$CMD -e NVIDIA_VISIBLE_DEVICES=all"
        print_info "GPU support enabled"
    fi
    
    # Add X11 forwarding if requested
    if [[ $X11 == "true" ]]; then
        CMD="$CMD -e DISPLAY=\$DISPLAY"
        CMD="$CMD -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
        print_info "X11 forwarding enabled"
    fi
    
    # Add environment variables
    CMD="$CMD -e PYTHONUNBUFFERED=1"
    
    # Add image and command
    CMD="$CMD $IMAGE_NAME"
    CMD="$CMD conda run -n mast3r python main.py"
    
    # Add remaining arguments
    if [[ $# -gt 0 ]]; then
        CMD="$CMD $@"
    else
        CMD="$CMD --help"
    fi
    
    print_info "Executing: $CMD"
    eval $CMD
}

# Function to start development container
start_dev() {
    print_info "Starting development container..."
    
    # Create directories if they don't exist
    mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$TEST_DIR"
    
    CMD="docker run -it --rm"
    CMD="$CMD --name $CONTAINER_NAME"
    CMD="$CMD -v $(realpath $INPUT_DIR):/app/input:ro"
    CMD="$CMD -v $(realpath $OUTPUT_DIR):/app/output"
    CMD="$CMD -v $(realpath $TEST_DIR):/app/test_output"
    CMD="$CMD -v $(pwd):/app"
    CMD="$CMD -e PYTHONUNBUFFERED=1"
    CMD="$CMD $IMAGE_NAME"
    CMD="$CMD conda run -n mast3r /bin/bash"
    
    print_info "Starting interactive shell..."
    eval $CMD
}

# Function to run visualization
run_visualization() {
    print_info "Running visualization..."
    
    # Create directories if they don't exist
    mkdir -p "$OUTPUT_DIR" "$TEST_DIR"
    
    CMD="docker run -it --rm"
    CMD="$CMD --name $CONTAINER_NAME"
    CMD="$CMD -v $(realpath $OUTPUT_DIR):/app/output:ro"
    CMD="$CMD -v $(realpath $TEST_DIR):/app/test_output:ro"
    CMD="$CMD -e PYTHONUNBUFFERED=1"
    
    # Add X11 forwarding if requested
    if [[ $X11 == "true" ]]; then
        CMD="$CMD -e DISPLAY=\$DISPLAY"
        CMD="$CMD -v /tmp/.X11-unix:/tmp/.X11-unix:rw"
        print_info "X11 forwarding enabled"
    fi
    
    CMD="$CMD $IMAGE_NAME"
    CMD="$CMD conda run -n mast3r python visualization.py"
    
    # Add remaining arguments
    if [[ $# -gt 0 ]]; then
        CMD="$CMD $@"
    else
        CMD="$CMD --help"
    fi
    
    print_info "Executing: $CMD"
    eval $CMD
}

# Function to run tests
run_tests() {
    print_info "Running tests..."
    
    # Create test directory if it doesn't exist
    mkdir -p "$TEST_DIR"
    
    CMD="docker run --rm"
    CMD="$CMD --name $CONTAINER_NAME"
    CMD="$CMD -v $(realpath $TEST_DIR):/app/test_output"
    CMD="$CMD -e PYTHONUNBUFFERED=1"
    CMD="$CMD $IMAGE_NAME"
    CMD="$CMD conda run -n mast3r python test_visualization.py"
    
    print_info "Executing: $CMD"
    eval $CMD
}

# Function to clean up
cleanup() {
    print_info "Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker ps -a --filter "name=$CONTAINER_NAME" --format "{{.ID}}" | xargs -r docker rm -f
    
    # Remove images
    docker images --filter "reference=$IMAGE_NAME" --format "{{.ID}}" | xargs -r docker rmi -f
    
    print_success "Cleanup completed!"
}

# Parse command line arguments
COMMAND=""
GPU="false"
X11="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        build|run|dev|viz|test|clean|help)
            COMMAND="$1"
            shift
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--test)
            TEST_DIR="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="true"
            shift
            ;;
        -x|--x11)
            X11="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            # Pass remaining arguments to the command
            break
            ;;
    esac
done

# Check if command is provided
if [[ -z "$COMMAND" ]]; then
    print_error "No command specified"
    show_usage
    exit 1
fi

# Execute command
case $COMMAND in
    build)
        build_image
        ;;
    run)
        run_pipeline "$@"
        ;;
    dev)
        start_dev
        ;;
    viz)
        run_visualization "$@"
        ;;
    test)
        run_tests
        ;;
    clean)
        cleanup
        ;;
    help)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac 