# 3D Reconstruction Pipeline with Diffraction Edge Detection

A comprehensive 3D reconstruction pipeline that combines multiple state-of-the-art reconstruction methods (COLMAP, TSDF Fusion, MAST3R, VGGT) with advanced point cloud segmentation and diffraction edge detection for electromagnetic simulation applications.

## 🚀 Features

- **Multiple Reconstruction Methods**: COLMAP, TSDF Fusion, MAST3R, and VGGT
- **VGGT Integration**: Latest neural 3D reconstruction with video support
- **Advanced Segmentation**: Enhanced RANSAC + DBSCAN plane segmentation
- **Diffraction Edge Detection**: Automatic detection of electromagnetic diffraction edges
- **Interactive Visualization**: Multi-camera rendering and customizable views
- **Point Cloud Comparison**: Enhanced differential analysis with confidence scoring
- **Docker Support**: Complete containerization for easy deployment
- **CI/CD Pipeline**: Automated testing and deployment with GitLab CI

## 📋 Prerequisites

### For Docker Installation (Recommended)
- Docker and Docker Compose
- NVIDIA GPU (recommended, for CUDA acceleration)
- NVIDIA Container Toolkit (for GPU support in Docker)

### For Local Installation
- Anaconda or Miniconda
- NVIDIA GPU with CUDA 11.8+ or CUDA 12.1+
- Python 3.10 (automatically installed via conda)
- Git (for cloning repositories)

### Optional
- GitLab account (for CI/CD)
- 16GB+ RAM (for large point clouds)
- 8GB+ VRAM (for optimal VGGT performance)

## 🚀 Installation Methods

### Method 1: Docker Installation (Recommended)

Docker provides the easiest and most reliable way to run the pipeline with all dependencies pre-configured.

#### 1. Clone the Repository
```bash
git clone <your-gitlab-repo-url>
cd 3d_converge
```

#### 2. Install Docker & NVIDIA Container Toolkit
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 3. Build and Run with Docker Compose
```bash
# Build the Docker image
docker-compose build

# Run VGGT reconstruction pipeline
docker-compose up vggt-pipeline

# Run MAST3R reconstruction pipeline
docker-compose up mast3r-pipeline

# For development with interactive shell
docker-compose run --rm mast3r-dev

# For visualization with GPU support
docker-compose run --rm --gpus all mast3r-viz
```

#### 4. Using Docker Directly
```bash
# Build the image with VGGT support
docker build -t vggt-3d-pipeline .

# Run VGGT reconstruction with GPU support
docker run -it --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/test_output:/app/test_output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/images \
    --output_dir /app/output \
    --recon_method vggt \
    --conf_threshold 3.0

# Check available options
docker run -it --rm vggt-3d-pipeline \
  conda run -n mast3r python main.py --help
```

---

### Method 2: Local Conda Installation

For users who prefer local installation or need to modify the code extensively.

#### 1. Clone the Repository
```bash
git clone <your-gitlab-repo-url>
cd 3d_converge

# Initialize VGGT submodule (if needed)
git submodule update --init --recursive
```

#### 2. Install Anaconda/Miniconda
```bash
# Download and install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 3. Create Environment from YAML
```bash
# Create the conda environment with all VGGT dependencies
conda env create -f environment.yml

# Activate the environment
conda activate mast3r

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

#### 4. Install VGGT (if using VGGT submodule)
```bash
# If you have the VGGT submodule
cd vggt
pip install -e .
cd ..

# Or install VGGT separately
pip install vggt
```

#### 5. Test the Installation
```bash
# Test basic functionality
python main.py --help

# Test with sample data (if available)
python main.py \
  --image_path input/sample_images \
  --output_dir output/test \
  --recon_method vggt \
  --conf_threshold 3.0
```

#### 6. Optional: Install Additional Dependencies
```bash
# For development and testing
pip install pytest black flake8 mypy

# For Jupyter notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name=mast3r
```

## 📁 Project Structure

```
3d_converge/
├── main.py                 # Main pipeline entry point
├── comparison.py           # Point cloud comparison utilities
├── segmentation.py         # Point cloud segmentation
├── diffraction.py          # Diffraction edge detection
├── vggt_recon.py          # VGGT reconstruction module
├── mast3r_recon.py        # MAST3R reconstruction module
├── colmap_recon.py        # COLMAP reconstruction module
├── tsdf_fusion.py         # TSDF fusion module
├── visualization.py        # Interactive visualization tool
├── test_visualization.py   # Visualization testing
├── example_visualization.py # Example camera configurations
├── converge-3ds/          # FastAPI services
│   ├── 3D_Pipeline/       # 3D reconstruction API
│   └── Visualization/     # Visualization API
├── vggt/                  # VGGT submodule
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── environment.yml        # Conda environment with VGGT deps
├── .gitlab-ci.yml         # GitLab CI/CD pipeline
├── input/                 # Input images/data (mounted volume)
├── output/                # Pipeline outputs (mounted volume)
└── test_output/           # Test data (mounted volume)
```

## 🔧 Usage

The pipeline can be used with either Docker or local conda installation. Examples below show both methods.

### VGGT Reconstruction (Recommended)

#### Docker Usage:
```bash
# Run VGGT reconstruction from images
docker run -it --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/images \
    --output_dir /app/output \
    --recon_method vggt \
    --conf_threshold 3.0 \
    --show_cameras

# Run VGGT reconstruction from video
docker run -it --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/video.mp4 \
    --output_dir /app/output \
    --recon_method vggt \
    --fps 2.0 \
    --conf_threshold 2.5
```

#### Local Conda Usage:
```bash
# Activate environment
conda activate mast3r

# Run VGGT reconstruction from images
python main.py \
  --image_path input/images \
  --output_dir output \
  --recon_method vggt \
  --conf_threshold 3.0 \
  --show_cameras

# Run VGGT reconstruction from video
python main.py \
  --image_path input/video.mp4 \
  --output_dir output \
  --recon_method vggt \
  --fps 2.0 \
  --conf_threshold 2.5
```

### MAST3R Reconstruction

#### Docker Usage:
```bash
# Run complete pipeline with MAST3R reconstruction
docker run -it --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/images \
    --output_dir /app/output \
    --recon_method mast3r \
    --image_size 512 \
    --resize_images
```

#### Local Conda Usage:
```bash
conda activate mast3r
python main.py \
  --image_path input/images \
  --output_dir output \
  --recon_method mast3r \
  --image_size 512 \
  --resize_images
```

### COLMAP Reconstruction

#### Docker Usage:
```bash
# Traditional photogrammetry with COLMAP
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/images \
    --output_dir /app/output \
    --recon_method colmap
```

#### Local Conda Usage:
```bash
conda activate mast3r
python main.py \
  --image_path input/images \
  --output_dir output \
  --recon_method colmap
```

### Point Cloud Comparison & Analysis
```bash
# Enhanced comparison with confidence scoring
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/input/new_scan.ply \
    --base_cloud_path /app/input/reference_scan.ply \
    --comparison_threshold 0.15 \
    --enhanced_comparison \
    --use_multi_scale \
    --use_confidence \
    --use_color_coding

# Basic differential analysis
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/input/current.ply \
    --base_cloud_path /app/input/baseline.ply \
    --comparison_threshold 0.1 \
    --comparison_voxel_size 0.02
```

### Advanced Segmentation & Diffraction Detection
```bash
# Enhanced segmentation with diffraction edge detection
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/input/scan.ply \
    --output_dir /app/output \
    --enhanced_segmentation \
    --max_planes 60 \
    --distance_threshold 0.04 \
    --create_mesh \
    --diffraction_method segmentation

# Normal-based edge detection (no segmentation required)
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/input/scan.ply \
    --output_dir /app/output \
    --skip_segmentation \
    --diffraction_method normals \
    --angle_threshold 45.0 \
    --normal_radius 0.05
```

### Multi-Camera Visualization & Rendering
```bash
# Interactive visualization with corner view
docker run -it --rm \
  -v $(pwd)/output:/app/output:ro \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  vggt-3d-pipeline \
  conda run -n mast3r python visualization.py \
    --point_cloud /app/output/point_cloud.ply \
    --mesh_with_edges /app/output/saalas_mesh.ply \
    --camera_position "10,10,10" \
    --camera_fov 90 \
    --auto_advance

# Render 8-camera views using FastAPI service
curl -X POST "http://localhost:8001/render-cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "point_cloud_path": "/data/output/point_cloud.ply",
    "width": 1280,
    "height": 720
  }'

# Custom camera positions for rendering
curl -X POST "http://localhost:8001/render-cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "point_cloud_path": "/data/output/scan.ply",
    "cameras": [
      {"position": [10, 10, 10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [-10, 10, 10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [10, -10, 10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [-10, -10, 10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [10, 10, -10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [-10, 10, -10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [10, -10, -10], "look_at": [0, 0, 0], "fov": 60},
      {"position": [-10, -10, -10], "look_at": [0, 0, 0], "fov": 60}
    ],
    "width": 1920,
    "height": 1080
  }'
```

## 🧪 Testing & Examples

### Run Complete Tests
```bash
# Test all functionality including VGGT
docker run -it --rm --gpus all \
  -v $(pwd)/test_output:/app/test_output \
  vggt-3d-pipeline \
  conda run -n mast3r python test_visualization.py

# Test FastAPI services
docker-compose up -d
curl http://localhost:8000/health  # 3D Pipeline API
curl http://localhost:8001/health  # Visualization API
```

### Test Individual Components
```bash
# Test VGGT reconstruction
docker run -it --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python vggt_recon.py \
    /app/input/images \
    /app/output \
    --conf_threshold 3.0

# Test enhanced segmentation
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python segmentation.py \
    /app/input/point_cloud.ply \
    --enhanced_segmentation \
    --max_planes 60

# Test diffraction edge detection with mesh creation
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python diffraction.py \
    /app/input/segmented.ply \
    /app/output/saalas_mesh.ply \
    --create_mesh \
    --visualize

# Test point cloud comparison
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python comparison.py \
    /app/input/base.ply \
    /app/input/target.ply \
    --enhanced_comparison \
    --threshold 0.15
```

## 🔄 GitLab CI/CD

The project includes a complete GitLab CI/CD pipeline with:

- **Testing**: Automated testing of all components
- **Building**: Docker image building and registry pushing
- **Deployment**: Staging and production deployment
- **Security**: Security scanning and vulnerability assessment

### Pipeline Stages
1. **Test**: Run unit tests and integration tests
2. **Build**: Build Docker image and push to registry
3. **Deploy**: Deploy to staging/production environments

### Environment Variables
Set these in GitLab CI/CD settings:
- `CI_REGISTRY`: GitLab container registry URL
- `CI_REGISTRY_USER`: Registry username
- `CI_REGISTRY_PASSWORD`: Registry password

## 📊 Real-World Examples & Use Cases

### Example 1: Building Scan Analysis
```bash
# Step 1: Reconstruct from drone footage
docker run -it --rm --gpus all \
  -v $(pwd)/building_scan:/app/input:ro \
  -v $(pwd)/results:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/drone_video.mp4 \
    --output_dir /app/output \
    --recon_method vggt \
    --fps 1.5 \
    --conf_threshold 2.8

# Step 2: Segment building components
docker run -it --rm \
  -v $(pwd)/results:/app/data \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/data/vggt_point_cloud.ply \
    --output_dir /app/data \
    --enhanced_segmentation \
    --max_planes 80 \
    --distance_threshold 0.03

# Step 3: Detect diffraction edges for RF analysis
docker run -it --rm \
  -v $(pwd)/results:/app/data \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/data/segmented_point_cloud.ply \
    --output_dir /app/data \
    --create_mesh \
    --diffraction_method segmentation
```

### Example 2: Change Detection Workflow
```bash
# Compare before/after construction scans
docker run -it --rm \
  -v $(pwd)/construction_site:/app/input:ro \
  -v $(pwd)/analysis:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/input/after_construction.ply \
    --base_cloud_path /app/input/before_construction.ply \
    --output_dir /app/output \
    --enhanced_comparison \
    --use_multi_scale \
    --use_confidence \
    --use_color_coding \
    --comparison_threshold 0.2 \
    --selective_segmentation
```

### Example 3: Multi-Camera Rendering for Presentation
```bash
# Generate 8-view renders for client presentation
curl -X POST "http://localhost:8001/render-cameras" \
  -H "Content-Type: application/json" \
  -d '{
    "point_cloud_path": "/data/final_model.ply",
    "width": 2560,
    "height": 1440,
    "background_color": [0.95, 0.95, 0.95]
  }' | jq '.images[]'
```

### Visualization Examples

#### Corner View (Wide FOV)
```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/output:/app/output:ro \
  vggt-3d-pipeline \
  conda run -n mast3r python visualization.py \
    --point_cloud /app/output/scan.ply \
    --camera_position "10,10,10" \
    --camera_fov 90 \
    --camera_zoom 0.6
```

#### Close-up View (Narrow FOV)
```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/output:/app/output:ro \
  vggt-3d-pipeline \
  conda run -n mast3r python visualization.py \
    --point_cloud /app/output/scan.ply \
    --camera_position "2,2,2" \
    --camera_fov 30 \
    --camera_zoom 1.2
```

#### Top-down View
```bash
docker run -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/output:/app/output:ro \
  vggt-3d-pipeline \
  conda run -n mast3r python visualization.py \
    --point_cloud /app/output/scan.ply \
    --camera_position "0,15,0" \
    --camera_look_at "0,0,0" \
    --camera_up "0,0,-1"
```

## 🔧 Configuration & Parameters

### Environment Variables
- `DISPLAY`: X11 display for visualization
- `PYTHONUNBUFFERED`: Unbuffered Python output
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `TORCH_HOME`: PyTorch model cache directory
- `HF_HOME`: Hugging Face cache directory

### Volume Mounts
- `/app/input`: Read-only input data
- `/app/output`: Writable output directory
- `/app/test_output`: Test data directory
- `/app/vggt`: VGGT model and dependencies

### Key Parameters Guide

#### VGGT Parameters
- `--conf_threshold`: Point confidence threshold (1.0-5.0, default: 3.0)
- `--fps`: Video frame extraction rate (0.5-5.0, default: 1.0)
- `--show_cameras`: Include camera positions in output
- `--no_cameras`: Exclude camera positions

#### Segmentation Parameters
- `--max_planes`: Maximum planes to detect (10-100, default: 60)
- `--distance_threshold`: Plane fitting threshold (0.01-0.1, default: 0.04)
- `--enhanced_segmentation`: Use RANSAC+DBSCAN method
- `--skip_preprocessing`: Skip outlier removal (faster but less accurate)

#### Diffraction Parameters
- `--create_mesh`: Generate surface meshes (optional)
- `--diffraction_method`: 'segmentation' or 'normals'
- `--angle_threshold`: Edge detection angle (30-60 degrees, default: 45)
- `--normal_radius`: Normal estimation radius (0.02-0.1, default: 0.05)

#### Comparison Parameters
- `--enhanced_comparison`: Use advanced multi-scale analysis
- `--comparison_threshold`: Distance threshold (0.05-0.3, default: 0.15)
- `--use_multi_scale`: Multi-resolution registration
- `--use_confidence`: Confidence-based filtering
- `--use_color_coding`: Color-coded difference visualization

## 🐛 Troubleshooting

### Common Issues

#### Docker-Specific Issues

1. **VGGT Model Download Issues**
   ```bash
   # Pre-download VGGT model
   docker run -it --rm \
     -v $(pwd)/models:/app/models \
     vggt-3d-pipeline \
     conda run -n mast3r python -c "
     import torch
     torch.hub.load_state_dict_from_url(
       'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt',
       map_location='cpu'
     )"
   ```

2. **GPU Memory Issues**
   ```bash
   # Reduce batch size and use mixed precision
   docker run -it --rm --gpus all \
     --memory=16g \
     --memory-swap=32g \
     -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
     vggt-3d-pipeline
   ```

3. **X11 Forwarding Issues**
   ```bash
   # Allow X11 connections
   xhost +local:docker
   
   # Run with X11 forwarding
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     --net=host \
     vggt-3d-pipeline
   ```

4. **CUDA Compatibility**
   ```bash
   # Check CUDA version compatibility
   docker run -it --rm --gpus all \
     vggt-3d-pipeline \
     conda run -n mast3r python -c "
     import torch
     print(f'PyTorch: {torch.__version__}')
     print(f'CUDA available: {torch.cuda.is_available()}')
     print(f'CUDA version: {torch.version.cuda}')
     print(f'GPU count: {torch.cuda.device_count()}')
     "
   ```

5. **Large Point Cloud Processing**
   ```bash
   # Use voxel downsampling for large datasets
   docker run -it --rm \
     -v $(pwd)/input:/app/input:ro \
     -v $(pwd)/output:/app/output \
     vggt-3d-pipeline \
     conda run -n mast3r python main.py \
       --point_cloud_path /app/input/large_scan.ply \
       --voxel_size 0.05 \
       --distance_threshold 0.08 \
       --max_planes 40
   ```

#### Local Conda Installation Issues

6. **Environment Creation Fails**
   ```bash
   # Clear conda cache and retry
   conda clean --all
   conda env create -f environment.yml --force
   
   # If still fails, create step by step
   conda create -n mast3r python=3.10
   conda activate mast3r
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt  # if you have one
   ```

7. **CUDA Version Mismatch**
   ```bash
   # Check your CUDA version
   nvidia-smi
   nvcc --version
   
   # Install compatible PyTorch
   conda activate mast3r
   # For CUDA 11.8
   pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   # For CUDA 12.1
   pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

8. **Import Errors**
   ```bash
   # Verify all packages are installed
   conda activate mast3r
   python -c "import torch, open3d, trimesh, gradio; print('All imports successful')"
   
   # If specific package fails, reinstall
   pip install --force-reinstall package_name
   ```

9. **Permission Issues**
   ```bash
   # Fix conda permissions
   sudo chown -R $USER:$USER ~/anaconda3/
   # or
   sudo chown -R $USER:$USER ~/miniconda3/
   ```

10. **Out of Memory Issues**
    ```bash
    # Monitor memory usage
    nvidia-smi -l 1
    
    # Reduce batch size and use CPU fallback
    export CUDA_VISIBLE_DEVICES=""  # Force CPU
    python main.py --voxel_size 0.1 --max_planes 20  # Reduce complexity
    ```

### Debug Mode
```bash
# Run with debug output and verbose logging
docker run -it --rm \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/test_images \
    --output_dir /app/output \
    --recon_method vggt \
    --conf_threshold 3.0

# Interactive debugging session
docker run -it --rm --gpus all \
  -v $(pwd):/app/workspace \
  vggt-3d-pipeline \
  bash
```

## 📈 Performance Optimization

### VGGT Optimization
- Use GPU with at least 8GB VRAM for best performance
- Adjust `conf_threshold` based on scene complexity (lower = more points)
- Use appropriate `fps` for video input (1-2 fps usually sufficient)
- Enable mixed precision training for memory efficiency

### Docker Optimization
- Use multi-stage builds for smaller images
- Leverage Docker layer caching
- Use `.dockerignore` to exclude unnecessary files
- Mount model cache directories to avoid re-downloading

### Pipeline Optimization
- Use appropriate voxel sizes for your data (0.02-0.05m typical)
- Adjust segmentation parameters based on point cloud density
- Use GPU acceleration when available
- Enable enhanced methods only when needed (they're slower but more accurate)

### Memory Management
```bash
# For large datasets, use these settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
docker run --memory=16g --memory-swap=32g --gpus all ...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a merge request

### Development Setup
```bash
# Clone and setup development environment
git clone <your-gitlab-repo-url>
cd 3d_converge

# Initialize VGGT submodule
git submodule update --init --recursive

# Start development container with GPU support
docker-compose run --rm --gpus all mast3r-dev

# Inside container, activate environment
conda activate mast3r

# Install VGGT dependencies
cd vggt && pip install -e .

# Run comprehensive tests
python test_visualization.py
python -m pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VGGT**: Facebook Research for the VGGT neural 3D reconstruction model
- **Open3D**: For comprehensive 3D visualization and processing capabilities
- **COLMAP**: For robust structure-from-motion reconstruction
- **MAST3R**: For advanced neural reconstruction techniques
- **PyTorch**: For deep learning framework and CUDA support
- **Hugging Face**: For model hosting and transformers library
- **GitLab**: For CI/CD infrastructure and container registry

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues in GitLab
3. Create a new issue with detailed information
4. Contact the development team

---

## 🚀 Quick Start Commands

### Docker Quick Start
```bash
# 1. Clone and setup
git clone <your-repo> && cd 3d_converge
docker-compose build

# 2. Run VGGT reconstruction from video
docker run -it --rm --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/video.mp4 \
    --recon_method vggt --fps 1.5

# 3. Generate multi-camera renders
curl -X POST "http://localhost:8001/render-cameras" \
  -H "Content-Type: application/json" \
  -d '{"point_cloud_path": "/data/output/vggt_point_cloud.ply"}'

# 4. Compare point clouds with enhanced analysis
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  vggt-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/data/new.ply \
    --base_cloud_path /app/data/reference.ply \
    --enhanced_comparison
```

### Local Conda Quick Start
```bash
# 1. Clone and setup environment
git clone <your-repo> && cd 3d_converge
conda env create -f environment.yml
conda activate mast3r

# 2. Run VGGT reconstruction from video
python main.py \
  --image_path input/video.mp4 \
  --recon_method vggt --fps 1.5

# 3. Start FastAPI services (in separate terminals)
cd converge-3ds/3D_Pipeline && uvicorn app.main:app --port 8000
cd converge-3ds/Visualization && uvicorn app.main:app --port 8001

# 4. Compare point clouds with enhanced analysis
python main.py \
  --point_cloud_path data/new.ply \
  --base_cloud_path data/reference.ply \
  --enhanced_comparison
```

**Note**: This pipeline is designed for electromagnetic simulation applications and includes specialized diffraction edge detection for RF propagation modeling with state-of-the-art neural reconstruction capabilities. 