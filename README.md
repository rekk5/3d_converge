# 3D Reconstruction Pipeline with Diffraction Edge Detection

A comprehensive 3D reconstruction pipeline that combines multiple reconstruction methods (COLMAP, TSDF Fusion, MAST3R) with advanced point cloud segmentation and diffraction edge detection for electromagnetic simulation applications.

## 🚀 Features

- **Multiple Reconstruction Methods**: COLMAP, TSDF Fusion, and MAST3R
- **Advanced Segmentation**: Enhanced RANSAC + DBSCAN plane segmentation
- **Diffraction Edge Detection**: Automatic detection of electromagnetic diffraction edges
- **Interactive Visualization**: Customizable camera views with Open3D
- **Docker Support**: Complete containerization for easy deployment
- **CI/CD Pipeline**: Automated testing and deployment with GitLab CI

## 📋 Prerequisites

- Docker and Docker Compose
- GitLab account (for CI/CD)
- NVIDIA GPU (optional, for CUDA acceleration)

## 🐳 Quick Start with Docker

### 1. Clone the Repository
```bash
git clone <your-gitlab-repo-url>
cd 3d_converge
```

### 2. Build and Run with Docker Compose
```bash
# Build the Docker image
docker-compose build

# Run the pipeline
docker-compose up mast3r-pipeline

# For development with interactive shell
docker-compose run --rm mast3r-dev

# For visualization
docker-compose run --rm mast3r-viz
```

### 3. Using Docker Directly
```bash
# Build the image
docker build -t mast3r-3d-pipeline .

# Run with volume mounts
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/test_output:/app/test_output \
  mast3r-3d-pipeline \
  conda run -n mast3r python main.py --help
```

## 📁 Project Structure

```
3d_converge/
├── main.py                 # Main pipeline entry point
├── comparison.py           # Point cloud comparison utilities
├── segmentation.py         # Point cloud segmentation
├── diffraction.py          # Diffraction edge detection
├── visualization.py        # Interactive visualization tool
├── test_visualization.py   # Visualization testing
├── example_visualization.py # Example camera configurations
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
├── environment.yml         # Conda environment
├── .gitlab-ci.yml          # GitLab CI/CD pipeline
├── input/                  # Input images/data (mounted volume)
├── output/                 # Pipeline outputs (mounted volume)
└── test_output/            # Test data (mounted volume)
```

## 🔧 Usage

### Basic Pipeline Execution
```bash
# Run complete pipeline with MAST3R reconstruction
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  mast3r-3d-pipeline \
  conda run -n mast3r python main.py \
    --image_path /app/input/images \
    --output_dir /app/output \
    --recon_method mast3r
```

### Point Cloud Comparison
```bash
# Compare two point clouds
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  mast3r-3d-pipeline \
  conda run -n mast3r python main.py \
    --point_cloud_path /app/input/base_cloud.ply \
    --base_cloud_path /app/input/base_cloud.ply \
    --comparison_threshold 0.15
```

### Visualization
```bash
# Interactive visualization with corner view
docker run -it --rm \
  -v $(pwd)/output:/app/output:ro \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  mast3r-3d-pipeline \
  conda run -n mast3r python visualization.py \
    --point_cloud /app/output/point_cloud.ply \
    --mesh_with_edges /app/output/saalas_mesh.ply \
    --camera_position "10,10,10" \
    --camera_fov 90 \
    --auto_advance
```

## 🧪 Testing

### Run Tests
```bash
# Test visualization functionality
docker run -it --rm \
  -v $(pwd)/test_output:/app/test_output \
  mast3r-3d-pipeline \
  conda run -n mast3r python test_visualization.py
```

### Test Individual Components
```bash
# Test segmentation
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  mast3r-3d-pipeline \
  conda run -n mast3r python segmentation.py /app/input/point_cloud.ply

# Test diffraction edge detection
docker run -it --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  mast3r-3d-pipeline \
  conda run -n mast3r python diffraction.py \
    /app/input/segmented.ply \
    /app/output/saalas_mesh.ply
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

## 📊 Visualization Examples

### Corner View (Wide FOV)
```bash
python visualization.py \
  --camera_position "10,10,10" \
  --camera_fov 90 \
  --camera_zoom 0.6
```

### Close-up View (Narrow FOV)
```bash
python visualization.py \
  --camera_position "2,2,2" \
  --camera_fov 30 \
  --camera_zoom 1.2
```

### Top-down View
```bash
python visualization.py \
  --camera_position "0,15,0" \
  --camera_look_at "0,0,0" \
  --camera_up "0,0,-1"
```

## 🔧 Configuration

### Environment Variables
- `DISPLAY`: X11 display for visualization
- `PYTHONUNBUFFERED`: Unbuffered Python output
- `CUDA_VISIBLE_DEVICES`: GPU device selection

### Volume Mounts
- `/app/input`: Read-only input data
- `/app/output`: Writable output directory
- `/app/test_output`: Test data directory

## 🐛 Troubleshooting

### Common Issues

1. **X11 Forwarding Issues**
   ```bash
   # Allow X11 connections
   xhost +local:docker
   
   # Run with X11 forwarding
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     mast3r-3d-pipeline
   ```

2. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   docker run -it --rm \
     --memory=8g \
     --memory-swap=16g \
     mast3r-3d-pipeline
   ```

3. **GPU Support**
   ```bash
   # Run with NVIDIA GPU support
   docker run -it --rm \
     --gpus all \
     -e NVIDIA_VISIBLE_DEVICES=all \
     mast3r-3d-pipeline
   ```

### Debug Mode
```bash
# Run with debug output
docker run -it --rm \
  -e PYTHONUNBUFFERED=1 \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  mast3r-3d-pipeline \
  conda run -n mast3r python main.py --verbose
```

## 📈 Performance Optimization

### Docker Optimization
- Use multi-stage builds for smaller images
- Leverage Docker layer caching
- Use `.dockerignore` to exclude unnecessary files

### Pipeline Optimization
- Use appropriate voxel sizes for your data
- Adjust segmentation parameters based on point cloud density
- Use GPU acceleration when available

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

# Start development container
docker-compose run --rm mast3r-dev

# Inside container, activate environment
conda activate mast3r

# Run tests
python test_visualization.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Open3D for 3D visualization and processing
- COLMAP for structure-from-motion reconstruction
- MAST3R for neural reconstruction
- GitLab for CI/CD infrastructure

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues in GitLab
3. Create a new issue with detailed information
4. Contact the development team

---

**Note**: This pipeline is designed for electromagnetic simulation applications and includes specialized diffraction edge detection for RF propagation modeling. 