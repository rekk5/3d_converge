"""
Centralized naming convention for the 3D reconstruction pipeline.
This ensures consistent file and directory naming across all components.
"""

import os
from typing import Dict, Any

class NamingConvention:
    """Centralized naming convention for pipeline outputs."""
    
    def __init__(self, base_output_dir: str = "output", project_name: str = "reconstruction"):
        self.base_output_dir = base_output_dir
        self.project_name = project_name
        
        # Create subdirectories
        self.dirs = {
            'base': base_output_dir,
            'reconstruction': os.path.join(base_output_dir, 'reconstruction'),
            'segmentation': os.path.join(base_output_dir, 'segmentation'), 
            'diffraction': os.path.join(base_output_dir, 'diffraction'),
            'comparison': os.path.join(base_output_dir, 'comparison'),
            'visualization': os.path.join(base_output_dir, 'visualization'),
            'intermediate': os.path.join(base_output_dir, 'intermediate')
        }
        
        # Standard file extensions
        self.extensions = {
            'point_cloud': '.ply',
            'mesh': '.ply', 
            'config': '.yaml',
            'log': '.log',
            'image': '.png'
        }
    
    def get_file_path(self, category: str, file_type: str, suffix: str = "") -> str:
        """
        Generate standardized file paths.
        
        Args:
            category: Directory category (reconstruction, segmentation, etc.)
            file_type: Type of file (point_cloud, mesh, etc.)
            suffix: Optional suffix for the filename
        
        Returns:
            Full file path with standardized naming
        """
        if suffix:
            filename = f"{self.project_name}_{file_type}_{suffix}{self.extensions.get(file_type, '.ply')}"
        else:
            filename = f"{self.project_name}_{file_type}{self.extensions.get(file_type, '.ply')}"
        
        return os.path.join(self.dirs[category], filename)
    
    def get_dir_path(self, category: str) -> str:
        """Get directory path for a category."""
        return self.dirs[category]
    
    def create_directories(self):
        """Create all necessary directories."""
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    # Standard filename patterns
    @property
    def file_patterns(self) -> Dict[str, str]:
        """Standard file naming patterns for the pipeline."""
        return {
            # Reconstruction outputs
            'colmap_point_cloud': self.get_file_path('reconstruction', 'point_cloud', 'colmap'),
            'tsdf_point_cloud': self.get_file_path('reconstruction', 'point_cloud', 'tsdf'),
            'mast3r_point_cloud': self.get_file_path('reconstruction', 'point_cloud', 'mast3r'),
            "vggt_point_cloud": self.get_file_path("reconstruction", "point_cloud", "vggt"),
            'tsdf_mesh': self.get_file_path('reconstruction', 'mesh', 'tsdf'),
            
            # Segmentation outputs
            'segmented_point_cloud': self.get_file_path('segmentation', 'point_cloud', 'segmented'),
            'segmented_differential': self.get_file_path('segmentation', 'point_cloud', 'segmented_differential'),
            
            # Diffraction outputs
            'diffraction_mesh': self.get_file_path('diffraction', 'mesh', 'with_edges'),
            'edge_points': self.get_file_path('diffraction', 'point_cloud', 'edges'),
            
            # Comparison outputs
            'differential_comparison': self.get_file_path('comparison', 'point_cloud', 'differential'),
            'base_vs_target': self.get_file_path('comparison', 'point_cloud', 'base_vs_target'),
            
            # Intermediate files
            'preprocessed': self.get_file_path('intermediate', 'point_cloud', 'preprocessed'),
            'downsampled': self.get_file_path('intermediate', 'point_cloud', 'downsampled'),
        }

def get_naming_convention(base_output_dir: str = "output", project_name: str = "reconstruction") -> NamingConvention:
    """Factory function to get a naming convention instance."""
    return NamingConvention(base_output_dir, project_name)

# Default instance for backward compatibility
default_naming = get_naming_convention() 