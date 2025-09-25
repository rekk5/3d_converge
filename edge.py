import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection

def plot_enhanced_diffraction():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add walls/planes
    wall1 = Rectangle((-0.1, 0), 0.2, 2, angle=0, color='gray', alpha=0.5)
    wall2 = Rectangle((0, -0.1), 2, 0.2, angle=0, color='gray', alpha=0.5)
    ax.add_patch(wall1)
    ax.add_patch(wall2)
    
    # Plot the edge (represented as a small circle/cylinder)
    edge_radius = 0.1
    edge_center = (0, 0)
    edge = Circle(edge_center, edge_radius, color='black', fill=True)
    ax.add_patch(edge)
    
    # Create multiple incident rays
    ray_starts = [(-2, y) for y in np.linspace(0.5, 1.5, 4)]
    
    for ray_start in ray_starts:
        # Plot incident ray
        ax.plot([ray_start[0], edge_center[0]], [ray_start[1], edge_center[1]], 
                'r-', linewidth=1, alpha=0.7)
        
        # Calculate reflection angle based on incident angle
        incident_vector = np.array([edge_center[0] - ray_start[0], 
                                  edge_center[1] - ray_start[1]])
        incident_angle = np.arctan2(incident_vector[1], incident_vector[0])
        
        # Add reflected and scattered rays
        num_scattered = 15  # Increased number of scattered rays
        # Wider angle range for more realistic diffraction
        scatter_angles = np.linspace(incident_angle - np.pi/2, 
                                   incident_angle + np.pi/2, 
                                   num_scattered)
        
        # Calculate intensity based on angle difference from specular reflection
        for angle in scatter_angles:
            scatter_length = 2.0  # Length of scattered rays
            scatter_end = (
                edge_center[0] + scatter_length * np.cos(angle),
                edge_center[1] + scatter_length * np.sin(angle)
            )
            
            # Calculate intensity based on angle difference from specular reflection
            angle_diff = abs(angle - incident_angle)
            intensity = np.exp(-2 * angle_diff)  # Exponential decay
            
            # Plot scattered ray with varying intensity
            ax.plot([edge_center[0], scatter_end[0]], 
                   [edge_center[1], scatter_end[1]],
                   'g-', alpha=intensity * 0.5, linewidth=1)
    
    # Add annotations
    ax.annotate('Wall', (-0.2, 1.5), rotation=90)
    ax.annotate('Wall', (1.0, -0.3))
    ax.annotate('Diffraction Edge', (0.2, 0.2))
    
    # Set axis limits and labels
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Enhanced Radio Ray Diffraction at Corner Edge')
    
    # Add legend with custom entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', label='Incident Rays'),
        Line2D([0], [0], color='g', label='Diffracted Rays'),
        Rectangle((0,0), 1, 1, color='gray', alpha=0.5, label='Walls')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.show()

if __name__ == "__main__":
    plot_enhanced_diffraction()
