#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# /// script
# dependencies = [
#   "open3d>=0.16.0",
#   "numpy",
# ]
# ///

"""
OBJ Folder Renderer
===================
Loads all OBJ files in a directory and renders them interactively in a 3D window.
Each OBJ is painted with a random color.

Usage:
  uv run render_objs.py -d ./OBJ
"""

import os
# Force GLFW to use X11/XWayland to avoid Wayland-specific OpenGL/GLEW initialization errors in Open3D
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.pop("WAYLAND_DISPLAY", None)
import argparse
import random
from pathlib import Path
import open3d as o3d
import numpy as np

def render_folder(folder_path: str) -> None:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder does not exist: {folder}")
        return

    # Find all OBJ files
    obj_files = [folder / f for f in os.listdir(folder) if f.lower().endswith('.obj')]
    if not obj_files:
        print(f"No OBJ files found in: {folder}")
        return

    print(f"Found {len(obj_files)} OBJ files. Loading and coloring...")

    geometries = []
    
    # Generate distinct random colors
    # Using HSL/HSV representation to ensure colors are bright and distinct
    random.seed(42)  # Seed for reproducible random colors
    
    for idx, filepath in enumerate(obj_files):
        try:
            # Read the mesh
            mesh = o3d.io.read_triangle_mesh(str(filepath))
            
            # If the mesh has no vertices, skip
            if not mesh.has_vertices():
                print(f"Skipping empty mesh: {filepath.name}")
                continue
                
            # Compute normals if they are missing (for proper shading)
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            # Generate a bright random color
            hue = random.random()
            saturation = 0.7 + 0.3 * random.random()  # 70% to 100%
            value = 0.6 + 0.3 * random.random()       # 60% to 90%
            
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            
            # Paint the mesh
            mesh.paint_uniform_color([r, g, b])
            geometries.append(mesh)
            
            print(f"  [{idx + 1}/{len(obj_files)}] Loaded {filepath.name} with color (R:{r:.2f}, G:{g:.2f}, B:{b:.2f})")
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    if not geometries:
        print("No valid geometry could be loaded.")
        return

    # Add a coordinate frame at the origin for reference (size: 1.0 meter)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    geometries.append(coord_frame)

    print("\nStarting Interactive 3D Viewer...")
    print("-----------------------------------")
    print("Viewer Controls:")
    print("  - Left Mouse Button + Drag : Rotate camera")
    print("  - Right Mouse Button + Drag: Pan camera")
    print("  - Scroll Wheel             : Zoom in/out")
    print("  - Key 'W'                  : Toggle wireframe mode")
    print("  - Key 'R'                  : Reset camera view")
    print("  - Key 'H'                  : Print help info in console")
    print("  - Press 'Esc' or Close Window to exit")
    print("-----------------------------------")

    # Launch visualizer
    o3d.visualization.draw_geometries(
        geometries,
        window_name="IFC Component OBJ Renderer",
        width=1024,
        height=768,
        left=50,
        top=50,
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render all OBJ files in a directory with random colors.")
    parser.add_argument("-d", "--dir", type=str, default="./OBJ", help="Directory containing OBJ files (default: ./OBJ)")
    args = parser.parse_args()
    
    render_folder(args.dir)
