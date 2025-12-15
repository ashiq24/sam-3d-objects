#!/usr/bin/env python3
"""Visualize mesh from multiple viewpoints and save renders to disk."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

# 
def flatten_multiblock(mb: pv.MultiBlock) -> list:
    """Recursively flatten a MultiBlock structure to get all leaf meshes."""
    meshes = []
    for block in mb:
        if block is None:
            continue
        elif isinstance(block, pv.MultiBlock):
            # Recursively flatten nested MultiBlocks
            meshes.extend(flatten_multiblock(block))
        elif hasattr(block, 'n_points') and block.n_points > 0:
            # Valid mesh with points
            meshes.append(block)
    return meshes


def render_turntable_views(
    mesh_path: Path,
    output_dir: Path,
    num_views: int = 8,
    num_elevations: int = 3,
    resolution: tuple[int, int] = (1024, 1024),
    save_grid: bool = True,
) -> None:
    """
    Render mesh from multiple viewpoints with varying azimuth and elevation.
    
    Args:
        mesh_path: Path to the mesh file (GLB, OBJ, etc.)
        output_dir: Directory to save rendered views
        num_views: Number of evenly-spaced azimuthal angles
        num_elevations: Number of elevation angles (polar angles)
        resolution: Image resolution (width, height)
        save_grid: If True, save a grid of all views; if False, save individual images
    """
    print(f"[INFO] Loading mesh from {mesh_path}")
    
    # Load mesh with PyVista
    loaded = pv.read(str(mesh_path))
    
    # Handle MultiBlock (GLB files with multiple meshes)
    if isinstance(loaded, pv.MultiBlock):
        # Flatten nested MultiBlock structure
        all_meshes = flatten_multiblock(loaded)
        
        print(f"  Found {len(all_meshes)} mesh(es) in file")
        for idx, mesh in enumerate(all_meshes):
            print(f"  Mesh {idx}: {type(mesh).__name__}, {mesh.n_points:,} points, {mesh.n_cells:,} cells")
        
        if not all_meshes:
            raise ValueError("No valid meshes found in file")
        
        # Combine all meshes
        if len(all_meshes) == 1:
            pv_mesh = all_meshes[0]
        else:
            # Merge all meshes into one
            pv_mesh = all_meshes[0]
            for mesh in all_meshes[1:]:
                pv_mesh = pv_mesh + mesh
    else:
        pv_mesh = loaded
    
    print(f"[INFO] Mesh loaded: {pv_mesh.n_points:,} vertices, {pv_mesh.n_cells:,} faces")
    
    # Center the mesh
    center = np.array(pv_mesh.center)
    pv_mesh.translate(-center, inplace=True)
    
    # Calculate camera distance based on bounds
    bounds = pv_mesh.bounds
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    camera_distance = max_dim * 2.5
    
    # Define camera viewpoints: azimuth x elevation grid + top + bottom
    total_views = num_views * num_elevations + 2  # Grid + top + bottom views
    print(f"[INFO] Rendering {total_views} views ({num_views} azimuth × {num_elevations} elevation + top + bottom)...")
    
    # Generate azimuthal angles (0° to 360°)
    azimuth_angles = np.linspace(0, 360, num_views, endpoint=False)
    
    # Generate elevation angles (0° = horizontal, 90° = top)
    # Exclude 0° and 90° as we'll handle those separately with top/bottom
    if num_elevations == 1:
        elevation_angles = [15]  # Single mid-level view
    elif num_elevations == 2:
        elevation_angles = [15, 45]
    else:
        elevation_angles = np.linspace(10, 70, num_elevations)
    
    images = []
    view_labels = []
    view_idx = 0
    
    def add_mesh_to_plotter(plotter, mesh):
        """Helper to add mesh with texture/color support."""
        if hasattr(mesh, 'texture') and mesh.texture is not None:
            plotter.add_mesh(mesh, texture=mesh.texture, smooth_shading=True, show_edges=False, lighting=True)
        elif hasattr(mesh, 'active_scalars') and mesh.active_scalars is not None:
            plotter.add_mesh(mesh, scalars=mesh.active_scalars, smooth_shading=True, show_edges=False, 
                           lighting=True, rgb=True if mesh.active_scalars.ndim > 1 else False)
        else:
            plotter.add_mesh(mesh, smooth_shading=True, color='white', show_edges=False, lighting=True)
    
    # Render grid of azimuth × elevation views
    for elev in elevation_angles:
        for azim in azimuth_angles:
            plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
            add_mesh_to_plotter(plotter, pv_mesh)
            
            # Convert spherical to cartesian coordinates
            azim_rad = np.deg2rad(azim)
            elev_rad = np.deg2rad(elev)
            
            # Spherical to Cartesian: 
            # x = r * cos(elev) * cos(azim)
            # y = r * cos(elev) * sin(azim)
            # z = r * sin(elev)
            camera_pos = [
                camera_distance * np.cos(elev_rad) * np.cos(azim_rad),
                camera_distance * np.cos(elev_rad) * np.sin(azim_rad),
                camera_distance * np.sin(elev_rad)
            ]
            plotter.camera_position = [camera_pos, [0, 0, 0], [0, 0, 1]]
            
            # Add multiple lights
            plotter.add_light(pv.Light(position=(10, 10, 10), intensity=0.6))
            plotter.add_light(pv.Light(position=(-10, -10, 10), intensity=0.3))
            plotter.add_light(pv.Light(position=(0, 0, -10), intensity=0.2))
            
            img = plotter.screenshot(return_img=True)
            images.append(img)
            view_labels.append(f'Az:{int(azim)}° El:{int(elev)}°')
            plotter.close()
            
            if not save_grid:
                view_path = output_dir / f"view_{view_idx:03d}_az{int(azim):03d}_el{int(elev):02d}.png"
                plt.imsave(view_path, img)
                print(f"  Saved view {view_idx+1}/{total_views}: {view_path.name}")
            
            view_idx += 1
    
    # Render top view (90° elevation)
    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    add_mesh_to_plotter(plotter, pv_mesh)
    camera_pos = [0, 0, camera_distance]
    plotter.camera_position = [camera_pos, [0, 0, 0], [0, 1, 0]]  # Up vector along Y
    plotter.add_light(pv.Light(position=(0, 0, 10), intensity=0.8))
    plotter.add_light(pv.Light(position=(5, 5, 10), intensity=0.3))
    img = plotter.screenshot(return_img=True)
    images.append(img)
    view_labels.append('Top View (90°)')
    plotter.close()
    
    if not save_grid:
        view_path = output_dir / f"view_{view_idx:03d}_top.png"
        plt.imsave(view_path, img)
        print(f"  Saved view {view_idx+1}/{total_views}: {view_path.name}")
    view_idx += 1
    
    # Render bottom view (-90° elevation)
    plotter = pv.Plotter(off_screen=True, window_size=list(resolution))
    add_mesh_to_plotter(plotter, pv_mesh)
    camera_pos = [0, 0, -camera_distance]
    plotter.camera_position = [camera_pos, [0, 0, 0], [0, 1, 0]]  # Up vector along Y
    plotter.add_light(pv.Light(position=(0, 0, -10), intensity=0.8))
    plotter.add_light(pv.Light(position=(5, 5, -10), intensity=0.3))
    img = plotter.screenshot(return_img=True)
    images.append(img)
    view_labels.append('Bottom View (-90°)')
    plotter.close()
    
    if not save_grid:
        view_path = output_dir / f"view_{view_idx:03d}_bottom.png"
        plt.imsave(view_path, img)
        print(f"  Saved view {view_idx+1}/{total_views}: {view_path.name}")
    
    # Save grid of all views
    if save_grid:
        cols = 4
        rows = (total_views + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if total_views > 1 else [axes]
        
        for idx, (label, img) in enumerate(zip(view_labels, images)):
            axes[idx].imshow(img)
            axes[idx].set_title(label, fontsize=12, fontweight='bold' if 'View' in label else 'normal')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(total_views, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        grid_path = output_dir / f"{mesh_path.stem}_turntable.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved turntable grid: {grid_path}")
    
    # Save mesh statistics
    stats_path = output_dir / f"{mesh_path.stem}_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"Mesh: {mesh_path.name}\n")
        f.write(f"Vertices: {pv_mesh.n_points:,}\n")
        f.write(f"Faces: {pv_mesh.n_cells:,}\n")
        f.write(f"Bounds (x): [{bounds[0]:.3f}, {bounds[1]:.3f}]\n")
        f.write(f"Bounds (y): [{bounds[2]:.3f}, {bounds[3]:.3f}]\n")
        f.write(f"Bounds (z): [{bounds[4]:.3f}, {bounds[5]:.3f}]\n")
    print(f"[OK] Saved statistics: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mesh_path",
        type=str,
        help="Path to mesh file (GLB, OBJ, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for rendered views (default: mesh_name_views/)"
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=8,
        help="Number of azimuthal views around the turntable (default: 8)"
    )
    parser.add_argument(
        "--num-elevations",
        type=int,
        default=3,
        help="Number of elevation angles (default: 3)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution in pixels (default: 1024)"
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Save individual view images instead of a grid"
    )
    
    args = parser.parse_args()
    
    mesh_path = Path(args.mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = mesh_path.parent / f"{mesh_path.stem}_views"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render views
    render_turntable_views(
        mesh_path=mesh_path,
        output_dir=output_dir,
        num_views=args.num_views,
        num_elevations=args.num_elevations,
        resolution=(args.resolution, args.resolution),
        save_grid=not args.individual
    )
    
    print(f"\n[DONE] All renders saved to: {output_dir}")


if __name__ == "__main__":
    main()
