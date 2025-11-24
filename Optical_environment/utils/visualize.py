import vtk
import os
from optiland.visualization import OpticViewer3D
from pathlib import Path
from .computing import compute_irr, compute_irr_with_aperture_viz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def save_3d_snapshot(sys, filename="optic_view.png", width=800, height=600, dark_mode=False, **kwargs):
    """
    Render a 3D optical system visualization and save it as a PNG image (off-screen).
    """
    # Create the 3D viewer
    viewer = OpticViewer3D(sys)
    renderer = vtk.vtkRenderer()

    # Enable off-screen rendering to speed up and avoid showing a window
    viewer.ren_win.SetOffScreenRendering(1)

    # Add renderer to the render window
    viewer.ren_win.AddRenderer(renderer)
    viewer.iren.SetRenderWindow(viewer.ren_win)

    # Draw rays and optical system
    viewer.rays.plot(renderer, **kwargs)
    viewer.system.plot(renderer)

    # Set background style (gradient vertical)
    renderer.GradientBackgroundOn()
    renderer.SetGradientMode(vtk.vtkViewport.GradientModes.VTK_GRADIENT_VERTICAL)

    # Choose background colors
    if dark_mode:
        renderer.SetBackground(0.13, 0.15, 0.19)   # bottom color
        renderer.SetBackground2(0.20, 0.22, 0.26)  # top color
    else:
        renderer.SetBackground(0.8, 0.9, 1.0)      # bottom color
        renderer.SetBackground2(0.4, 0.5, 0.6)     # top color

    # Configure rendering window size and render once
    viewer.ren_win.SetSize(width, height)
    viewer.ren_win.Render()

    # Set a fixed camera position for consistent snapshots
    camera = renderer.GetActiveCamera()
    camera.SetPosition(1, 0, 0)     # camera location
    camera.SetFocalPoint(0, 0, 0)   # look-at point
    camera.SetViewUp(0, 1, 0)       # camera "up" direction
    renderer.ResetCamera()
    camera.Elevation(0)
    camera.Azimuth(150)
    viewer.ren_win.Render()

    # Capture the rendered image into a PNG
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(viewer.ren_win)
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

def visualize_optimization_process(
    sys,
    params_log,
    detector_mirror=3,
    detector_surface1=5,
    detector_surface2=8,
    aperture_radius=1.0,
    mirror_radius=20.0,
    num_rays=1000,
    res=(256, 256),
    output_base_dir='./optimization_visualization',
    save_3d=True,
    save_irr=True,
    cmap='gray',   
    Transpose=True,  # transpose is very important when optiland backen is numpy, use true
    **kwargs
):
    """
    Reproduce the optimization process and save visualization results,
    including aperture, optional physical size boundary, and optional loss trend.
    """
    base_path = Path(output_base_dir)
    irr_dir = base_path / 'irradiance'
    vis_3d_dir = base_path / '3d_view'

    irr_dir.mkdir(parents=True, exist_ok=True)
    if save_3d:
        vis_3d_dir.mkdir(parents=True, exist_ok=True)

    print(f"Replaying optimization process ({len(params_log)} iterations)...")

    for i, p in enumerate(params_log):
        iteration = p['iteration']
        rx1, ry1, rx2, ry2 = map(float, [p['rx1'], p['ry1'], p['rx2'], p['ry2']])
        print(f"[{i+1}/{len(params_log)}] Iteration {iteration}")


        # Update mirror angles
        sys.set_mirror_angle(rx1=rx1, ry1=ry1, rx2=rx2, ry2=ry2, gradient=False)

        # ----- Save irradiance maps -----
        if save_irr:
            irr1, x_edges, y_edges, mirror_aperture_info, _, _ = compute_irr_with_aperture_viz(
                sys,
                aperture_radius=mirror_radius,
                detector_surface=detector_mirror,
                show_plot=False,
                linewidth=2,
                num_rays=num_rays,
                res=res
            )
            irr2, x_edges, y_edges, aperture_info, _, _ = compute_irr_with_aperture_viz(
                sys,
                aperture_radius=aperture_radius,
                detector_surface=detector_surface1,
                show_plot=False,
                linewidth=2,
                num_rays=num_rays,
                res=res
            )
            irr3, x_edges, y_edges, _, _, _ = compute_irr_with_aperture_viz(
                sys,
                aperture_radius=aperture_radius,
                detector_surface=detector_surface2,
                show_plot=False,
                linewidth=2,
                num_rays=num_rays,
                res=res
            )

            # --- Convert to NumPy if tensors ---
            if isinstance(irr1, torch.Tensor):
                irr1 = irr1.detach().cpu().numpy()
            if isinstance(irr2, torch.Tensor):
                irr2 = irr2.detach().cpu().numpy()
            if isinstance(irr3, torch.Tensor):
                irr3 = irr3.detach().cpu().numpy()

            # --- Set up figure ---
            fig = plt.figure(figsize=(15, 4))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])

            # --- Plot irradiance maps ---
            vmin, vmax = 0, 1
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            if Transpose:
                im1 = ax1.imshow(irr1.T, cmap=cmap, vmin=vmin, vmax=vmax,
                                extent=extent, origin="lower")
                im2 = ax2.imshow(irr2.T, cmap=cmap, vmin=vmin, vmax=vmax,
                                extent=extent, origin="lower")
                im3 = ax3.imshow(irr3.T, cmap=cmap, vmin=vmin, vmax=vmax,
                             extent=extent, origin="lower")
            else:
                im1 = ax1.imshow(irr1, cmap=cmap, vmin=vmin, vmax=vmax,
                                extent=extent, origin="lower")
                im2 = ax2.imshow(irr2, cmap=cmap, vmin=vmin, vmax=vmax,
                                extent=extent, origin="lower")
                im3 = ax3.imshow(irr3, cmap=cmap, vmin=vmin, vmax=vmax,
                                extent=extent, origin="lower")
            ax1.set_title("Mirror 2 Profile")
            ax2.set_title("Pinhole 1 Profile")
            ax3.set_title("Pinhole 2 Profile")

            # --- Draw circles ---
            if mirror_aperture_info is not None:
                x0, y0 = mirror_aperture_info["center"]
            else:
                x0 = 0.5 * (x_edges[0] + x_edges[-1])
                y0 = 0.5 * (y_edges[0] + y_edges[-1])

            for ax in [ax1]:
                if mirror_aperture_info is not None:
                    ax.add_patch(patches.Circle(
                        (x0, y0), mirror_aperture_info["radius"],
                        fill=False, edgecolor='red',
                        linewidth=1, linestyle='-',
                        label=f"Aperture (r={mirror_aperture_info['radius']})"
                    ))
                ax.legend(loc='upper right')
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")
                ax.set_aspect('equal')

            if aperture_info is not None:
                x0, y0 = aperture_info["center"]
            else:
                x0 = 0.5 * (x_edges[0] + x_edges[-1])
                y0 = 0.5 * (y_edges[0] + y_edges[-1])
            for ax in [ax2, ax3]:
                if aperture_info is not None:
                    ax.add_patch(patches.Circle(
                        (x0, y0), aperture_info["radius"],
                        fill=False, edgecolor='red',
                        linewidth=1, linestyle='-',
                        label=f"Aperture (r={aperture_info['radius']})"
                    ))
                ax.legend(loc='upper right')
                ax.set_xlabel("X (mm)")
                ax.set_ylabel("Y (mm)")
                ax.set_aspect('equal')


            # --- Common formatting ---
            cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation="vertical", shrink=0.9)
            cbar.set_label("Irradiance (Normalized)")

            irr_filename = irr_dir / f'iter_{iteration:04d}_irradiance.png'
            plt.savefig(str(irr_filename), dpi=300, bbox_inches="tight")
            plt.close(fig)

        # ----- Save 3D view -----
        if save_3d:
            try:
                vis_3d_filename = vis_3d_dir / f'iter_{iteration:04d}_3d_view.png'
                save_3d_snapshot(sys, str(vis_3d_filename), **kwargs)
            except Exception as e:
                print(f"   [Warning] Failed to save 3D view: {e}")

    print("All visualizations completed successfully.")
    print(f"Irradiance maps: {irr_dir.resolve()}")
    if save_3d:
        print(f"   3D views: {vis_3d_dir.resolve()}")
