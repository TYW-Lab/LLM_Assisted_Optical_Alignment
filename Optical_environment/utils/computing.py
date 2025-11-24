import torch
import numpy as np
from optiland.analysis.irradiance import IncoherentIrradiance


def _to_torch(x, device):
    """Convert numpy array or list to torch tensor on specified device."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=torch.float32)
    elif torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    else:
        return torch.as_tensor(x, device=device, dtype=torch.float32)


def _compute_irradiance(optical_sys, detector_surface=-1, res=(256, 256), num_rays=100000):
    """Core irradiance computation using IncoherentIrradiance."""
    irradiance = IncoherentIrradiance(
        optical_sys,
        num_rays=num_rays,
        detector_surface=detector_surface,
        res=res,
        distribution="random",
    )
    return irradiance.data[0][0]  # Returns (irr_map, x_edges, y_edges)

def _create_circular_mask(
    x_edges,
    y_edges,
    radius,
    res=(256, 256),
    center=None,
    device=None,
    invert=False,
    smooth_ratio=0.0
):
    """
    Create a circular mask based on physical coordinates (improved version).

    Args:
        x_edges: Bin edges along X axis.
        y_edges: Bin edges along Y axis.
        radius: Radius in same units as edges.
        res (tuple): Output resolution (H, W). Default (256, 256).
        center: (x0, y0) center position, defaults to field center.
        device: Target device for tensor.
        invert (bool): If True, mask outside the circle instead of inside.
        smooth_ratio (float): Edge smoothing ratio (0 = hard edge, 0.02–0.1 = soft edge).

    Returns:
        mask (torch.Tensor): Float32 tensor (H, W), values in [0, 1].
    """

    # Compute bin centers (physical space)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    if center is None:
        x0 = 0.5 * (x_edges[0] + x_edges[-1])
        y0 = 0.5 * (y_edges[0] + y_edges[-1])
    else:
        x0, y0 = center

    # Generate physical coordinate mesh in output resolution
    x_lin = np.linspace(x_edges[0], x_edges[-1], res[1])
    y_lin = np.linspace(y_edges[0], y_edges[-1], res[0])
    X, Y = np.meshgrid(x_lin, y_lin)

    # Compute Euclidean distance in physical space
    dist = np.sqrt((X - x0)**2 + (Y - y0)**2)

    # Optional smooth edge
    if smooth_ratio > 0:
        smooth = radius * smooth_ratio
        mask = 1.0 - np.clip((dist - radius + smooth) / (2 * smooth), 0, 1)
    else:
        mask = (dist <= radius).astype(np.float32)

    if invert:
        mask = 1.0 - mask

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    return mask

def get_detector_edges(detector_size,res=(256, 256)):
    """Get Detector physical edges"""
    detector_half_width = detector_size[0]  # mm
    detector_half_height = detector_size[1]  # mm
    x_edges = np.linspace(-detector_half_width, detector_half_width, res[0]+1)
    y_edges = np.linspace(-detector_half_height, detector_half_height, res[1]+1)
    return x_edges, y_edges


def _normalize_irradiance(irr_map):
    """Normalize irradiance map to zero mean and unit variance."""
    if not torch.is_tensor(irr_map):
        irr_map = torch.tensor(irr_map, dtype=torch.float32)
    
    if torch.max(irr_map) > 0:
        irr_map = irr_map / torch.max(irr_map)
    return irr_map

def compute_irr(optical_sys, detector_surface=-1,
                        res=(256, 256), num_rays=100000,
                        normalize=True):
    """
    Compute irradiance
    
    Args:
        optical_sys: Optical system object
        detector_surface: Detector surface index
        radius: Pinhole radius in physical units
        res: Resolution (H, W)
        num_rays: Number of rays to trace
        center: (x0, y0) center position, defaults to field center
        normalize: Whether to normalize the irradiance map
    
    Returns:
        valid_irr_map: Masked irradiance map
        mask: Binary mask used
    """
    irr_map, _, _ = _compute_irradiance(
        optical_sys, detector_surface, res, num_rays
    )
    
    # Convert to torch and normalize
    if not torch.is_tensor(irr_map):
        irr_map = torch.tensor(irr_map, dtype=torch.float32)

    if normalize:
        irr_map = _normalize_irradiance(irr_map)


    return irr_map


def compute_irr_in_mask(optical_sys, detector_surface=-1, radius=1.0,
                        res=(256, 256), num_rays=100000, center=None, 
                        normalize=True):
    """
    Compute irradiance within a circular pinhole (radius=radius).
    Radius and edges are in the same physical units.
    
    Args:
        optical_sys: Optical system object
        detector_surface: Detector surface index
        radius: Pinhole radius in physical units
        res: Resolution (H, W)
        num_rays: Number of rays to trace
        center: (x0, y0) center position, defaults to field center
        normalize: Whether to normalize the irradiance map
    
    Returns:
        valid_irr_map: Masked irradiance map
        mask: Binary mask used
    """
    irr_map, x_edges, y_edges = _compute_irradiance(
        optical_sys, detector_surface, res, num_rays
    )
    
    # Convert to torch and normalize
    if not torch.is_tensor(irr_map):
        irr_map = torch.tensor(irr_map, dtype=torch.float32)
    device = irr_map.device
    
    if normalize:
        irr_map = _normalize_irradiance(irr_map)
    
    # Create circular mask
    mask = _create_circular_mask(x_edges, y_edges, radius, res=res, device=device, center=center, invert=False)
    
    valid_irr_map = irr_map * mask

    return valid_irr_map, mask


def compute_power(optical_sys, detector_surface=-1, radius=1, 
                 res=(256, 256), num_rays=100000):
    """
    Compute total power based on incoherent irradiance map within a circular region.
    
    Args:
        optical_sys: Optical system object
        detector_surface: Detector surface index
        radius: Circular region radius
        res: Resolution (H, W)
        num_rays: Number of rays to trace
    
    Returns:
        p_now: Total power within the circular region
    """
    irr_map, x_edges, y_edges = _compute_irradiance(
        optical_sys, detector_surface, res, num_rays
    )
    # Convert to torch and normalize
    if not torch.is_tensor(irr_map):
        irr_map = torch.tensor(irr_map, dtype=torch.float32)
    device = irr_map.device
    
    mask = _create_circular_mask(x_edges, y_edges, radius, res=res, device=device, invert=False)
    
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    
    
    valid_irr_map = irr_map * mask
    pixel_area = dx * dy
    p_now = torch.sum(valid_irr_map) * pixel_area
    
    return p_now


def spot_centroid_from_irr(irr_map, x_edges, y_edges, tau=0.05, eps=1e-12, 
                           res=(256, 256), negate=False):
    """
    Compute the irradiance-weighted centroid (x_c, y_c) in the same units as x_edges/y_edges.
    
    Args:
        irr_map: (H, W) irradiance map
        x_edges: (W+1,) bin edges along X
        y_edges: (H+1,) bin edges along Y
        tau: Soft threshold fraction of max to suppress background (0~0.2 typical)
        eps: Small constant for numerical stability
        mask_radius: If provided, apply circular mask with this radius before computing centroid
        res: Resolution (H, W), needed if mask_radius is used
        negate: If True, negate the centroid coordinates
    
    Returns:
        x_c, y_c: Centroid coordinates
        M0: Total weighted irradiance
    """
    # Convert to torch tensors on common device
    device = irr_map.device if torch.is_tensor(irr_map) else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    irr_map = _to_torch(irr_map, device)
    x_edges = _to_torch(x_edges, device)
    y_edges = _to_torch(y_edges, device)
    
    # Bin centers (size: W and H)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])  # (W,)
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])  # (H,)
    
    # Meshgrids matching irr_map shape (H, W)
    X, Y = torch.meshgrid(x_centers, y_centers, indexing='ij')

    # Soft mask to ignore background but keep gradients
    m = irr_map.max().detach()
    th = tau * m
    ramp = 0.1 * (m + eps)  # Temperature of the sigmoid
    soft_mask = torch.sigmoid((irr_map - th) / ramp)  # (H, W)
    
    Wgt = irr_map * soft_mask  # Weighted irradiance
    M0 = Wgt.sum() + eps       # Total weight
    Mx = (Wgt * X).sum()
    My = (Wgt * Y).sum()
    
    x_c = Mx / M0
    y_c = My / M0
    
    if negate:
        x_c = -x_c
        y_c = -y_c
    
    return x_c, y_c, M0


def spot_centroid_from_irr_mask(irr_map, x_edges, y_edges, mask_radius=1.0, 
                                res=(256, 256), tau=0.05, eps=1e-12):
    """
    Compute the irradiance-weighted centroid with a circular mask applied.
    This is a convenience wrapper around spot_centroid_from_irr.
    
    Args:
        irr_map: (H, W) irradiance map
        x_edges: (W+1,) bin edges along X
        y_edges: (H+1,) bin edges along Y
        radius: Mask radius (inverted - keeps outside)
        res: Resolution (H, W)
        tau: Soft threshold fraction
        eps: Numerical stability constant
    
    Returns:
        x_c, y_c: Centroid coordinates
        M0: Total weighted irradiance
    """
      # Apply mask if requested (inverted mask - keeps outside)
    if mask_radius is not None:
        mask = _create_circular_mask(x_edges, y_edges, radius=mask_radius, res=res, invert=False)
        irr_map = irr_map * mask
        
    return spot_centroid_from_irr(
        irr_map, x_edges, y_edges, 
        tau=tau, eps=eps, 
        res=res, 
        negate=False
    )

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_irr_with_aperture_viz(optical_sys, detector_surface=-1,
                                   res=(256, 256), num_rays=10000,
                                   normalize=True, aperture_radius=None,
                                   aperture_center=None, physical_radius=None,
                                   show_plot=False, linewidth=1, linestyle='-',
                                   color='red', physical_color='blue',
                                   cmap='gray',
                                   Transpose=True):
    """
    Compute the irradiance map and optionally visualize it with aperture and physical size circles.

    Args:
        optical_sys: Optical system object.
        detector_surface: Detector surface index.
        res (tuple): Resolution (H, W).
        num_rays (int): Number of rays to trace.
        normalize (bool): Whether to normalize the irradiance map.
        aperture_radius (float, optional): Radius of the circular aperture (in physical units).
        aperture_center (tuple, optional): (x0, y0) aperture center position.
        physical_radius (float, optional): Radius of physical dimension circle (e.g., sensor boundary).
        show_plot (bool): Whether to display irradiance map.
        linewidth (float): Line width for circle outlines.
        linestyle (str): Line style for the aperture.
        color (str): Aperture boundary color.
        physical_color (str): Physical boundary color.
        cmap (str): Colormap.

    Returns:
        irr_map (torch.Tensor), x_edges, y_edges, aperture_info, fig, ax
    """

    irr_map, x_edges, y_edges = _compute_irradiance(
        optical_sys, detector_surface, res, num_rays
    )

    if not torch.is_tensor(irr_map):
        irr_map = torch.tensor(irr_map, dtype=torch.float32)
    if normalize:
        irr_map = _normalize_irradiance(irr_map)

    # Determine centers
    if aperture_center is None:
        x0 = 0.5 * (x_edges[0] + x_edges[-1])
        y0 = 0.5 * (y_edges[0] + y_edges[-1])
    else:
        x0, y0 = aperture_center

    aperture_info = {"radius": aperture_radius, "center": (x0, y0)} if aperture_radius else None
    physical_info = {"radius": physical_radius, "center": (x0, y0)} if physical_radius else None

    fig, ax = None, None
    if show_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        if Transpose:
            im = ax.imshow(
                irr_map.detach().cpu().numpy().T,
                extent=extent, origin='lower', cmap=cmap, vmin=0, vmax=1
            )
        else:
             im = ax.imshow(
                irr_map.detach().cpu().numpy(),
                extent=extent, origin='lower', cmap=cmap, vmin=0, vmax=1
            )
        # Draw aperture
        if aperture_info is not None:
            circle = patches.Circle(
                aperture_info["center"], aperture_info["radius"],
                fill=False, edgecolor=color, linewidth=linewidth,
                linestyle=linestyle, label=f"Aperture (r={aperture_info['radius']})"
            )
            ax.add_patch(circle)

        # Draw physical size circle
        if physical_info is not None:
            circle_phys = patches.Circle(
                physical_info["center"], physical_info["radius"],
                fill=False, edgecolor=physical_color, linewidth=linewidth,
                linestyle='--', label=f"Physical size (r={physical_info['radius']})"
            )
            ax.add_patch(circle_phys)

        if aperture_info or physical_info:
            ax.legend(loc='upper right')

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Irradiance Map with Aperture & Physical Boundary')
        plt.colorbar(im, ax=ax, label='Normalized Irradiance' if normalize else 'Irradiance')
        plt.tight_layout()

    return irr_map, x_edges, y_edges, aperture_info, fig, ax
