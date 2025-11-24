import numpy as np

def rot_x(a):
    """Rotation matrix around the X-axis by angle a (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=float)

def rot_y(a):
    """Rotation matrix around the Y-axis by angle a (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=float)

def rot_z(a):
    """Rotation matrix around the Z-axis by angle a (radians)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)

def mirror_normal(rx, ry, rz, base_n=np.array([0, 0, 1.0])):
    """
    Compute the normal vector of a mirror given its rotations.

    Parameters
    ----------
    rx, ry, rz : float
        Rotation angles (radians) around x, y, z axes.
    base_n : np.ndarray, default [0, 0, 1]
        Base normal vector before rotation.

    Returns
    -------
    n : np.ndarray
        Normalized mirror normal vector.
    """
    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    n = R @ base_n
    return n / np.linalg.norm(n)

def reflect_with_position_aperture(mirror_pos, rx, ry, rz,
                                    aperture=20,
                                    r0=np.array([0, 0, 0], dtype=float),
                                    d_in=np.array([0, 0, 1], dtype=float)):
    """
    Compute reflection of a ray on a mirror with finite aperture.

    If the ray misses the mirror (outside aperture), returns False.
    """
    d_in = d_in / np.linalg.norm(d_in)
    n = mirror_normal(rx, ry, rz)

    # Intersection of ray and mirror plane
    denom = np.dot(d_in, n)
    if abs(denom) < 1e-9:
        raise ValueError("Ray is parallel to the mirror plane")
    t = np.dot(mirror_pos - r0, n) / denom
    p_hit = r0 + t * d_in

    # --- Check aperture (optional) ---
    if aperture is not None:
        # Define local mirror axes u,v
        u = np.cross(n, np.array([0, 0, 1]))
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, np.array([0, 1, 0]))
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        local_hit = p_hit - mirror_pos
        x = np.dot(local_hit, u)
        y = np.dot(local_hit, v)
        r_hit = np.sqrt(x**2 + y**2)
        if r_hit > aperture:
            # Missed the mirror aperture
            return False, None

    # Reflected direction
    d_out = d_in - 2 * np.dot(d_in, n) * n
    d_out = d_out / np.linalg.norm(d_out)


    return p_hit, d_out

def reflect_with_position(mirror_pos, rx, ry, rz, dist,
                          r0=np.array([0, 0, 0], dtype=float),
                          d_in=np.array([0, 0, 1], dtype=float)):
    """
    Compute reflection of a ray on a mirror and propagate to a new position.

    Parameters
    ----------
    mirror_pos : np.ndarray
        Mirror position in 3D.
    rx, ry, rz : float
        Mirror rotation angles (radians).
    dist : float
        Propagation distance along the reflected direction.
    r0 : np.ndarray, optional
        Ray origin (default: [0,0,0]).
    d_in : np.ndarray, optional
        Incident ray direction (default: [0,0,1]).

    Returns
    -------
    p_hit : np.ndarray
        Intersection point with the mirror.
    d_out : np.ndarray
        Reflected direction (normalized).·111111
    p_out : np.ndarray
        Position after propagating distance dist along d_out.
    """
    d_in = d_in / np.linalg.norm(d_in)
    n = mirror_normal(rx, ry, rz)

    # Intersection of ray and mirror plane
    denom = np.dot(d_in, n)
    if abs(denom) < 1e-9:
        raise ValueError("Ray is parallel to the mirror")
    t = np.dot(mirror_pos - r0, n) / denom
    p_hit = r0 + t * d_in
    # print(f"----------------------{t}")

    # Reflected direction
    d_out = d_in - 2 * np.dot(d_in, n) * n
    d_out = d_out / np.linalg.norm(d_out)

    # Propagate along reflected direction
    p_out = p_hit + dist * d_out

    return p_hit, d_out, p_out

def plane_angles_from_normal(n):
    """
    Given a plane normal vector n (3D), return the rotations (rx, ry, rz)
    that rotate the default normal [0,0,1] to n.

    Parameters
    ----------
    n : np.ndarray
        Plane normal vector.

    Returns
    -------
    rx, ry, rz : float
        Rotation angles (radians).
    """
    n = n / np.linalg.norm(n)
    rx = -np.arcsin(n[1])         # rotation around X
    ry = np.arctan2(n[0], n[2])   # rotation around Y
    rz = 0.0                      # rotation around Z is ignored
    return rx, ry, rz

def calculate_position(p, d, dist):
    """
    Compute a new position along direction d starting from point p.

    Parameters
    ----------
    p : np.ndarray
        Starting point.
    d : np.ndarray
        Direction (assumed normalized).
    dist : float
        Distance to move along d.

    Returns
    -------
    np.ndarray
        New position.
    """
    return p + d * dist

# Example: two-mirror system
if __name__ == "__main__":
    # Incident ray: starting from origin along +z
    r0 = np.array([0, 0, 0], dtype=float)
    d_in = np.array([0, 0, 1], dtype=float)

    # Mirror 1: position (0,0,10), rotated 45° around x-axis
    mirror1_pos = np.array([0, 0, 10], dtype=float)
    rx1, ry1, rz1 = np.deg2rad(45), np.deg2rad(0), 0
    dist1 = 10.0

    p1_hit, d1_out, p1_out = reflect_with_position(mirror1_pos, rx1, ry1, rz1, dist1, r0=r0, d_in=d_in)
    print("-------After Mirror 1---------")
    print("Mirror 1 hit point =", p1_hit)
    print("Mirror 1 reflection direction =", d1_out)
    print("Point after Mirror 1 =", p1_out)

    # Mirror 2: placed along the reflected ray path
    # mirror2_pos = np.array([p1_out[0], p1_out[1], p1_out[2]], dtype=float)
    mirror2_pos = np.array([0, 50, 0], dtype=float)
    rx2, ry2, rz2 = np.deg2rad(45), np.deg2rad(0), 0
    dist2 = 10.0

    p2_hit, d2_out, p2_out = reflect_with_position_aperture(mirror2_pos, rx2, ry2, rz2, dist2, r0=p1_hit, d_in=d1_out)
    print("-------After Mirror 2---------")
    print("Mirror 2 hit point =", p2_hit)
    print("Mirror 2 reflection direction =", d2_out)
    print("Point after Mirror 2 =", p2_out)

    # Receiver plane normal = -d2_out
    # n_plane = -d2_out
    # rx_plane, ry_plane, rz_plane = plane_angles_from_normal(n_plane)

    # print("Receiver plane normal =", n_plane)
    # print("Receiver plane rotation angles (deg): rx=%.3f, ry=%.3f, rz=%.3f" %
    #       (np.degrees(rx_plane), np.degrees(ry_plane), np.degrees(rz_plane)))
