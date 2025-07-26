import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

from diffdrr.drr import DRR
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr
from diffdrr.detector import make_intrinsic_matrix
from diffdrr.pose import convert

# Read in the volume and get its origin and spacing in world coordinates
subject = load_example_ct(bone_attenuation_multiplier=7.5)

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define reorientation matrix (identity for standard orientation)
reorient = torch.eye(3, device=device)

# Initialize DRR with full detector parameters
drr = DRR(
    subject,     # An object storing the CT volume, origin, and voxel spacing
    sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
    height=1536,  # Image height (if width is not provided, the generated DRR is square)
    width=1536,   # Image width
    delx=0.5,    # Pixel spacing in X direction (in mm)
    dely=0.5,    # Pixel spacing in Y direction (in mm)
    x0=0.0,      # Principal point x-coordinate (in mm)
    y0=0.0,      # Principal point y-coordinate (in mm)
    renderer="trilinear",  # For smoother images
    # Render the full detector at once to eliminate block‑edge artefacts
    # and achieve the highest possible image fidelity.
    patch_size=None
).to(device)

# Get the intrinsic matrix from the detector
K_mm = make_intrinsic_matrix(drr.detector)
print("Intrinsic matrix (physical units – millimetres):\n", K_mm)

# Why mm?  `Detector` is parameterised with `delx`/`dely` (mm / pixel) and `x0`,`y0` (mm),
# so the resulting focal length and principal-point coordinates in **K** inherit the same millimetre scale.

# Need pixel units for OpenCV or NumPy?  Divide by the pixel size:
K_px = torch.tensor([[K_mm[0,0] / drr.detector.delx, 0, K_mm[0,2] / drr.detector.delx],
                     [0, K_mm[1,1] / drr.detector.dely, K_mm[1,2] / drr.detector.dely],
                     [0, 0, 1]])
print("Intrinsic matrix (pixel units):\n", K_px)

# ============================================================
# Global rendering parameters
# ------------------------------------------------------------
# Empirically, a higher number of samples per ray **greatly**
# reduces aliasing in the final DRRs without a noticeable
# performance hit on the Apple M4 Max.
# ============================================================
N_SAMPLES = 2000  # points traced along each ray


# Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
# Left camera
left_rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
left_translations = torch.tensor([[50.0, 850.0, 0.0]], device=device)

# Right camera  
right_rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
right_translations = torch.tensor([[-50.0, 850.0, 0.0]], device=device)

# ---------- LEFT CAMERA ----------
left_RT = convert(
    left_rotations, left_translations,
    parameterization="euler_angles",
    convention="ZXY",        # same as you used for drr()
    degrees=False,            # your tensors are in radians
)               # -> RigidTransform instance
left_extrinsic = left_RT.matrix        # 4 x 4  SE(3) homogeneous matrix

# ---------- RIGHT CAMERA ----------
right_RT = convert(
    right_rotations, right_translations,
    parameterization="euler_angles",
    convention="ZXY",
    degrees=False
)
right_extrinsic = right_RT.matrix

print("Left   extrinsic (world → detector):\n", left_extrinsic)
print("Right  extrinsic (world → detector):\n", right_extrinsic)

# If you prefer an OpenCV-style [R | t] that maps CAMERA→WORLD, invert once:
print("Left   camera→world  matrix:\n", left_RT.inverse().matrix)
print("Right  camera→world  matrix:\n", right_RT.inverse().matrix)

# ============================================================
# 1.  Helper: skew‑symmetric matrix (or use diffdrr.pose.hat)
# ------------------------------------------------------------
def skew(v: torch.Tensor) -> torch.Tensor:
    """
    Build the 3 × 3 skew‑symmetric matrix [v]× such that
    [v]× @ x = v × x  (cross‑product).
    `v` is a flat 3‑vector Tensor.
    """
    z = torch.tensor(0., device=v.device, dtype=v.dtype)
    return torch.stack((
        torch.stack((   z, -v[2],  v[1])),
        torch.stack(( v[2],    z, -v[0])),
        torch.stack((-v[1],  v[0],   z))
    ))

# ============================================================
# 2.  Camera centres in WORLD coordinates
# ------------------------------------------------------------
C_left  = left_RT.inverse().translation[0]    # shape (3,)
C_right = right_RT.inverse().translation[0]

# Vector from left → right, still in WORLD frame
t_world = C_right - C_left                    # (3,)

# ============================================================
# 3.  Relative rotation and translation expressed **in left‑camera coordinates**
# ------------------------------------------------------------
R_left  = left_RT.rotation[0]                 # world → left
R_right = right_RT.rotation[0]                # world → right

# Rotation of the right camera **with respect to** the left camera
R_rel = R_right @ R_left.mT                   # (3,3)

# Translation from left to right, but expressed in the left camera frame
t_left = R_left @ t_world                     # (3,)

# ============================================================
# 4.  Essential matrix   E = [t]×  R
# ------------------------------------------------------------
E = skew(t_left) @ R_rel                      # (3,3)

# Optionally normalise (scale is irrelevant for epipolar geometry)
E = E / E.norm()

# ============================================================
# 5.  Fundamental matrix  F = K⁻ᵀ E K⁻¹
# ------------------------------------------------------------
K     = K_px                                  # we already computed this above
K_inv = torch.inverse(K)

F = K_inv.mT @ E @ K_inv
F = F / F.norm()                              # normalise for convenience

# ============================================================
# 6.  Display
# ------------------------------------------------------------
print("\nEssential matrix  (E):\n", E)
print("\nFundamental matrix (F):\n", F)

# Optional: Define calibration transforms (identity for no calibration offset)
calibration = torch.eye(4, device=device).unsqueeze(0)

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
left_img = drr(
    left_rotations,
    left_translations,
    parameterization="euler_angles",
    convention="ZXY",
    n_points=N_SAMPLES,
)
right_img = drr(
    right_rotations,
    right_translations,
    parameterization="euler_angles",
    convention="ZXY",
    n_points=N_SAMPLES,
)

# Debug tensor information
print(f"\nTensor shape: {left_img.shape}")
print(f"Tensor dtype: {left_img.dtype}")
print(f"Tensor min: {left_img.min():.4f}, max: {left_img.max():.4f}")

# Save individual images at full resolution using cv2
# Convert tensors to numpy arrays
left_img_np = left_img[0, 0].cpu().numpy()  # shape (1536, 1536)
right_img_np = right_img[0, 0].cpu().numpy()

# Normalize to 0-255 range
left_img_uint8 = ((left_img_np - left_img_np.min()) / (left_img_np.max() - left_img_np.min()) * 255).astype(np.uint8)
right_img_uint8 = ((right_img_np - right_img_np.min()) / (right_img_np.max() - right_img_np.min()) * 255).astype(np.uint8)

# Save full resolution images
cv2.imwrite('data/left_view.png', left_img_uint8)
cv2.imwrite('data/right_view.png', right_img_uint8)

# ------------------------------------------------------------
# Stereo pair (side‑by‑side)
# ------------------------------------------------------------
fig_stereo, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))
plot_drr(left_img,  ticks=False, axs=ax_l)
plot_drr(right_img, ticks=False, axs=ax_r)
ax_l.axis('off')
ax_r.axis('off')
fig_stereo.subplots_adjust(wspace=0)
plt.savefig('stereo_views.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print("Images saved successfully:")
print("- left_view.png")
print("- right_view.png")
print("- stereo_views.png (both views together)")
