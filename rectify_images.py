import cv2
import numpy as np

# 1. Hardcode camera parameters from camera_params.md
# Note: Principal point must be at image center (768, 768) for 1536x1536 image
# DiffDRR's (1536, 1536) measured from physical corner needs adjustment for OpenCV
K_px = np.array([[4080.0, 0.0, 768.0],
                 [0.0, 4080.0, 768.0],
                 [0.0, 0.0, 1.0]], dtype=np.float32)

left_extrinsic = np.array([[1., 0., 0., 50.],
                           [0., 1., 0., 850.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]], dtype=np.float32)

right_extrinsic = np.array([[1., 0., 0., -50.],
                            [0., 1., 0., 850.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]], dtype=np.float32)

# 2. Extract rotation and translation components
left_R = left_extrinsic[:3, :3]
left_t = left_extrinsic[:3, 3]
right_R = right_extrinsic[:3, :3]
right_t = right_extrinsic[:3, 3]

# 3. Compute relative pose (right w.r.t left)
R_rel = right_R @ left_R.T  # Identity matrix since both R are identity
t_rel = right_t - left_t    # [-100, 0, 0] in mm

print("Relative rotation R_rel:")
print(R_rel)
print("\nRelative translation t_rel (mm):")
print(t_rel)

# 4. Call stereoRectify with full 1536x1536 image size
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1=K_px,
    distCoeffs1=np.zeros(5),
    cameraMatrix2=K_px,
    distCoeffs2=np.zeros(5),
    imageSize=(1536, 1536),  # Full DRR size
    R=R_rel,
    T=t_rel,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

print("\nRectification rotation R1:")
print(R1)
print("\nRectification rotation R2:")
print(R2)

# 5. Create remapping lookup tables for full resolution
map1x, map1y = cv2.initUndistortRectifyMap(
    K_px, np.zeros(5), R1, P1, (1536, 1536), cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    K_px, np.zeros(5), R2, P2, (1536, 1536), cv2.CV_32FC1
)

# 6. Load full-size images and rectify
left_img = cv2.imread('data/left_view.png', cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread('data/right_view.png', cv2.IMREAD_GRAYSCALE)

print(f"\nOriginal image shapes - left: {left_img.shape}, right: {right_img.shape}")

left_rectified = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, map2x, map2y, cv2.INTER_LINEAR)

# Save rectified images
cv2.imwrite('data/left_rectified.png', left_rectified)
cv2.imwrite('data/right_rectified.png', right_rectified)

print("\nRectified images saved:")
print("- data/left_rectified.png")
print("- data/right_rectified.png")

# Optional: Verify epipolar alignment by drawing horizontal lines
# Create a visualization showing epipolar lines are now horizontal
viz_left = cv2.cvtColor(left_rectified, cv2.COLOR_GRAY2BGR)
viz_right = cv2.cvtColor(right_rectified, cv2.COLOR_GRAY2BGR)

# Draw horizontal lines at same y-coordinates on both images
for y in range(200, 1400, 200):
    cv2.line(viz_left, (0, y), (1536, y), (0, 255, 0), 1)
    cv2.line(viz_right, (0, y), (1536, y), (0, 255, 0), 1)

# Save visualization
viz_combined = np.hstack([viz_left, viz_right])
cv2.imwrite('data/rectification_check.png', viz_combined)
print("- data/rectification_check.png (visualization with epipolar lines)")