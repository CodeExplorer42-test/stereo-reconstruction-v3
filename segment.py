# import the necessary libraries
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import OrderedDict, deque
from skimage.morphology import skeletonize

# Initialize benchmark tracking
benchmark_times = OrderedDict()
start_total = time.time()

# 1. load both rectified images
start_step = time.time()
left_image = cv2.imread('data/left_rectified.png', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('data/right_rectified.png', cv2.IMREAD_GRAYSCALE)
benchmark_times['1. Image Loading'] = time.time() - start_step

# 1.1 Check the dimension of both images
print(f"Left image dimensions: {left_image.shape}")
print(f"Right image dimensions: {right_image.shape}")

# 2. Normalize and threshold both images to extract relevant X-ray regions.
start_step = time.time()
threshold = 0.10
left_normalized = left_image.astype(np.float32) / 255.0
right_normalized = right_image.astype(np.float32) / 255.0

# keep intensities above threshold, set others to 0
left_thresholded = np.where(left_normalized > threshold, left_normalized, 0)
right_thresholded = np.where(right_normalized > threshold, right_normalized, 0)

# convert back to uint8 for display and edge detection
left_thresholded_uint8 = (left_thresholded * 255).astype(np.uint8)
right_thresholded_uint8 = (right_thresholded * 255).astype(np.uint8)
benchmark_times['2. Normalization & Thresholding'] = time.time() - start_step

# 3. Gaussian blur to smooth images.
start_step = time.time()
left_smoothed = cv2.GaussianBlur(left_thresholded_uint8, (15, 15), 0)
right_smoothed = cv2.GaussianBlur(right_thresholded_uint8, (15, 15), 0)
# Cast to float32 to avoid overflow in region growing
left_smoothed_float = left_smoothed.astype(np.float32)
right_smoothed_float = right_smoothed.astype(np.float32)
benchmark_times['3. Gaussian Blur'] = time.time() - start_step

# 4. Find dark valleys between bright structures.
start_step = time.time()
left_valleys = (left_smoothed < 30).astype(np.uint8) * 255
right_valleys = (right_smoothed < 30).astype(np.uint8) * 255
benchmark_times['4. Valley Detection'] = time.time() - start_step

# 5. Dilate valleys to improve region separation.
start_step = time.time()
valley_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
left_valleys = cv2.dilate(left_valleys, valley_kernel)
right_valleys = cv2.dilate(right_valleys, valley_kernel)
benchmark_times['5. Valley Dilation'] = time.time() - start_step

# 6. Calculate intensity gradients (Sobel filter) on thresholded images.
start_step = time.time()
left_grad_x = cv2.Sobel(left_thresholded_uint8, cv2.CV_64F, 1, 0, ksize=3)
left_grad_y = cv2.Sobel(left_thresholded_uint8, cv2.CV_64F, 0, 1, ksize=3)
right_grad_x = cv2.Sobel(right_thresholded_uint8, cv2.CV_64F, 1, 0, ksize=3)
right_grad_y = cv2.Sobel(right_thresholded_uint8, cv2.CV_64F, 0, 1, ksize=3)
benchmark_times['6. Sobel Gradient Calculation'] = time.time() - start_step

# 7. Compute gradient magnitude images.
start_step = time.time()
left_grad_magnitude = np.sqrt(left_grad_x**2 + left_grad_y**2)
right_grad_magnitude = np.sqrt(right_grad_x**2 + right_grad_y**2)
benchmark_times['7. Gradient Magnitude'] = time.time() - start_step

# 8. Normalize gradients for display.
start_step = time.time()
left_gradient_normalized = (left_grad_magnitude / left_grad_magnitude.max() * 255).astype(np.uint8)
right_gradient_normalized = (right_grad_magnitude / right_grad_magnitude.max() * 255).astype(np.uint8)
benchmark_times['8. Gradient Normalization'] = time.time() - start_step

# 9. Threshold gradient to get boundary pixels.
start_step = time.time()
gradient_threshold = 5
left_boundaries = (left_grad_magnitude > gradient_threshold).astype(np.uint8) * 255
right_boundaries = (right_grad_magnitude > gradient_threshold).astype(np.uint8) * 255

print(f"Gradient boundary pixels - left: {np.count_nonzero(left_boundaries)}, right: {np.count_nonzero(right_boundaries)}")
benchmark_times['9. Boundary Thresholding'] = time.time() - start_step

# 10. Get coordinates of gradient pixels.
start_step = time.time()
# 10.1 Get all gradient pixels first
left_all_gradient_coords = np.column_stack(np.where(left_grad_magnitude > gradient_threshold))
right_all_gradient_coords = np.column_stack(np.where(right_grad_magnitude > gradient_threshold))

# 10.2 Filter to keep only gradients within X-ray regions (intensity > threshold)
left_gradient_coords = []
for coord in left_all_gradient_coords:
    y, x = coord
    if left_normalized[y, x] > threshold:
        left_gradient_coords.append(coord)
left_gradient_coords = np.array(left_gradient_coords)

right_gradient_coords = []
for coord in right_all_gradient_coords:
    y, x = coord
    if right_normalized[y, x] > threshold:
        right_gradient_coords.append(coord)
right_gradient_coords = np.array(right_gradient_coords)
benchmark_times['10. Extract Gradient Coordinates'] = time.time() - start_step

print(f"Gradient pixel coordinates - left: {left_gradient_coords.shape}, right: {right_gradient_coords.shape}")
print(f"Sample left coords (first 5): {left_gradient_coords[:5].tolist()}")
print(f"Sample right coords (first 5): {right_gradient_coords[:5].tolist()}")

# create boundary image directly from gradient coordinates
start_step = time.time()
left_boundary_image = np.zeros_like(left_image)
right_boundary_image = np.zeros_like(right_image)

# set gradient pixels to white
left_boundary_image[left_gradient_coords[:, 0], left_gradient_coords[:, 1]] = 255
right_boundary_image[right_gradient_coords[:, 0], right_gradient_coords[:, 1]] = 255
benchmark_times['11. Create Boundary Images'] = time.time() - start_step

# 12. Intensity-constrained Region Growing
start_step = time.time()

# 12.1 Initialize seed masks and label maps
left_seed_mask = np.zeros_like(left_image, dtype=np.uint8)
right_seed_mask = np.zeros_like(right_image, dtype=np.uint8)
left_label_map = np.zeros_like(left_image, dtype=np.int32)
right_label_map = np.zeros_like(right_image, dtype=np.int32)

# 12.2 Extract seeds from boundary fragments
# For left image:
left_seed_queue = []
for coord in left_gradient_coords:
    y, x = coord
    # Compute normalized gradient vector at this boundary pixel
    grad_x = left_grad_x[y, x]
    grad_y = left_grad_y[y, x]
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    if grad_mag > 0:
        # Normalize gradient
        grad_x_norm = grad_x / grad_mag
        grad_y_norm = grad_y / grad_mag
        
        # Walk 10 pixels opposite to gradient direction
        seed_x = int(x - 10 * grad_x_norm)
        seed_y = int(y - 10 * grad_y_norm)
        
        # Check if seed is valid
        if (0 <= seed_x < left_image.shape[1] and 
            0 <= seed_y < left_image.shape[0] and
            left_boundary_image[seed_y, seed_x] == 0 and
            left_seed_mask[seed_y, seed_x] == 0):
            # Mark seed and enqueue
            left_seed_mask[seed_y, seed_x] = 1
            left_seed_queue.append((seed_y, seed_x))

# For right image:
right_seed_queue = []
for coord in right_gradient_coords:
    y, x = coord
    # Compute normalized gradient vector at this boundary pixel
    grad_x = right_grad_x[y, x]
    grad_y = right_grad_y[y, x]
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    if grad_mag > 0:
        # Normalize gradient
        grad_x_norm = grad_x / grad_mag
        grad_y_norm = grad_y / grad_mag
        
        # Walk 10 pixels opposite to gradient direction
        seed_x = int(x - 10 * grad_x_norm)
        seed_y = int(y - 10 * grad_y_norm)
        
        # Check if seed is valid
        if (0 <= seed_x < right_image.shape[1] and 
            0 <= seed_y < right_image.shape[0] and
            right_boundary_image[seed_y, seed_x] == 0 and
            right_seed_mask[seed_y, seed_x] == 0):
            # Mark seed and enqueue
            right_seed_mask[seed_y, seed_x] = 1
            right_seed_queue.append((seed_y, seed_x))

print(f"Generated seeds - left: {len(left_seed_queue)}, right: {len(right_seed_queue)}")

# 12.3 Region growing with intensity constraints
delta_I = 5  # intensity tolerance
min_blob_area = 10  # minimum blob size

# Process left image
left_current_label = 1
left_blob_areas = {}  # track area per label

for seed_y, seed_x in left_seed_queue:
    if left_label_map[seed_y, seed_x] > 0:
        continue  # already processed
    
    # Initialize flood fill for this seed
    seed_intensity = left_smoothed_float[seed_y, seed_x]
    growth_queue = deque([(seed_y, seed_x)])
    left_label_map[seed_y, seed_x] = left_current_label
    blob_area = 1
    
    # 8-connected flood fill
    while growth_queue:
        cy, cx = growth_queue.popleft()
        
        # Check all 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                    
                ny, nx = cy + dy, cx + dx
                
                # Boundary checks
                if (0 <= ny < left_image.shape[0] and 
                    0 <= nx < left_image.shape[1] and
                    left_boundary_image[ny, nx] == 0 and  # not a wall
                    left_label_map[ny, nx] == 0):  # not yet labeled
                    
                    # Intensity constraint check
                    neighbor_intensity = left_smoothed_float[ny, nx]
                    if abs(seed_intensity - neighbor_intensity) <= delta_I:
                        left_label_map[ny, nx] = left_current_label
                        growth_queue.append((ny, nx))
                        blob_area += 1
    
    # Store blob area
    left_blob_areas[left_current_label] = blob_area
    left_current_label += 1

# Process right image
right_current_label = 1
right_blob_areas = {}  # track area per label

for seed_y, seed_x in right_seed_queue:
    if right_label_map[seed_y, seed_x] > 0:
        continue  # already processed
    
    # Initialize flood fill for this seed
    seed_intensity = right_smoothed_float[seed_y, seed_x]
    growth_queue = deque([(seed_y, seed_x)])
    right_label_map[seed_y, seed_x] = right_current_label
    blob_area = 1
    
    # 8-connected flood fill
    while growth_queue:
        cy, cx = growth_queue.popleft()
        
        # Check all 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                    
                ny, nx = cy + dy, cx + dx
                
                # Boundary checks
                if (0 <= ny < right_image.shape[0] and 
                    0 <= nx < right_image.shape[1] and
                    right_boundary_image[ny, nx] == 0 and  # not a wall
                    right_label_map[ny, nx] == 0):  # not yet labeled
                    
                    # Intensity constraint check
                    neighbor_intensity = right_smoothed_float[ny, nx]
                    if abs(seed_intensity - neighbor_intensity) <= delta_I:
                        right_label_map[ny, nx] = right_current_label
                        growth_queue.append((ny, nx))
                        blob_area += 1
    
    # Store blob area
    right_blob_areas[right_current_label] = blob_area
    right_current_label += 1

# 12.4 Filter small blobs
for label in range(1, left_current_label):
    if left_blob_areas[label] < min_blob_area:
        left_label_map[left_label_map == label] = 0

for label in range(1, right_current_label):
    if right_blob_areas[label] < min_blob_area:
        right_label_map[right_label_map == label] = 0

print(f"Blobs after filtering - left: {np.unique(left_label_map[left_label_map > 0]).size}, " \
      f"right: {np.unique(right_label_map[right_label_map > 0]).size}")

# 12.5 Create color overlay visualization
left_overlay = np.zeros((*left_image.shape, 3), dtype=np.float32)
right_overlay = np.zeros((*right_image.shape, 3), dtype=np.float32)

# Apply colormap to non-zero labels with zero-label guard
if left_label_map.max() > 0:
    left_labels_normalized = left_label_map.astype(np.float32) / left_label_map.max()
    left_colored = plt.cm.nipy_spectral(left_labels_normalized)[:, :, :3]
    left_mask = (left_label_map > 0).astype(np.float32)[:, :, np.newaxis]
    left_overlay = left_colored * left_mask

if right_label_map.max() > 0:
    right_labels_normalized = right_label_map.astype(np.float32) / right_label_map.max()
    right_colored = plt.cm.nipy_spectral(right_labels_normalized)[:, :, :3]
    right_mask = (right_label_map > 0).astype(np.float32)[:, :, np.newaxis]
    right_overlay = right_colored * right_mask

# Composite with original image
left_base = np.stack([left_thresholded_uint8]*3, axis=-1) / 255.0
left_composite = left_base * 0.6 + left_overlay * 0.4

right_base = np.stack([right_thresholded_uint8]*3, axis=-1) / 255.0
right_composite = right_base * 0.6 + right_overlay * 0.4

benchmark_times['12. Intensity-constrained Region Growing'] = time.time() - start_step

# 13. Extended visualization with segmented blobs
visualization_start = time.time()
print("\nStarting watershed segmentation visualization...")
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Row 1: Left image analysis
axes[0, 0].imshow(left_image, cmap='gray')
axes[0, 0].set_title('Left Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(left_gradient_normalized, cmap='gray')
axes[0, 1].set_title('Left Gradient')
axes[0, 1].axis('off')

axes[0, 2].imshow(left_boundary_image, cmap='gray')
axes[0, 2].set_title('Left Boundaries')
axes[0, 2].axis('off')

axes[0, 3].imshow(left_composite)
axes[0, 3].set_title('Left Segmented Blobs')
axes[0, 3].axis('off')

# Row 2: Right image analysis
axes[1, 0].imshow(right_image, cmap='gray')
axes[1, 0].set_title('Right Original')
axes[1, 0].axis('off')

axes[1, 1].imshow(right_gradient_normalized, cmap='gray')
axes[1, 1].set_title('Right Gradient')
axes[1, 1].axis('off')

axes[1, 2].imshow(right_boundary_image, cmap='gray')
axes[1, 2].set_title('Right Boundaries')
axes[1, 2].axis('off')

axes[1, 3].imshow(right_composite)
axes[1, 3].set_title('Right Segmented Blobs')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('watershed_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()

# Print benchmark summary
benchmark_times['13. Visualization'] = time.time() - visualization_start

# ============================================================
# 14. Stereo Matching
# ============================================================

# 14.1 Define absolute disparity search window
min_disparity = 0       # never allow negative shift
max_disparity = 256     # generous upper limit in pixels

# 14.2 Thin boundaries to reduce computational load
start_step = time.time()
print("\n=== Stereo Matching ===")
print(f"Disparity search range: {min_disparity} to {max_disparity} pixels")

# Apply morphological skeletonization to get true 1-pixel skeleton
left_skeleton = (skeletonize(left_boundary_image > 0) * 255).astype(np.uint8)
right_skeleton = (skeletonize(right_boundary_image > 0) * 255).astype(np.uint8)

# Get thinned boundary coordinates
left_thin_coords = np.column_stack(np.where(left_skeleton > 0))
right_thin_coords = np.column_stack(np.where(right_skeleton > 0))

print(f"Boundary thinning: {len(left_gradient_coords)} -> {len(left_thin_coords)} pixels")
benchmark_times['14.2 Boundary Thinning'] = time.time() - start_step

# Initialize unified disparity output
disparity_output = np.full(left_image.shape, np.nan, dtype=np.float32)

# Select matching algorithm
selected_matching = 'chamfer'  # or 'block', 'sgm', 'orb'
print(f"\nSelected matching algorithm: {selected_matching}")

# 14.3 Edge-based Chamfer along the line
if selected_matching == 'chamfer':
    start_step = time.time()
    print("\n14.3 Running Chamfer distance matching...")
    
    # Compute distance transform for right skeleton once
    right_distance_transform = cv2.distanceTransform(
        255 - right_skeleton, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    
    # Process thinned boundary pixels
    chamfer_window_size = 31
    half_window = chamfer_window_size // 2
    matched_count = 0
    
    for coord in left_thin_coords:
        y, x = coord
        
        # Skip pixels too close to image edges
        if (y < half_window or y >= left_skeleton.shape[0] - half_window or
            x < max_disparity or x >= left_skeleton.shape[1] - half_window):
            continue
        
        # Extract window around left boundary pixel
        y_min = y - half_window
        y_max = y + half_window + 1
        x_min = x - half_window
        x_max = x + half_window + 1
        
        left_window = left_skeleton[y_min:y_max, x_min:x_max]
        
        if np.sum(left_window) == 0:  # skip empty windows
            continue
        
        # Vectorized disparity search
        # Create disparity range
        disparities = np.arange(min_disparity, min(max_disparity + 1, x + 1))
        right_xs = x - disparities
        
        # Extract all candidate windows at once using advanced indexing
        valid_mask = (right_xs >= half_window) & (right_xs < right_distance_transform.shape[1] - half_window)
        valid_disparities = disparities[valid_mask]
        valid_right_xs = right_xs[valid_mask]
        
        if len(valid_disparities) == 0:
            continue
        
        # Compute scores for all valid positions
        scores = np.zeros(len(valid_disparities))
        left_mask = left_window > 0
        
        for i, rx in enumerate(valid_right_xs):
            right_dt_window = right_distance_transform[y_min:y_max, rx - half_window:rx + half_window + 1]
            scores[i] = np.sum(right_dt_window[left_mask])
        
        # Find best match
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        best_disparity = valid_disparities[best_idx]
        
        # Store disparity if match was good enough
        if best_score < np.sum(left_window > 0) * 5:  # adaptive threshold
            disparity_output[y, x] = best_disparity
            matched_count += 1
    
    print(f"Chamfer matching: {matched_count}/{len(left_thin_coords)} pixels matched")
    benchmark_times['14.3 Chamfer Matching'] = time.time() - start_step

# 14.4 Block matching (dense SSD)
elif selected_matching == 'block':
    start_step = time.time()
    print("\n14.4 Running block matching (SSD)...")
    
    # Use rectified grayscale images for texture
    block_size = 11
    half_block = block_size // 2
    matched_count = 0
    
    # Convert images to float32 once
    left_float = left_image.astype(np.float32)
    right_float = right_image.astype(np.float32)
    
    for coord in left_thin_coords:
        y, x = coord
        
        # Skip pixels too close to image edges
        if (y < half_block or y >= left_image.shape[0] - half_block or
            x < max_disparity or x >= left_image.shape[1] - half_block):
            continue
        
        # Extract template block around left pixel
        template = left_float[y - half_block:y + half_block + 1, 
                             x - half_block:x + half_block + 1]
        
        # Define search region in right image
        search_x_min = max(0, x - max_disparity - half_block)
        search_x_max = min(right_image.shape[1] - block_size + 1, x - min_disparity + half_block + 1)
        
        if search_x_max <= search_x_min:
            continue
            
        # Extract search region
        search_region = right_float[y - half_block:y + half_block + 1,
                                   search_x_min:search_x_max + block_size - 1]
        
        # Use matchTemplate for fast SSD computation
        result = cv2.matchTemplate(search_region, template, cv2.TM_SQDIFF)
        
        # Find minimum (best match)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        
        # Convert back to disparity
        matched_x = search_x_min + min_loc[0] + half_block
        disparity = x - matched_x
        
        # Store disparity if match was good enough and within valid range
        if (min_val < block_size * block_size * 500 and 
            min_disparity <= disparity <= max_disparity):
            disparity_output[y, x] = disparity
            matched_count += 1
    
    print(f"Block matching: {matched_count}/{len(left_thin_coords)} pixels matched")
    benchmark_times['14.4 Block Matching'] = time.time() - start_step

# 14.5 Semi-Global Matching (SGM)
elif selected_matching == 'sgm':
    start_step = time.time()
    print("\n14.5 Running Semi-Global Matching (SGM)...")
    
    # Create SGBM matcher
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=max_disparity - min_disparity,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # Run SGM on full rectified images
    disparity_sgm = sgbm.compute(left_image, right_image).astype(np.float32) / 16.0
    
    # Mask to keep only boundary pixels
    # Note: Invalid disparities in OpenCV SGBM are marked as (minDisparity - 1)
    invalid_disparity = sgbm.getMinDisparity() - 1
    matched_count = 0
    
    for coord in left_thin_coords:
        y, x = coord
        disp_value = disparity_sgm[y, x]
        if disp_value != invalid_disparity and disp_value >= min_disparity:  # valid disparity
            disparity_output[y, x] = disp_value
            matched_count += 1
    
    print(f"SGM matching: {matched_count}/{len(left_thin_coords)} boundary pixels have valid disparities")
    benchmark_times['14.5 SGM'] = time.time() - start_step

# 14.6 Sparse feature (SIFT/ORB) fallback
elif selected_matching == 'orb':
    start_step = time.time()
    print("\n14.6 Running ORB feature matching...")
    
    # Create ORB detector (always available in OpenCV)
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    
    # Detect keypoints on rectified images for better texture
    kp_left, desc_left = orb.detectAndCompute(left_image, None)
    kp_right, desc_right = orb.detectAndCompute(right_image, None)
    
    print(f"Detected keypoints - left: {len(kp_left)}, right: {len(kp_right)}")
    
    # Match descriptors
    if desc_left is not None and desc_right is not None:
        # Use Hamming distance for ORB binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc_left, desc_right, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        print(f"Good matches after ratio test: {len(good_matches)}")
        
        # Process matches that land on boundaries
        matched_count = 0
        for match in good_matches:
            # Get coordinates
            left_pt = kp_left[match.queryIdx].pt
            right_pt = kp_right[match.trainIdx].pt
            
            x_left, y_left = int(left_pt[0]), int(left_pt[1])
            x_right, y_right = int(right_pt[0]), int(right_pt[1])
            
            # Check if both points are on boundaries and on same epipolar line
            if (0 <= y_left < left_boundary_image.shape[0] and
                0 <= x_left < left_boundary_image.shape[1] and
                0 <= y_right < right_boundary_image.shape[0] and
                0 <= x_right < right_boundary_image.shape[1] and
                left_boundary_image[y_left, x_left] > 0 and
                right_boundary_image[y_right, x_right] > 0 and
                abs(y_left - y_right) < 2):  # rectified constraint
                
                disparity = x_left - x_right
                if min_disparity <= disparity <= max_disparity:
                    disparity_output[y_left, x_left] = disparity
                    matched_count += 1
    
        print(f"ORB matching: {matched_count} boundary pixels matched")
    else:
        print("ORB matching: No descriptors found")
    benchmark_times['14.6 ORB Matching'] = time.time() - start_step

# 14.7 Save disparity visualization
start_step = time.time()
valid_disparities = disparity_output[~np.isnan(disparity_output)]
if len(valid_disparities) > 0:
    print(f"\nDisparity statistics:")
    print(f"  Valid pixels: {len(valid_disparities)}")
    print(f"  Min disparity: {valid_disparities.min():.1f}")
    print(f"  Max disparity: {valid_disparities.max():.1f}")
    print(f"  Mean disparity: {valid_disparities.mean():.1f}")
    
    # Create colormap visualization
    disparity_vis = disparity_output.copy()
    disparity_vis[np.isnan(disparity_vis)] = 0
    disparity_vis_normalized = (disparity_vis / max_disparity * 255).astype(np.uint8)
    disparity_colored = cv2.applyColorMap(disparity_vis_normalized, cv2.COLORMAP_JET)
    
    # Mask out invalid pixels
    mask = ~np.isnan(disparity_output)
    disparity_colored[~mask] = 0
    
    cv2.imwrite(f'disparity_{selected_matching}.png', disparity_colored)
    print(f"Saved disparity map: disparity_{selected_matching}.png")
    
    # Quick numeric sanity check
    print("\nSample disparities (10 random boundary pixels):")
    valid_indices = np.argwhere(~np.isnan(disparity_output))
    if len(valid_indices) >= 10:
        sample_indices = valid_indices[np.random.choice(len(valid_indices), 10, replace=False)]
        for idx in sample_indices:
            y, x = idx
            disp = disparity_output[y, x]
            print(f"  Pixel ({y}, {x}): disparity = {disp:.1f} px")
else:
    print("\nNo valid disparities found!")

benchmark_times['14.7 Disparity Visualization'] = time.time() - start_step

# Final benchmark update
print("\n=== Final Benchmark Summary ===")
for step, duration in benchmark_times.items():
    print(f"{step}: {duration:.3f}s")
print(f"Total time: {time.time() - start_total:.3f}s")
