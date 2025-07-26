# import the necessary libraries
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from collections import OrderedDict, deque

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
    seed_intensity = left_smoothed[seed_y, seed_x]
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
                    neighbor_intensity = left_smoothed[ny, nx]
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
    seed_intensity = right_smoothed[seed_y, seed_x]
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
                    neighbor_intensity = right_smoothed[ny, nx]
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
print("\n=== Benchmark Summary ===")
for step, duration in benchmark_times.items():
    print(f"{step}: {duration:.3f}s")
print(f"Total time: {time.time() - start_total:.3f}s")
