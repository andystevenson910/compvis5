import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("Wirebond.tif", cv2.IMREAD_GRAYSCALE)

# List of kernels for erosion
kernels = [np.ones((15,15),np.uint8), np.ones((5,5),np.uint8), np.ones((50,50),np.uint8)]

# Create a figure
plt.figure(figsize=(20, 5))


plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis('off')
# Loop through the kernels and perform erosion
for i, kernel in enumerate(kernels):
    erosion = cv2.erode(image, kernel, iterations = 1)
    
    # Display the eroded image
    plt.subplot(1, 4, i+2)
    plt.imshow(erosion, cmap="gray")
    plt.title(f"Erosion with kernel size: {kernel.shape[0]}x{kernel.shape[1]}")
    plt.axis('off')

# Show all the images
plt.tight_layout()





image = cv2.imread("Shapes.tif", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis('off')

rectOpen = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((20,20),np.uint8))
plt.subplot(1, 4, 2)
plt.imshow(rectOpen, cmap="gray")
plt.title("Open Operation")
plt.axis('off')

rectClose = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((20,20),np.uint8))
plt.subplot(1, 4, 3)
plt.imshow(rectClose, cmap="gray")
plt.title("Close Operation")
plt.axis('off')

rectOpenClose = cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((20,20),np.uint8))
rectOpenClose = cv2.morphologyEx(rectOpenClose, cv2.MORPH_OPEN, np.ones((20,20),np.uint8))
plt.subplot(1, 4, 4)
plt.imshow(rectOpenClose, cmap="gray")
plt.title("Open Then Close Operation")
plt.axis('off')


##subprob 2
# Load the image
image = cv2.imread('Dowels.tif', cv2.IMREAD_GRAYSCALE)

# Generate a disk structuring element of radius 5
structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Perform open-close operation
open_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, structuring_element)
open_close_image = cv2.morphologyEx(open_image, cv2.MORPH_CLOSE, structuring_element)

# Perform close-open operation
close_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, structuring_element)
close_open_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, structuring_element)

# Display the resultant images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(open_close_image, cmap='gray')
ax1.set_title('Open-Close Operation')
ax1.axis('off')

ax2.imshow(close_open_image, cmap='gray')
ax2.set_title('Close-Open Operation')
ax2.axis('off')

plt.tight_layout()


radii = [2, 3, 4, 5]

# Initialize images to original image for iterative processing
open_close_series_image = image.copy()
close_open_series_image = image.copy()

# Apply series of operations
for r in radii:
    # Create structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    
    # Open-close operation
    open_image = cv2.morphologyEx(open_close_series_image, cv2.MORPH_OPEN, se)
    open_close_series_image = cv2.morphologyEx(open_image, cv2.MORPH_CLOSE, se)
    
    # Close-open operation
    close_image = cv2.morphologyEx(close_open_series_image, cv2.MORPH_CLOSE, se)
    close_open_series_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, se)

# Display the resultant images side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(open_close_series_image, cmap='gray')
ax1.set_title('Series of Open-Close Operations')
ax1.axis('off')

ax2.imshow(close_open_series_image, cmap='gray')
ax2.set_title('Series of Close-Open Operations')
ax2.axis('off')

plt.tight_layout()

##subprob 3
# Read the image
image_small_squares = cv2.imread("SmallSquares.tif", cv2.IMREAD_GRAYSCALE)

# Structuring element for hit-or-miss
structuring_element_hitmiss = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 0]
], dtype=np.uint8)

# Apply the hit-or-miss operation
result_hitmiss = cv2.morphologyEx(image_small_squares, cv2.MORPH_HITMISS, structuring_element_hitmiss)

# Display the resultant image
plt.figure(figsize=(6, 6))
plt.imshow(result_hitmiss, cmap="gray")
plt.title("Resultant Image")
plt.axis('off')
plt.tight_layout()
plt.show()

# Count the number of foreground pixels
foreground_pixel_count = np.sum(result_hitmiss > 0)
print(f"Number of foreground pixels that satisfy the conditions: {foreground_pixel_count}")

plt.show()



##problem 2


def FindComponentLabels(im, se):
    """
    Function to label connected objects in a binary image.
    
    Parameters:
    - im: Binary input image
    - se: Structuring element
    
    Returns:
    - labelIm: Labeled image
    - num: Number of connected objects
    """
    
    # Initialize labels and output image
    label = 1
    labelIm = np.zeros_like(im)
    
    # Create a set to keep track of visited points
    visited = set()
    
    # Iterate over each pixel in the image
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            
            # If the pixel is a foreground pixel, not visited, and not already labeled
            if im[i, j] == 255 and (i, j) not in visited and labelIm[i, j] == 0:
                
                # Start a new component label
                current_label = label
                label += 1
                
                # Set of component pixels
                component_pixels = {(i, j)}
                
                # While there are pixels in the component set
                while component_pixels:
                    new_pixels = set()
                    
                    # For each pixel in the component set
                    for x, y in component_pixels:
                        
                        # Mark as visited
                        visited.add((x, y))
                        
                        # Label the pixel
                        labelIm[x, y] = current_label
                        
                        # Check neighbors
                        for dx in range(-se.shape[0]//2, se.shape[0]//2 + 1):
                            for dy in range(-se.shape[1]//2, se.shape[1]//2 + 1):
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < im.shape[0] and 0 <= ny < im.shape[1] and
                                    im[nx, ny] == 255 and (nx, ny) not in visited and
                                    se[dx + se.shape[0]//2, dy + se.shape[1]//2] == 1):
                                    new_pixels.add((nx, ny))
                    
                    component_pixels = new_pixels
                    
    # Return the labeled image and number of components
    num = label - 1
    return labelIm, num

# Define the structuring element (a 3x3 square)
structuring_element = np.ones((3, 3), dtype=np.uint8)

# Threshold the ball_image to create a binary image
ball_image = cv2.imread("ball.tif",cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(ball_image, 127, 255, cv2.THRESH_BINARY)

# Call the function on the binary image
labelIm, num = FindComponentLabels(binary_image, structuring_element)

# Display the labeled components
print("Components found by my function: " + str(num))
plt.figure(figsize=(10, 10))
plt.imshow(labelIm, cmap='nipy_spectral')
plt.title("Labeled Components using FindComponentLabels")
plt.axis('off')
plt.show()

ret, labels = cv2.connectedComponents(binary_image)

# Display the labeled components using the built-in function
plt.figure(figsize=(10, 10))
plt.imshow(labels, cmap='nipy_spectral')
plt.title("Labeled Components using OpenCV's built-in function")
plt.axis('off')
plt.show()

print("Labeled components found by builtin: " + str(ret-1))

def extract_border_particles(binary_img, labels_img):
    """
    Function to extract particles residing on the border of the image.
    
    Parameters:
    - binary_img: Binary input image
    - labels_img: Labeled image
    
    Returns:
    - border_img: Image containing only the border particles
    """
    
    # Create an empty image
    border_img = np.zeros_like(binary_img)
    
    # Extract unique labels on the border
    top_labels = set(labels_img[0, :])
    bottom_labels = set(labels_img[-1, :])
    left_labels = set(labels_img[:, 0])
    right_labels = set(labels_img[:, -1])
    
    # Union of all border labels
    border_labels = top_labels.union(bottom_labels).union(left_labels).union(right_labels)
    
    # Remove background label (0)
    if 0 in border_labels:
        border_labels.remove(0)
    
    # Extract particles with labels on the border
    for label in border_labels:
        border_img[labels_img == label] = 255
    
    return border_img

# Extract border particles
border_image = extract_border_particles(binary_image, labelIm)

# Display the original and border images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 15))

axes[0].imshow(ball_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(border_image, cmap='gray')
axes[1].set_title("Particles Residing on the Border")
axes[1].axis('off')

plt.tight_layout()
plt.show()

def estimate_particle_size(binary_img, labels_img):
    """
    Function to estimate the size of an individual particle.
    
    Parameters:
    - binary_img: Binary input image
    - labels_img: Labeled image
    
    Returns:
    - estimated_size: Estimated size of an individual particle
    """
    
    # Count the number of pixels for each label
    label_counts = np.bincount(labels_img.ravel())
    
    # Remove the background count (label 0)
    label_counts = label_counts[1:]
    
    # Sort the counts and take the median as the estimated size
    sorted_counts = np.sort(label_counts)
    estimated_size = np.median(sorted_counts)
    
    return estimated_size

def extract_overlapping_particles(binary_img, labels_img, estimated_size):
    """
    Function to extract overlapping particles.
    
    Parameters:
    - binary_img: Binary input image
    - labels_img: Labeled image
    - estimated_size: Estimated size of an individual particle
    
    Returns:
    - overlap_img: Image containing only the overlapping particles
    """
    
    # Create an empty image
    overlap_img = np.zeros_like(binary_img)
    
    # Count the number of pixels for each label
    label_counts = np.bincount(labels_img.ravel())
    
    # Find labels with count greater than estimated size
    overlapping_labels = np.where(label_counts > 1.5 * estimated_size)[0]
    
    # Remove background label (0)
    overlapping_labels = set(overlapping_labels) - {0}
    
    # Extract particles with labels that are overlapping
    for label in overlapping_labels:
        overlap_img[labels_img == label] = 255
    
    return overlap_img

# Estimate particle size
particle_size = estimate_particle_size(binary_image, labelIm)

# Extract overlapping particles
overlapping_image = extract_overlapping_particles(binary_image, labelIm, particle_size)

# Display the original and overlapping images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 15))

axes[0].imshow(ball_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(overlapping_image, cmap='gray')
axes[1].set_title("Overlapping Particles")
axes[1].axis('off')

plt.tight_layout()
plt.show()


def extract_partial_border_particles(binary_img, labels_img, estimated_size):
    """
    Function to extract partial particles residing on the border of the image.
    
    Parameters:
    - binary_img: Binary input image
    - labels_img: Labeled image
    - estimated_size: Estimated size of an individual particle
    
    Returns:
    - partial_border_img: Image containing only the partial border particles
    """
    
    # Create an empty image
    partial_border_img = np.zeros_like(binary_img)
    
    # Extract unique labels on the border
    top_labels = set(labels_img[0, :])
    bottom_labels = set(labels_img[-1, :])
    left_labels = set(labels_img[:, 0])
    right_labels = set(labels_img[:, -1])
    
    # Union of all border labels
    border_labels = top_labels.union(bottom_labels).union(left_labels).union(right_labels)
    
    # Remove background label (0)
    if 0 in border_labels:
        border_labels.remove(0)
    
    # Count the number of pixels for each label
    label_counts = np.bincount(labels_img.ravel())
    
    # Extract particles with labels on the border and size less than estimated size
    for label in border_labels:
        if label_counts[label] < 0.8 * estimated_size:  # 80% threshold for partial
            partial_border_img[labels_img == label] = 255
    
    return partial_border_img

# Extract partial border particles
partial_border_image = extract_partial_border_particles(binary_image, labelIm, particle_size)

# Display the original and partial border images side-by-side
fig, axes = plt.subplots(1, 2, figsize=(15, 15))

axes[0].imshow(ball_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(partial_border_image, cmap='gray')
axes[1].set_title("Partial Particles on the Border")
axes[1].axis('off')

plt.tight_layout()
plt.show()


def border_particle_count(labelIm, border_image):
    """
    Function to count the unique labels corresponding to border particles.

    Parameters:
    - labelIm: Labeled image
    - border_image: Image of border particles

    Returns:
    - count: Number of unique labels corresponding to border particles
    """
    
    # Extract labels from the labeled image where the border image has values of 255
    labels_from_original = labelIm[border_image == 255]

    # Count unique labels excluding the background label
    unique_labels = np.unique(labels_from_original)
    unique_labels = unique_labels[unique_labels != 0]
    
    count = len(unique_labels)
    return count

def overlapping_particle_count(labelIm, overlapping_image):
    """
    Function to count the unique labels corresponding to overlapping particles.

    Parameters:
    - labelIm: Labeled image
    - overlapping_image: Image of overlapping particles

    Returns:
    - count: Number of unique labels corresponding to overlapping particles
    """
    
    # Extract labels from the labeled image where the overlapping image has values of 255
    labels_from_original = labelIm[overlapping_image == 255]

    # Count unique labels excluding the background label
    unique_labels = np.unique(labels_from_original)
    unique_labels = unique_labels[unique_labels != 0]
    
    count = len(unique_labels)
    return count

def partial_border_particle_count(labelIm, partial_border_image):
    """
    Function to count the unique labels corresponding to partial border particles.

    Parameters:
    - labelIm: Labeled image
    - partial_border_image: Image of partial border particles

    Returns:
    - count: Number of unique labels corresponding to partial border particles
    """
    
    # Extract labels from the labeled image where the partial border image has values of 255
    labels_from_original = labelIm[partial_border_image == 255]

    # Count unique labels excluding the background label
    unique_labels = np.unique(labels_from_original)
    unique_labels = unique_labels[unique_labels != 0]
    
    count = len(unique_labels)
    return count

# Using the new functions to get corrected counts
border_count = border_particle_count(labelIm, border_image)
overlap_count = overlapping_particle_count(labelIm, overlapping_image)
partial_border_count = partial_border_particle_count(labelIm, partial_border_image)

print("Number of connected particles residing on the border:", border_count)

print("Number of overlapping connected particles not residing on the border:", overlap_count)

print("Number of visually partial individual round particles on the border:", partial_border_count)

