"""
Module for parsing board images to extract individual drawings.
"""

import cv2
import numpy as np


class BoardParser:
    def __init__(self, **kwargs):
        """
        Initializes the BoardParser with optional parameters.

        Args:
            kwargs: Optional parameters for border detection and dilation.
        """
        # Border detection parameters
        self.canny_lower_thresh = kwargs.get('canny_lower_thresh', 5)
        self.canny_upper_thresh = kwargs.get('canny_upper_thresh', 80)
        self.blur_kernel_size = kwargs.get('blur_kernel_size', (5, 5))

        # Dilation parameters
        self.dilation_kernel_size = kwargs.get('dilation_kernel_size', (25, 25))
        self.dilation_iterations = kwargs.get('dilation_iterations', 5)

        # Connectivity for connected components
        self.connectivity = kwargs.get('connectivity', 8)

    def __find_borders(self, image: np.ndarray) -> list[np.ndarray]:
        """Finds contours in the image that represent borders of components."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        edges = cv2.Canny(blurred, self.canny_lower_thresh, self.canny_upper_thresh)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def __separate_ink(self, image: np.ndarray, masks: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Processes an image against multiple masks to separate ink from background.
        
        Args:
            image (np.array): The source image (BGR).
            masks (list of np.array): List of single-channel binary masks 
                                    (same height/width as image).
                                    
        Returns:
            tuple: (output_image, ink_colors)
                - output_image: A white background image with extracted ink in black.
                - ink_colors: A list of (B, G, R) tuples representing the ink color 
                            for each mask.
        """
        # Initialize output image as solid white
        h, w = image.shape[:2]
        output_image = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        ink_colors = []

        for mask in masks:
            # 1. Extract valid pixels: We only want to cluster pixels inside the mask
            # valid_pixels will be a list of [B, G, R] values
            valid_indices = np.where(mask > 0)
            valid_pixels = image[valid_indices]

            # Check if mask is not empty
            if len(valid_pixels) == 0:
                ink_colors.append(None)
                continue

            # 2. Prepare data for K-Means (must be float32)
            data = np.float32(valid_pixels)

            # 3. Apply K-Means with k=2
            # criteria: (type, max_iter, epsilon)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2
            compactness, labels, centers = cv2.kmeans(
                data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            # 4. Identify the "Ink" cluster (the darker one)
            # We sum the BGR channels to determine brightness. Lower sum = darker.
            center_sums = np.sum(centers, axis=1)
            ink_cluster_index = np.argmin(center_sums)
            
            # Store the ink color (convert back to uint8 tuple for readability)
            ink_bgr = tuple(map(int, centers[ink_cluster_index]))
            ink_colors.append(ink_bgr)

            # 5. Map ink pixels to the output image
            # flatten labels to match valid_pixels length
            labels = labels.flatten()
            
            # Find which pixels in our 'valid_pixels' array belong to the ink cluster
            ink_pixel_mask = (labels == ink_cluster_index)
            
            # Get the original (y, x) coordinates for these specific ink pixels
            # valid_indices is a tuple (y_array, x_array). We filter them by the ink mask.
            ink_y_coords = valid_indices[0][ink_pixel_mask]
            ink_x_coords = valid_indices[1][ink_pixel_mask]

            # Set these coordinates to Black (0, 0, 0) on the white canvas
            output_image[ink_y_coords, ink_x_coords] = [0, 0, 0]

        return output_image, ink_colors
    
    def __aggregate_masks(self, masks: list[np.ndarray], shape: tuple[int, int], ink_colors: list[np.ndarray]) -> list[np.ndarray]:
        """
        Aggregate multiple binary masks into 2 aggregate binary masks based on 
        brightness of provided colors using K-means clustering.
        
        Args:
            masks (list of np.array): List of single-channel binary masks.
            shape (tuple): Shape of the output image (height, width).
            colors (list of tuple): List of (B, G, R) color tuples for each mask.
            
        Returns:
            np.array: Color-coded image where each mask is filled with its corresponding color.
        """
        final_masks = [np.zeros(shape, dtype=np.uint8) for _ in range(2)]

        ink_colors_array = np.array([color for color in ink_colors if color is not None], dtype=np.float32)
        if len(ink_colors_array) <= 0:
            return final_masks

        # Use brightness (sum of channels) for clustering
        brightness = np.sum(ink_colors_array, axis=1).reshape(-1, 1)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            brightness, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Ensure label 0 is the darker one (Black) and label 1 is the lighter one (Blue)
        if centers[0] > centers[1]:
            labels = 1 - labels
            centers = centers[::-1]

        for i, mask in enumerate(masks):
            if ink_colors[i] is None:
                continue
            label = labels[i][0]
            final_masks[label] = cv2.bitwise_or(final_masks[label], mask.astype(np.uint8) * 255)
        
        return final_masks

    def __extract_drawings(self, image: np.ndarray, masks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Finds connected components in a list of masks and returns a flat list of 
        crops from the original image.
        
        Args:
            image (np.array): The source image (BGR).
            masks (list of np.array): List of binary masks.
            dilation_kernel_size (tuple): Size of kernel to connect close components.
            iterations (int): How many times to apply dilation.
            
        Returns:
            list: A list of np.array images (crops).
        """
        all_crops = []
        
        # Define the kernel for dilation
        kernel = np.ones(self.dilation_kernel_size, np.uint8)

        for mask in masks:
            # 1. Morphological Operation: Dilate to connect broken/close parts
            # This merges individual ink spots/letters into larger blocks
            connected_mask = cv2.dilate(mask, kernel, iterations=self.dilation_iterations)
            
            # 2. Find Contours (External only to avoid crops inside crops)
            contours, _ = cv2.findContours(connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                # 3. Get Bounding Box
                x, y, w, h = cv2.boundingRect(cnt)
                
                # 4. Extract Crop
                # Ensure the crop has valid dimensions
                if w > 0 and h > 0:
                    crop = image[y:y+h, x:x+w]
                    all_crops.append(crop)

        return all_crops
    
    def parse_board(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Main method to parse the board image and extract individual drawings.

        Args:
            image (np.array): The source image (BGR).

        Returns:
            list: A list of np.array images (crops) representing individual drawings.
        """
        # Step 1: Find Borders
        contours = self.__find_borders(image)

        # Step 2: Create Masks from Contours
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,   # draw all
            color=255,
            thickness=50
        )
        num_labels, labels_im = cv2.connectedComponents(mask, connectivity=self.connectivity)
        masks = [labels_im == label for label in range(1, num_labels)]

        # Step 3: Separate Ink from Background
        ink_image, ink_colors = self.__separate_ink(image, masks)
        cv2.drawContours(ink_image, contours, -1, (0, 0, 0), 2)

        # Step 4: Aggregate Masks based on Ink Colors
        aggregated_masks = self.__aggregate_masks(masks, (h, w), ink_colors)

        # Step 5: Extract Individual Drawings
        drawings = self.__extract_drawings(ink_image, aggregated_masks)

        return drawings
        