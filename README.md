# skin-lesion-segmentation-feature-extraction

>  This is an educational project developed during my university studies, reflecting my hands-on experience.

This project is an image processing pipeline for skin lesion segmentation and feature extraction using the ISIC 2018 dataset. It covers the process from preprocessing (cropping, contrast enhancement) to ROI extraction and statistical feature calculation.

## Step 1: RGB to Grayscale Conversion

For this project, I used the `isic_binary_augmented` (https://huggingface.co/ahishamm/isic_binary_augmented) dataset from Hugging Face. Since the images are stored as dictionaries within the dataframe.

As the first processing step, I performed RGB to grayscale conversion using the OpenCV library with the function `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)`. This conversion reduces the image complexity from three color channels to a single intensity channel. 

The selected sample images for visualization are indices: 14266, 27, and 8654.

## Step 2: Pre-processing

I applied a fixed 18% crop to the edges of each image. This step was necessary to remove black corners and peripheral artifacts commonly found in the ISIC dataset. By removing these dark borders, I significantly reduced the risk of false-positive results during the segmentation stage. An 18% crop was chosen as a balanced strategy to eliminate noise without losing the main Region of Interest (ROI) in larger lesions.

For contrast enhancement, I implemented the Contrast Stretching method. This technique effectively expands the dynamic range of pixel intensities, making the dark lesion more distinct against the lighter healthy skin. Contrast Stretching provides a more natural result that highlights the lesion's boundaries without over-amplifying background noise. 

As the final pre-processing step, I applied a Median Blur with a kernel size of 5. This filter is superior for medical skin images because it effectively removes small artifacts, such as fine hairs and salt-and-pepper noise, while preserving the sharp edges of the lesion. This ensures that boundaries remain clear for accurate shape feature extraction in later steps.

## Step 3: Thresholding (Segmentation)

Image segmentation is the process of partitioning a digital image into multiple segments (set of pixels). The goal of segmentation is to simplify the representation of an image into something that is more meaningful and easier to analyze. 

In this project, segmentation is used to locate the Region of Interest, or the skin lesion, and separate it from the background (healthy skin). Successful segmentation is the most critical step because all subsequent feature extractions depend on the accuracy of the lesion's boundaries.

Thresholding is the simplest and most common method of image segmentation. It is a non-linear operation that converts a grayscale image into a binary image (an image with only two colors: black and white). The process works by choosing a specific intensity value called a threshold (T):
- pixels with intensity values above T are assigned to one class;
- pixels with intensity values below T are assigned to another class.

I applied three different thresholding methods to separate the lesion from the healthy skin. I will compare their threshold values, visualize the results on the selected images, and justify why a specific method was chosen as the most effective.

Since skin lesions are typically **darker** than healthy skin, I applied **inverted binary thresholding** (`THRESH_BINARY_INV`). In the resulting masks, the lesion pixels are white (255) and the background (healthy skin) is black (0).

Before implementing thresholding methods, I researched the OpenCV and Scikit-Image libraries. In the Global thresholding (or simple thresholding), the same threshold value is applied to every pixel. I used the `cv2.threshold` function with `cv2.THRESH_BINARY_INV` flag. In this mode, if a pixel value is smaller than or equal to the threshold, it is set to the maximum value (255), otherwise, it is set to 0. `_, thresh_global = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY_INV)` is my code.
- the first argument is the source grayscale image;
- the second argument is the threshold value (140), used to classify the pixels;
- the third argument is the maximum value (255) assigned to pixels that satisfy the condition;
- the final argument is the thresholding type, which in my case is inverted binary.

Otsu's method is an automatic thresholding algorithm that calculates the optimal threshold by **minimizing the intra-class variance**. It treats the image histogram as a bimodal distribution and finds the threshold that maximizes the spread between the two peaks. This makes it highly effective for images where the contrast between the lesion and skin varies. My implementation is `val = threshold_otsu(img)` followed by img < val.

Li's thresholding is based on the **Minimum Cross-Entropy** criterion proposed by Li and Lee in 1993. The algorithm finds a threshold that minimizes the cross-entropy between the original image and the resulting binary mask. It is particularly robust for medical images where the foreground (lesion) might be much smaller than the background. My implementation is `li_val = threshold_li(img)`. 

*Cross-entropy is a measure from information theory that quantifies the difference between two distributions. The algorithm finds a threshold that minimizes the discrepancy between the original grayscale image and the resulting binary mask. By minimizing this cross-entropy, the method ensures that the segmentation regions preserve the information and average intensities of the original image as accurately as possible.*

<img width="1218" height="478" alt="1" src="https://github.com/user-attachments/assets/c150696a-759e-4980-98d8-238262457044" />

To choose the best value for Global Thresholding, I analyzed the histograms of the pre-processed images. It is important to understand the following:
- The left part of the histogram (low intensity values) represents the dark pixels, which corresponds to the lesion;
- The right part of the histogram (high intensity values) represents the light pixels, which corresponds to the healthy skin.

The goal of thresholding is to find the minimum point between these two peaks to separate the lesion from the skin effectively. For the Global Thresholding method, I chose the value 140. I found that 140 is the optimal average of Otsu because it consistently falls into the minimum point of the histogram. 

For my project, I chose **Otsu Thresholding** because it provides the best balance between reducing background noise and preserving the full Region of Interest.

## Step 4: Post-Processing

We began by defining morphological transformations. These are simple operations based on the image shape, typically performed on binary images. These operations require two inputs: the original binary mask and a structuring element (kernel) which decides the nature of the operation. The two most fundamental operators are Erosion and Dilation, from which more complex forms like Opening and Closing are derived.

In my project, I utilized the **Opening** morphological operation. Opening is specifically defined as erosion followed by dilation. It is the most appropriate technique here because it effectively removes small noise pixels that are detached from the main target. As described in the official OpenCV documentation, it is highly useful for noise removal using the function `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)`. 

The underlying mechanism of **Erosion** is similar to soil erosion; it erodes away the boundaries of the foreground object (white pixels). During this process, the kernel slides through the image like a 2D convolution. A pixel in the original image is retained as a '1' only if all pixels under the kernel are '1'; otherwise, it is eroded to '0'. Consequently, all pixels near the boundary are discarded depending on the size of the kernel, decreasing the thickness of the white region. This is essential for removing small white "salt" noise and detaching objects that might be weakly connected.

I chose an **Elliptical Kernel (cv2.MORPH_ELLIPSE)** with a 7x7 size. An elliptical kernel is superior to a rectangular one for medical imaging because it treats corners more smoothly. Since biological structures, such as lesions, are rarely perfectly square, the ellipse preserves the natural organic curvature of the boundaries. The 7x7 size was chosen as a balanced parameter: large enough to eliminate noise clusters, but small enough to preserve the integrity of the actual ROI. 

Following the morphological cleanup, I applied **Connected Component Labeling (CCL)** using the function `cv2.connectedComponentsWithStats(morphed, connectivity=8)`.

CCL is an algorithmic technique used to detect and group 'islands' of pixels. Every isolated group of connected pixels is assigned a unique integer ID or label. I used a connectivity of 8, which means the algorithm considers pixels connected if they touch horizontally, vertically, or diagonally. This provides a more comprehensive detection than 4-connectivity. The 'WithStats' version of this function is particularly useful as it calculates metadata for every detected object, including its bounding box and its total area in pixels.

To ensure the final output contains only the target lesion, I implemented a **Largest Area strategy**. The logic of the implementation is as follows:

`largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])`

- The stats matrix stores the background in the first row (index 0). By slicing the array with `1:`, we ignore the background and search only through the foreground objects. `np.argmax` finds the index of the largest area among these objects. We add `+1` back to the result to align it with the original label IDs, ensuring we correctly identify the primary ROI.

`final_mask = (labels == largest_label_idx).astype(np.uint8) * 255`
- This line creates the final binary mask by isolating only the pixels belonging to the largest label ID. All other smaller noise components are discarded (turned to 0), resulting in a clean ROI.

`component_counts.append(num_labels - 1)`
- Finally, we track the number of components found by subtracting 1 to exclude the background.

To qualitatively evaluate the segmentation accuracy, I used the `cv2.addWeighted` function to create an ROI overlay. 

`overlay = cv2.addWeighted(blurred, 0.7, final_mask, 0.3, 0)`
- this blending technique superimposed the final binary mask onto the preprocessed grayscale image. By assigning a weight of 0.7 to the image and 0.3 to the mask, we can visually verify the detected ROI.

## Step 5: Feature Extraction

In this stage, I perform feature extraction from the segmented Region of Interest where the goal is to convert the visual information into a structured numerical format. Features are extracted in three distinct categories: First-Order Statistics, 2D Shape features, and Second-Order Texture features (GLCM).

**First-Order features** describe the distribution of grayscale pixel intensities within the ROI, ignoring spatial relationships. I implemented a function to calculate the following 10 statistical measures:
- central tendency: mean and median;
- dispersion: standard deviation, minimum and maximum values;
- shape of distribution: skewness (asymmetry) and Kurtosis (peakedness);
- information theory: entropy (complexity of the pixel distribution) and energy (uniformity).

**2D Shape Features** are to quantify the geometric properties of the detected lesions. I utilized the `skimage.measure.regionprops` library.` the following shape descriptors were extracted:
- size: area, convex area, perimeter, and equivalent diameter;
- geometry: major and mnor axis lengths, and the calculated aspect ratio;
- compactness & complexity: circularity (how close the shape is to a circle), Eccentricity (how elongated the shape is), Solidity (ratio of area to convex area), and Extent (ratio of area to bounding box area).

**Second-Order Texture Features** describe the spatial relationship between pairs of pixels. I used the gray-level co-occurrence matrix (GLCM) for this purpose. Parameters and preprocessing:
- quantization: to reduce computational complexity and noise sensitivity, the original 256-level grayscale image was quantized into 32 levels (by dividing pixel values by 8);
- settings: I used a distance of 1 pixel and an angle of 0 degrees;
- features: from glcm, I extracted contrast, dissimilarity, homogeneity, energy, correlation, and ASM (Angular Second Moment).

The extraction process follows the same pipeline as the post-processing step to ensure consistency. For every image in the dataset, the ROI is isolated, features are computed, and the results are organized into a structured table. 
