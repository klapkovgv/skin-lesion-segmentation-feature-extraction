# skin-lesion-segmentation-feature-extraction
This project is an image processing pipeline for skin lesion segmentation and feature extraction using the ISIC 2018 dataset. It covers the process from preprocessing (cropping, contrast enhancement) to ROI extraction and statistical feature calculation.

## Step 1: RGB to Grayscale Conversion

For this project, I used the `isic_binary_augmented` dataset from Hugging Face. Since the images are stored as dictionaries within the dataframe, I decoded them into Numpy arrays using a custom `get_image_as_array` function. 

As the first processing step, I performed RGB to grayscale conversion using the OpenCV library with the function `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)`. This conversion reduces the image complexity from three color channels to a single intensity channel. 

The selected sample images for visualization are indices: 14266, 27, and 8654.

## Step 2: Pre-processing

I applied a fixed 18% crop to the edges of each image. This step was necessary to remove black corners and peripheral artifacts commonly found in the ISIC dataset. By removing these dark borders, I significantly reduced the risk of false-positive results during the segmentation stage. An 18% crop was chosen as a balanced strategy to eliminate noise without losing the main Region of Interest (ROI) in larger lesions.

For contrast enhancement, I implemented the Contrast Stretching method. This technique effectively expands the dynamic range of pixel intensities, making the dark lesion more distinct against the lighter healthy skin. Contrast Stretching provides a more natural result that highlights the lesion's boundaries without over-amplifying background noise. 

As the final pre-processing step, I applied a Median Blur with a kernel size of 5. This filter is superior for medical skin images because it effectively removes small artifacts, such as fine hairs and salt-and-pepper noise, while preserving the sharp edges of the lesion. This ensures that boundaries remain clear for accurate shape feature extraction in later steps.
