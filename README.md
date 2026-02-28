# skin-lesion-segmentation-feature-extraction
This project is an image processing pipeline for skin lesion segmentation and feature extraction using the ISIC 2018 dataset. It covers the process from preprocessing (cropping, contrast enhancement) to ROI extraction and statistical feature calculation.

## Step 1: RGB to Grayscale Conversion

For this project, I used the `isic_binary_augmented` dataset from Hugging Face. Since the images are stored as dictionaries within the dataframe, I decoded them into Numpy arrays using a custom `get_image_as_array` function. 

As the first processing step, I performed RGB to grayscale conversion using the OpenCV library with the function `cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)`. This conversion reduces the image complexity from three color channels to a single intensity channel. 

The selected sample images for visualization are indices: 14266, 24, and 8654.
