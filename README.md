# 🏦💳 Bank Card and National ID OCR

**Final Project for Fundamentals of Computer Vision (Summer 1402)**
*Iran University of Science and Technology - Faculty of Computer Engineering*

**Authors:**
* Mohammad Darmanloo (99521262)
* Seyed Mohammad Ali Fakhari (99521496)

**Instructor:** Dr. Mohammadi

---

## 📖 Overview

This project is an Optical Character Recognition (OCR) pipeline designed to read the 16-digit number from bank cards and the 10-digit number from Iranian national ID cards.

The system first detects and crops the card from an input image, then identifies the card type (bank vs. national ID), and finally uses one of two different trained neural networks to recognize and extract the corresponding numbers.

### 🎥 Features

* **Hybrid Card Detection:** Uses Semantic Segmentation as the primary method and a simple image processing approach as a fallback.
* **Robust Cropping:** Successfully isolates cards even against complex and "busy" backgrounds.
* **Card Type Identification:** Distinguishes between bank cards and national ID cards using template matching.
* **Dual OCR Models:**
    * A custom CNN for recognizing English digits on bank cards.
    * A transfer-learning model (ResNet50) for recognizing Persian digits on national ID cards.

## ⚙️ Pipeline Architecture

The main pipeline logic is orchestrated in the `CV_Project2_OCR.ipynb` notebook. The `predict_card(path)` function executes the following steps:

### 1. Card Cropping

To read the information, the card must first be isolated from the image. This project implements a robust two-pronged approach.

#### Method 1: Semantic Segmentation (Primary)
The primary method uses a **U-Net** architecture with a **MobileNet** backbone for semantic segmentation.

* **Dataset:** This model was trained on the **MIDV500** dataset, which contains diverse images of identification cards.
* **Process:** The image is preprocessed, resized, and fed into the model (`model.predict`). The resulting mask is used to find the card's contours, and `warpPerspective` is applied to get a top-down, cropped image.
* **Result:** This method is highly effective and can handle cluttered backgrounds where simple image processing fails.

#### Method 2: Simple Image Processing (Fallback)
If the semantic segmentation model fails to return a clean crop, the pipeline falls back to a traditional computer vision method using OpenCV.

* **Process:**
    1.  Denoise the image using `cv2.fastNlMeansDenoising`.
    2.  Convert to grayscale and apply adaptive thresholding.
    3.  Invert the pixels and use a morphological **closing operation** with a large circular kernel (radius 17) to connect the card area into a single white blob.
    4.  Find the edges using `cv2.Canny`.
    5.  Find the largest 4-sided contour and use its corners to apply `cv2.warpPerspective` and crop the card.
* **Limitation:** This method struggles with "busy backgrounds".

### 2. Card Type Detection
After cropping, the `detect_card()` function determines the card type.

* **Method:** It uses `cv2.matchTemplate` to search the cropped image for three pre-saved template logos from the Iranian national ID card.
* **Logic:** If the match confidence is high (e.g., > 0.5), it's classified as a "National Card"; otherwise, it's a "Credit Card".

### 3. Number Recognition

Based on the card type, a specific OCR pipeline is called.

#### Bank Card (16 English Digits)
1.  **Localization:** The 16-digit number block is located using morphological operators (open/close) to connect the digits into one large contour.
2.  **Segmentation:** The 16-digit region is cropped, and a second contour search is performed to find each individual digit.
3.  **Recognition:** Each digit is resized to 32x32 and passed to a custom-trained CNN (`model2`) for classification.
    * **Model:** A Sequential CNN with ~8.25M parameters.
    * **Dataset:** Trained on the [English Font/Number Recognition dataset](https://www.kaggle.com/datasets/yaswanthgali/english-fontnumber-recognition) from Kaggle.
    * **Performance:** Achieved ~94% training accuracy and ~98-99% validation accuracy.

#### National ID Card (10 Persian Digits)
1.  **Localization:** The 10-digit number is located using a hardcoded coordinate crop (a specific region of interest) on the 600x600 resized national card.
2.  **Segmentation:** Individual Persian digits are found using contours.
3.  **Recognition:** Each digit is resized to 64x64 and passed to a fine-tuned ResNet50 model (`model_pesian_number`) for classification.
    * **Model:** A pre-trained **ResNet50** model (on ImageNet) with new pooling and dense layers added for transfer learning.
    * **Dataset:** Trained on the [Persian Alpha/Number dataset](https://www.kaggle.com/datasets/mehdisahraei/persian-alpha) from Kaggle.
    * **Performance:** Reached 100% accuracy quickly. Data augmentation was used to improve generalization.

## 🤔 Other Approaches Explored

### YOLOv7
A YOLOv7 model was also trained on a custom dataset (created using Roboflow) for card detection. While the results were "generally good," this approach was **not used** in the final pipeline for two main reasons:
1.  The bounding box was not precise enough for the required geometric cropping.
2.  It required several external libraries not permitted by the project's scope.

## 🚀 How to Run

1.  Ensure you have all required libraries (see `CV_Project2_OCR.ipynb` imports).
2.  Download the trained models:
    * Semantic Segmentation Model (`card-v2`)
    * English Digit CNN (`detect-credit-number-v2`)
    * Persian Digit ResNet50 (`presian_number_model.h5`)
3.  Load the models in the notebook.
4.  Call the `predict_card(path_to_your_image)` function to run the full pipeline.
