# ImageSegmentationApp

## Overview
**ImageSegmentationApp** is a Python-based graphical user interface (GUI) application for image segmentation and filtering. The application allows users to open an image, convert it to grayscale, and apply various filters such as edge detection, line detection, and custom user-defined filters.

## Features
- Open and display images.
- Convert images to grayscale.
- Apply various predefined filters including Sobel, Prewitt, Laplacian, and more.
- User-defined filters with customizable size and coefficients.
- Save the processed image.
- Simple and intuitive GUI built with Tkinter.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Omar-Raddad/ImageSegmentationApp.git
   cd ImageSegmentationApp
2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt

##Usage

1. Run the application:
   ```bash
     python ImageSegmentationApp.py

2. Using the GUI:

• Open Image: Select and open an image file.

• Gray Scale: Convert the opened image to grayscale.

• Apply Filters: Choose from a variety of predefined filters to apply to the grayscale image.

• User Defined: Enter custom filter size and coefficients to apply your own filter.

• Save Image: Save the processed image to your local machine.

• Exit: Close the application.

## Filter Options

• Point Detection

• Horizontal Line Detection

• Vertical Line Detection

• +45 Line Detection

• -45 Line Detection

• LOG Filter

• Sobel Filter

• Prewitt Filter

• Laplacian

• Zero Crossing

• Horizontal Edge

• Vertical Edge

• -45 Edge

• +45 Edge

• Threshold

• Adaptive Threshold

• Horizontal Prewitt

• Vertical Prewitt

• -45 Prewitt

• +45 Prewitt

##Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
• Tkinter for the GUI

• OpenCV for image processing

• Pillow for image handling
