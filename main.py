import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from functools import partial

class ImageSegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Segmentation App")
        self.master.configure(bg="#2b2b2b") 

        self.image_path = None
        self.original_image = None
        self.grayscale_image = None
        self.filter_image = None

        self.sidebar_frame = tk.Frame(self.master, width=200, bg="#2b2b2b")  
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.image_frame = tk.Frame(self.master, bg="#2b2b2b")  
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame, bg="#2b2b2b", fg="white") 
        self.image_label.pack(expand=True)

        self.user_filter_size = 3
        self.user_filter_coeffs = [0] * (self.user_filter_size ** 2)

        self.filter_map = {
            "Point Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
            "Horizontal Line Detection": np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32),
            "Vertical Line Detection": np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32),
            "+45 Line Detection": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32),
            "-45 Line Detection": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32),
            "LOG Filter": np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]], dtype=np.float32) / 16.0,
            "Sobel Filter": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
            "Prewitt Filter": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
            "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
            "Zero Crossing": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
            "Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
            "Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
            "-45 Edge": np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=np.float32),
            "+45 Edge": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32),
            "Threshold": None,  # Placeholder for custom thresholding
            "Adaptive Threshold": None,  # Placeholder for adaptive thresholding
            "Horizontal Prewit": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
            "Vertical Prewit": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
            "-45 Prewit": np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype=np.float32),
            "+45 Prewit": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=np.float32),
        }

        self.create_tool_bar()

    def create_tool_bar(self):
        tk.Button(self.sidebar_frame, text="Open Image", command=self.open_image, width=20, height=1, bg="#3c3c3c", fg="white").pack(pady=5)  # Dark background and white text
        tk.Button(self.sidebar_frame, text="Gray Scale", command=self.convert_to_grayscale, width=20, height=1, bg="#3c3c3c", fg="white").pack(pady=5)  # Dark background and white text
        tk.Label(self.sidebar_frame, text="Omar Raddad", font=("Arial", 8), bg="#2b2b2b", fg="white").pack(side=tk.BOTTOM , pady=10)

        button_frame_1 = tk.Frame(self.sidebar_frame, bg="#2b2b2b")
        button_frame_1.pack(side=tk.LEFT, padx=5)

        # Buttons in the first column
        for filter_name in list(self.filter_map.keys())[:len(self.filter_map)//2]:
            tk.Button(button_frame_1, text=filter_name, command=partial(self.apply_filter, self.filter_map[filter_name]), width=20, height=1, bg="#3c3c3c", fg="white").pack(pady=5)  # Dark background and white text

        button_frame_2 = tk.Frame(self.sidebar_frame, bg="#2b2b2b")
        button_frame_2.pack(side=tk.LEFT, padx=5)

        # Buttons in the second column
        for filter_name in list(self.filter_map.keys())[len(self.filter_map)//2:]:
            tk.Button(button_frame_2, text=filter_name, command=partial(self.apply_filter, self.filter_map[filter_name]), width=20, height=1, bg="#3c3c3c", fg="white").pack(pady=5)  # Dark background and white text

        tk.Label(self.sidebar_frame, text="Filter Size:", bg="#2b2b2b", fg="white").pack(pady=5)  # Dark background and white text
        self.filter_size_entry = tk.Entry(self.sidebar_frame, bg="#3c3c3c", fg="white")  # Dark background and white text
        self.filter_size_entry.pack(pady=5)

        tk.Label(self.sidebar_frame, text="Filter Coefficients (comma-separated):", bg="#2b2b2b", fg="white").pack(pady=5)  # Dark background and white text
        self.filter_coeffs_entry = tk.Entry(self.sidebar_frame, bg="#3c3c3c", fg="white")  # Dark background and white text
        self.filter_coeffs_entry.pack(pady=5)

        tk.Button(self.sidebar_frame, text="User Defined", command=self.apply_user_defined_filter, width=15, height=1, bg="#4f4f4f", fg="white").pack(pady=5)  # Dark background and white text

        # Add threshold entry
        tk.Label(self.sidebar_frame, text="Threshold Value:", bg="#2b2b2b", fg="white").pack(pady=5)  # Dark background and white text
        self.threshold_entry = tk.Entry(self.sidebar_frame, bg="#3c3c3c", fg="white")  # Dark background and white text
        self.threshold_entry.pack(pady=5)

        tk.Button(self.sidebar_frame, text="Save Image", command=self.save_image, width=20, height=1, bg="#3c3c3c", fg="white").pack(pady=5)  # Dark background and white text
        tk.Button(self.sidebar_frame, text="Exit", command=self.master.destroy, width=20, height=1, bg="#3c3c3c", fg="white").pack(pady=5)  # Dark background and white text

    def open_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            self.grayscale_image = self.original_image.convert("L")
            self.filter_image = self.grayscale_image.copy()  # Copy for filter application
            self.display_image(self.original_image)  # Display original image

    def convert_to_grayscale(self):
        if self.original_image:
            self.display_image(self.grayscale_image)

    def apply_filter(self, kernel):
        if self.filter_image and kernel is not None:
            np_image = np.array(self.grayscale_image)
            filtered_image = cv2.filter2D(np_image, -1, kernel)
            normalized_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX)

            # Apply thresholding for specific filters
            filter_name = [key for key, value in self.filter_map.items() if np.array_equal(value, kernel)]
            if filter_name and filter_name[0] in ["Threshold", "Adaptive Threshold"]:
                threshold_value = self.get_threshold_value()
                _, thresholded_image = cv2.threshold(normalized_image, threshold_value, 255, cv2.THRESH_BINARY)
                self.filter_image = Image.fromarray(thresholded_image.astype('uint8'))
            else:
                self.filter_image = Image.fromarray(normalized_image.astype('uint8'))

            self.display_image(self.filter_image)

    def apply_user_defined_filter(self):
        try:
            size = int(self.filter_size_entry.get())
            coeffs = [float(x.strip()) for x in self.filter_coeffs_entry.get().split(",")]
            kernel = np.array(coeffs).reshape(size, size)
            self.apply_filter(kernel)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def get_threshold_value(self):
        try:
            return int(self.threshold_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid threshold value. Please enter a valid integer.")
            return 0

    def save_image(self):
        if self.filter_image:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if save_path:
                self.filter_image.save(save_path)

    def display_image(self, image):
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
