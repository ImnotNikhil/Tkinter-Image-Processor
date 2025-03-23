import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import heapq
import os

# Function to select the image from user's system
def select_image():
    global img, img_display
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if not file_path:
        return
    
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_display = ImageTk.PhotoImage(img)
    label_img.config(image=img_display)
    label_img.image = img_display
    label_img.file_path = file_path
    label_status.config(text="Image selected")

# Function for processing the image (upsample/downsample)
def process_image(mode):
    if not hasattr(label_img, "file_path"):
        messagebox.showerror("Error", "Please select an image first!")
        return
    
    # Load image using OpenCV (as we need to apply advanced processing like DCT)
    image = cv2.imread(label_img.file_path)
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(ycbcr_image)

    # Chroma Subsampling (4:2:0)
    Cb_sub = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
    Cr_sub = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # Padding to fit 8x8 blocks
    block_size = 8
    new_rows = (Y.shape[0] + block_size - 1) // block_size * block_size
    new_cols = (Y.shape[1] + block_size - 1) // block_size * block_size
    Y_padded = cv2.copyMakeBorder(Y, 0, new_rows - Y.shape[0], 0, new_cols - Y.shape[1], cv2.BORDER_REPLICATE)

    # DCT & Quantization
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    Y_blocks = Y_padded.reshape(new_rows // 8, 8, -1, 8).swapaxes(1, 2)
    dct_Y = np.array([[cv2.dct(block.astype(np.float32)) for block in row] for row in Y_blocks])
    quantized_Y = np.round(dct_Y / Q).astype(np.int32)

    # Zig-Zag Scan
    zigzag_indices = np.array(sorted(((x, y) for x in range(8) for y in range(8)), key=lambda s: (s[0] + s[1], (s[0] + s[1]) % 2 == 0)))
    def zigzag_scan(blocks):
        return blocks[:, :, zigzag_indices[:, 0], zigzag_indices[:, 1]]
    
    zigzag_Y = zigzag_scan(quantized_Y)

    # Run-Length Encoding
    def run_length_encode(arr):
        marker = -9999  
        diff = np.diff(np.concatenate(([marker], arr, [marker])))
        indices = np.where(diff != 0)[0]
        values = arr[indices[:-1]]
        counts = np.diff(indices)
        return np.column_stack((values, counts)).flatten()

    rle_Y = np.array([run_length_encode(zigzag_Y[i, j]) for i in range(zigzag_Y.shape[0]) for j in range(zigzag_Y.shape[1])], dtype=object)

    # Huffman Encoding
    def huffman_encode(data):
        shift = abs(min(data))
        freq = np.bincount(data + shift)
        heap = [[weight, [symbol - shift, ""]] for symbol, weight in enumerate(freq) if weight > 0]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        huff_dict = {pair[0]: pair[1] for pair in heap[0][1:]}  
        encoded_data = "".join(huff_dict.get(symbol, "") for symbol in data)

        return huff_dict, encoded_data

    huffman_dict, huffman_Y = huffman_encode(np.concatenate(rle_Y))

    # Process image according to the selected mode (upsample or downsample)
    if mode == "upsample":
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    elif mode == "downsample":
        image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)

    # Save the processed image as JPEG
    output_path = os.path.splitext(label_img.file_path)[0] + f"_{mode}.jpg"
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Display success message
    messagebox.showinfo("Success", f"Image processed and saved as {output_path}")
    label_status.config(text="Processing complete")

# Function to exit the application
def exit_app():
    root.destroy()

# Initialize Tkinter
root = tk.Tk()
root.title("Image Processor")
root.geometry("400x500")

# Button to select an image
btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack(pady=10)

# Label to display selected image
label_img = tk.Label(root)
label_img.pack()

# Status label
label_status = tk.Label(root, text="No image selected", fg="red")
label_status.pack()

# Buttons for processing options
btn_upsample = tk.Button(root, text="Upsample", command=lambda: process_image("upsample"))
btn_upsample.pack(pady=5)

btn_downsample = tk.Button(root, text="Downsample", command=lambda: process_image("downsample"))
btn_downsample.pack(pady=5)

# Exit button
btn_exit = tk.Button(root, text="Exit", command=exit_app)
btn_exit.pack(pady=10)

# Start the Tkinter main loop
root.mainloop()
