import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

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

def process_image(mode):
    if not hasattr(label_img, "file_path"):
        messagebox.showerror("Error", "Please select an image first!")
        return
    
    img = Image.open(label_img.file_path)
    
    if mode == "upsample":
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    elif mode == "downsample":
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    else:
        messagebox.showerror("Error", "Invalid mode selected!")
        return
    
    save_path = os.path.splitext(label_img.file_path)[0] + "_processed.jpg"
    img.save(save_path, "JPEG")
    
    img.thumbnail((300, 300))
    img_display = ImageTk.PhotoImage(img)
    label_img.config(image=img_display)
    label_img.image = img_display
    
    messagebox.showinfo("Success", f"Image saved as {save_path}")

def exit_app():
    root.destroy()

# Initialize Tkinter
root = tk.Tk()
root.title("Image Processor")
root.geometry("400x500")

btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack(pady=10)

label_img = tk.Label(root)
label_img.pack()

label_status = tk.Label(root, text="No image selected", fg="red")
label_status.pack()

btn_upsample = tk.Button(root, text="Upsample", command=lambda: process_image("upsample"))
btn_upsample.pack(pady=5)

btn_downsample = tk.Button(root, text="Downsample", command=lambda: process_image("downsample"))
btn_downsample.pack(pady=5)

btn_exit = tk.Button(root, text="Exit", command=exit_app)
btn_exit.pack(pady=10)

root.mainloop()
