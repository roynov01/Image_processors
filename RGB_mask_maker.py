import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import functools

class ImageMaskApp:
    def __init__(self, master, img_path, num_masks=3):
        self.master = master
        self.master.title("Image Masking App")

        self.frame = tk.Frame(self.master)
        self.frame.pack(side=tk.LEFT)

        self.image = cv2.imread(img_path)
        self.masks = [np.zeros_like(self.image) for _ in range(num_masks)]
        self.drawing = False
        self.current_mask = 0  # Start with the first mask

        self.image_for_tk = self.convert_cv_to_tk(self.image)
        self.canvas = tk.Canvas(self.frame, width=self.image.shape[1], height=self.image.shape[0])
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_for_tk)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Create buttons for each mask
        for i in range(num_masks):
            button = tk.Button(master, text=f"Draw Mask {i + 1}", command=functools.partial(self.select_mask, i))
            button.pack(side=tk.TOP)
        self.stop_btn = tk.Button(master, text="Close", command=self.stop)
        self.stop_btn.pack(side=tk.TOP)

    def stop(self):
        cv2.imshow("results", self.combined)
        self.master.destroy()



    def convert_cv_to_tk(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def on_button_press(self, event):
        if self.drawing:
            self.pts = [(event.x, event.y)]

    def on_move_press(self, event):
        if self.drawing:
            self.pts.append((event.x, event.y))
            self.redraw()

    def on_button_release(self, event):
        if self.drawing:
            self.pts.append((event.x, event.y))
            mask = self.masks[self.current_mask]
            cv2.fillPoly(mask, [np.array(self.pts)], (255, 255, 255))
            self.redraw()

    def select_mask(self, mask_number):
        self.current_mask = mask_number
        self.drawing = True

    def redraw(self):
        combined = self.image.copy()
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Define more colors as needed

        for i, mask in enumerate(self.masks):
            color = colors[i % len(colors)]  # Cycle through colors
            mask_colored = np.zeros_like(self.image)
            mask_colored[mask[:, :, 0] > 0] = color
            combined = cv2.addWeighted(combined, 1, mask_colored, 0.3, 0)
        self.combined = combined
        self.image_for_tk = self.convert_cv_to_tk(combined)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_for_tk)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageMaskApp(root, 'data/input/test.jpg')
    root.mainloop()
