import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
input_dir  = r"C:path"
output_dir = r"C:path"
os.makedirs(output_dir, exist_ok=True)
THRESH_VAL = 200
DILATE_KERNEL = (5, 5)
DILATE_ITER  = 2

INPAINT_RADIUS = 3
INPAINT_METHOD = cv2.INPAINT_TELEA


def is_image_file(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def remove_text(img):
    """Detect bright text and remove with inpainting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY)

    kernel = np.ones(DILATE_KERNEL, np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=DILATE_ITER)

    cleaned = cv2.inpaint(img, mask, INPAINT_RADIUS, INPAINT_METHOD)
    return cleaned, mask


def preview_one(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    cleaned, mask = remove_text(img)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Detected Text Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    plt.title("Cleaned (No Text)")
    plt.axis("off")

    plt.show()


def process_all():
    files = [f for f in os.listdir(input_dir) if is_image_file(f)]
    print(f"Found {len(files)} images.")
    for f in tqdm(files, desc="Cleaning"):
        in_path = os.path.join(input_dir, f)
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cleaned, _ = remove_text(img)


        out_path = os.path.join(output_dir, os.path.splitext(f)[0] + ".png")
        cv2.imwrite(out_path, cleaned)
    print("Done! Clean images saved to:", output_dir)


if __name__ == "__main__":

    process_all()

