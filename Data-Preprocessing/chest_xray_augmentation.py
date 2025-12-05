import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageEnhance

input_dir  = Path(r"")
output_dir = Path(r"")
output_dir.mkdir(parents=True, exist_ok=True)

def flip_horizontal(image):
    return cv2.flip(image, 1)

def rotate_light(image):
    angle = random.uniform(-8, 8)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h))

def adjust_brightness(image):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    factor = random.uniform(0.9, 1.2)
    return np.array(enhancer.enhance(factor))

def adjust_contrast(image):
    pil_img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    factor = random.uniform(0.9, 1.2)
    return np.array(enhancer.enhance(factor))

def add_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    noisy = image.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def zoom_image(image):
    scale = random.uniform(1.0, 1.1)
    h, w = image.shape[:2]
    zoomed = cv2.resize(image, None, fx=scale, fy=scale)
    zh, zw = zoomed.shape[:2]
    start_x = max(0, (zw - w) // 2)
    start_y = max(0, (zh - h) // 2)
    cropped = zoomed[start_y:start_y + h, start_x:start_x + w]
    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = cv2.resize(cropped, (w, h))
    return cropped

safe_augmentations = [
    flip_horizontal,
    rotate_light,
    adjust_brightness,
    adjust_contrast,
    add_noise,
    zoom_image
]

def apply_safe_augmentation(image):
    aug_fn = random.choice(safe_augmentations)
    return aug_fn(image)

exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
image_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]

print("Input dir:", input_dir)
print("Exists?:", input_dir.exists())
print(f"Total images found: {len(image_files)}")

if len(image_files) == 0:
    raise SystemExit(
        f"No images found in: {input_dir}\n"
        "Check: correct folder, images in subfolders, or extensions like .jpeg/.JPG."
    )

target_count = 4477
current_count = len(image_files)
augmentation_needed = max(0, target_count - current_count)

print(f"Augmentation needed: {augmentation_needed}")

for i in tqdm(range(augmentation_needed)):
    img_path = random.choice(image_files)
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aug_img = apply_safe_augmentation(img)
    aug_img = cv2.resize(aug_img, (800, 800))
    pil_img = Image.fromarray(aug_img)
    save_path = output_dir / f"aug_{i}.jpg"
    quality = 95
    while True:
        pil_img.save(save_path, "JPEG", quality=quality, dpi=(120, 120))
        if os.path.getsize(save_path) <= 100 * 1024 or quality <= 60:
            break
        quality -= 5

print(" Chest X-ray Augmentation Completed Successfully!")
