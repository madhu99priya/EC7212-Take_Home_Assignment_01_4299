import cv2
import numpy as np
import os

def reduce_intensity_levels(image, levels):
    factor = 256 // levels
    reduced_image = (image // factor) * factor
    return reduced_image.astype(np.uint8)

def average_filter(image, ksize):
    return cv2.blur(image, (ksize, ksize))

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def block_average(image, block_size):
    h, w = image.shape[:2]
    output = np.zeros_like(image)
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            mean_val = block.mean(axis=(0, 1)).astype(np.uint8)
            output[y:y+block_size, x:x+block_size] = mean_val
    return output

def display_image(title, img):
    cv2.imshow(f" {title} ", img)
    print(f"📷 Showing: {title}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    path = input("🔍 Enter the path to the image file: ").strip().strip('"').strip("'")

    if not os.path.exists(path):
        print(f"❌ Error: Image not found at path: {path}")
        return

    image_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_color = cv2.imread(path)

    if image_gray is None or image_color is None:
        print("❌ Error: Could not load the image properly.")
        return

    # 1. Reduce intensity levels
    try:
        levels = int(input("🔢 Enter number of intensity levels (2, 4, 8, 16...): "))
        if levels not in [2, 4, 8, 16, 32, 64, 128, 256]:
            raise ValueError("Levels must be an integer power of 2 (up to 256)")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return

    reduced = reduce_intensity_levels(image_gray, levels)
    display_image(f"Intensity Reduced to {levels} Levels", reduced)

    # 2. Spatial averaging
    for size in [3, 10, 20]:
        avg = average_filter(image_gray, size)
        display_image(f"{size}x{size} Averaged Image", avg)

    # 3. Rotate by 45 and 90 degrees
    rot_45 = rotate_image(image_color, 45)
    display_image("Image Rotated 45°", rot_45)

    rot_90 = rotate_image(image_color, 90)
    display_image("Image Rotated 90°", rot_90)

    # 4. Block averaging
    for block_size in [3, 5, 7]:
        block = block_average(image_color, block_size)
        display_image(f"{block_size}x{block_size} Block Averaged Image", block)

    print("✅ All image processing tasks completed.")

if __name__ == "__main__":
    main()
