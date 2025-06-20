import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_image():
    """Get image from user input"""
    while True:
        path = input("Enter image path: ").strip().strip('"').strip("'")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print("Image loaded Successfully!")
            return img, path
        print("Invalid path. Please enter an valid path.")

def get_intensity_levels():
    """Get intensity levels from user"""
    while True:
        try:
            levels = int(input("Enter intensity levels (power of 2: 2,4,8,16,32,64,128): "))
            if levels > 0 and (levels & (levels - 1)) == 0 and levels <= 128:
                return levels
            print("Must be power of 2 and ≤ 128")
        except ValueError:
            print("Enter valid integer")

def show_images(images, titles):
    """Display images in smaller windows"""
    num_images = len(images)
    
    if num_images <= 2:
        figsize = (8, 3)
    elif num_images <= 4:
        figsize = (10, 3)
    else:
        figsize = (12, 3)
    
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title, fontsize=9) 
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 1. Intensity Level Reduction
def reduce_intensity(img, levels):
    """Reduce intensity levels"""
    factor = 256 // levels
    return ((img // factor) * factor).astype(np.uint8)

# 2. Spatial Averaging
def spatial_average(img, size):
    """Apply spatial averaging"""
    kernel = np.ones((size, size)) / (size * size)
    return cv2.filter2D(img, -1, kernel).astype(np.uint8)

# 3. Image Rotation
def rotate_image(img, angle):
    """Rotate image"""
    h, w = img.shape
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos, sin = np.abs(M[0,0]), np.abs(M[0,1])
    new_w, new_h = int(h*sin + w*cos), int(h*cos + w*sin)
    
    M[0,2] += new_w/2 - center[0]
    M[1,2] += new_h/2 - center[1]
    
    return cv2.warpAffine(img, M, (new_w, new_h))

# 4. Block Resolution Reduction
def block_reduce(img, block_size):
    """Reduce resolution using blocks"""
    h, w = img.shape
    new_h, new_w = (h//block_size)*block_size, (w//block_size)*block_size
    
    cropped = img[:new_h, :new_w]
    blocks = cropped.reshape(new_h//block_size, block_size, new_w//block_size, block_size)
    averages = blocks.mean(axis=(1,3))
    
    return np.repeat(np.repeat(averages, block_size, axis=0), block_size, axis=1).astype(np.uint8)

def main():
    """Main function"""
    print("Computer Vision Assignment 1 - EG/2020/4299")
    print("="*43)
    
    img,path = get_image()
    
    # 1. Intensity Reduction
    print("\n1. INTENSITY REDUCTION")
    levels = get_intensity_levels()
    reduced = reduce_intensity(img, levels)
    show_images([img, reduced], ['Original', f'{levels} levels'])
    
    # 2. Spatial Averaging
    print("\n2. SPATIAL AVERAGING")
    sizes = [3, 10, 20]
    images = [img]
    titles = ['Original']
    
    for size in sizes:
        avg = spatial_average(img, size)
        images.append(avg)
        titles.append(f'{size}x{size}')
    
    show_images(images, titles)
    
    # 3. Rotation
    print("\n3. ROTATION")
    angles = [45, 90]
    images = [img]
    titles = ['Original']
    
    for angle in angles:
        rotated = rotate_image(img, angle)
        images.append(rotated)
        titles.append(f'{angle}°')
    
    show_images(images, titles)
    
    # 4. Block Reduction
    print("\n4. BLOCK REDUCTION")
    blocks = [3, 5, 7]
    images = [img]
    titles = ['Original']
    
    for block in blocks:
        reduced = block_reduce(img, block)
        images.append(reduced)
        titles.append(f'{block}x{block}')
    
    show_images(images, titles)
    
    print("\nAll operations completed!")

if __name__ == "__main__":
    main()