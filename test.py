import os
import random
import cv2
import matplotlib.pyplot as plt

LABELS_DIR = 'dataset/labels'
IMAGES_DIR = 'dataset/images'
NUM_IMAGES = 10

def get_image_path(label_filename):
    # Replace .txt with .jpg or .png as needed
    base = os.path.splitext(label_filename)[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(IMAGES_DIR, base + ext)
        if os.path.exists(img_path):
            return img_path
    return None

def draw_boxes(image, label_path):
    h, w = image.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, bw, bh = map(float, parts)
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    return image

label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
sampled_labels = random.sample(label_files, min(NUM_IMAGES, len(label_files)))

plt.figure(figsize=(15, 10))
for idx, label_file in enumerate(sampled_labels):
    img_path = get_image_path(label_file)
    if img_path is None:
        continue
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = draw_boxes(img, os.path.join(LABELS_DIR, label_file))
    plt.subplot(2, 5, idx+1)
    plt.imshow(img)
    plt.title(label_file)
    plt.axis('off')

plt.tight_layout()
plt.show()