import os
from PIL import Image
import math

def get_image_paths_in_dir(dataset_dir, extensions={'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}):
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        if any(part.startswith('.') for part in root.split(os.sep)):
            continue  # Skip hidden folders
        for file in files:
            if file.startswith('.'):
                continue  # Skip hidden files
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def compute_stats(image_paths):
    total_width = total_height = 0
    valid_images = 0

    portrait_ratios = []
    landscape_ratios = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                if width == 0 or height == 0:
                    continue
                total_width += width
                total_height += height
                valid_images += 1

                if height > width:
                    portrait_ratios.append(height / width)
                else:
                    landscape_ratios.append(width / height)

        except Exception as e:
            continue  # Skip corrupted images silently

    def mean_std(values):
        if not values:
            return 0, 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return mean, math.sqrt(variance)

    avg_width = total_width / valid_images if valid_images > 0 else 0
    avg_height = total_height / valid_images if valid_images > 0 else 0
    p_mean, p_std = mean_std(portrait_ratios)
    l_mean, l_std = mean_std(landscape_ratios)

    return (avg_width, avg_height, p_mean, p_std, l_mean, l_std, valid_images)

if __name__ == "__main__":
    parent_dir = "/usr/project/xtmp/rt195/Sentence_Embedding/F5/VLM2Vec3/vlm2vec_plus/VLM2Vec/MMEB-eval/eval_images"

    print("Dataset\tAvg_Width\tAvg_Height\tPortrait_Mean\tPortrait_Std\tLandscape_Mean\tLandscape_Std\tTotal_Images")

    for dataset_name in sorted(os.listdir(parent_dir)):
        if dataset_name.startswith('.'):
            continue  # Skip hidden dirs
        dataset_path = os.path.join(parent_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        image_paths = get_image_paths_in_dir(dataset_path)
        avg_w, avg_h, p_mean, p_std, l_mean, l_std, count = compute_stats(image_paths)

        print(f"{dataset_name}\t{avg_w:.2f}\t{avg_h:.2f}\t{p_mean:.2f}\t{p_std:.2f}\t{l_mean:.2f}\t{l_std:.2f}\t{count}")