from pathlib import Path
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.data.dataset import load_dataset, CLASS_NAMES


DATASET_PATH = "data/raw/pcb-defect-dataset"
OUTPUT_DIR = Path("src/visualization")
NUM_IMAGES = 10


def main():

    print("Loading dataset...")

    dataset = load_dataset(DATASET_PATH, "train")

    print(f"Loaded {len(dataset)} defect instances.")

    image_paths = list(set(sample[0] for sample in dataset))

    print(f"Found {len(image_paths)} images.")

    random.seed(42)

    selected_images = random.sample(
        image_paths,
        min(NUM_IMAGES, len(image_paths))
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving visualizations to: {OUTPUT_DIR}")

    for index, image_path in enumerate(selected_images, start=1):

        print(f"Processing {index}/{len(selected_images)}: {image_path}")

        image = Image.open(image_path).convert("RGB")

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.imshow(image)

        image_annotations = [
            sample
            for sample in dataset
            if sample[0] == image_path
        ]

        for _, class_id, bbox in image_annotations:

            xmin, ymin, xmax, ymax = bbox

            width = xmax - xmin
            height = ymax - ymin

            rectangle = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )

            ax.add_patch(rectangle)

            ax.text(
                xmin,
                max(0, ymin - 5),
                CLASS_NAMES[class_id],
                fontsize=9,
                color="red",
                backgroundcolor="white",
            )

        ax.set_title(Path(image_path).name)
        ax.axis("off")

        output_path = OUTPUT_DIR / f"sample_{index}.png"

        plt.savefig(
            output_path,
            bbox_inches="tight",
            dpi=150,
        )

        plt.close(fig)

        print(f"Saved: {output_path}")

    print("\nDone!")
    print(f"Open the images inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()