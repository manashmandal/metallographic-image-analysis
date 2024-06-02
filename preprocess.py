from transformers import ViTImageProcessor
import os
from torchvision.transforms import v2
from torchvision.io import read_image
from pathlib import Path
from torch import Tensor
from PIL import Image
import io
from torchvision.transforms import ToTensor, Resize
import torch
from typing import Generator, NamedTuple
from random import shuffle
from datasets import Dataset
from multiprocessing import cpu_count

load_model = False

if cpu_count() == 24:
    dataset_path = "/mnt/i/amit-dataset/Microstructure Dataset (5 Classes)"
else:
    dataset_path = "./dataset"


class ImageWithLabel(NamedTuple):
    image_path: str
    label_index: int
    label: str


class TrainTestSplit(NamedTuple):
    train_data: list[ImageWithLabel]
    test_data: list[ImageWithLabel]


class ImageLabelTensor(NamedTuple):
    image_tensor: Tensor
    label_index: int


def read_bmp_as_tensor(bmp_path: str) -> Tensor:
    with Image.open(bmp_path) as img:
        img = img.convert("RGB")  # Ensure the image is in RGB format
        tensor_image = ToTensor()(img)  # Convert the image to a tensor
        return tensor_image


transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if load_model:
    model_name_or_path = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

labels = os.listdir(dataset_path)
label_to_dir_map = {label: [] for label in labels}
label_to_index_map = {label: i for i, label in enumerate(labels)}
index_to_label_map = {index: labels[index] for index in range(len(labels))}


def get_all_images_with_labels() -> list[ImageWithLabel]:
    all_images_with_labels = []
    for label in labels:
        label_to_dir_map[label] = list(Path(f"{dataset_path}/{label}").rglob("*.bmp"))

    for label, image_path in label_to_dir_map.items():
        for path in image_path:
            all_images_with_labels.append(
                ImageWithLabel(path, label_to_index_map[label], label)
            )

    return all_images_with_labels


def train_test_split(split_ratio: float = 0.8) -> TrainTestSplit:
    all_images_with_labels = get_all_images_with_labels()
    shuffle(all_images_with_labels)
    split_index = int(len(all_images_with_labels) * split_ratio)
    train_data = all_images_with_labels[:split_index]
    test_data = all_images_with_labels[split_index:]
    return TrainTestSplit(train_data, test_data)


def batch_generator_with_transform(
    data: list[ImageWithLabel],
) -> Generator[list[ImageWithLabel], None, None]:
    for item in data:
        yield ImageLabelTensor(
            image_tensor=transforms(read_bmp_as_tensor(item.image_path)),
            label_index=item.label_index,
        )


def collate_fn(batch: list[ImageWithLabel]) -> dict[str, Tensor]:
    batch = list(batch_generator_with_transform(batch))
    images = torch.stack([item.image_tensor for item in batch])
    labels = torch.tensor([item.label_index for item in batch])
    return {"pixel_values": images, "labels": labels}
