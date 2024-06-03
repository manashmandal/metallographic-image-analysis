from transformers import ViTForImageClassification
from preprocess import (
    train_test_split,
    label_to_index_map,
    index_to_label_map,
    labels,
    batch_generator_with_transform,
    ImageLabelTensor,
)
import torch

if __name__ in ("__main__", "predict"):
    model_path = "vit-metal"
    model = ViTForImageClassification.from_pretrained(model_path)
    train_test = train_test_split()

    data_generator = batch_generator_with_transform(data=train_test.test_data[:5])

    item: ImageLabelTensor
    for item in data_generator:
        print(item.label_index, index_to_label_map[item.label_index])
        print(torch.argmax(model(item.image_tensor.unsqueeze(0)).logits))
