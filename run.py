from preprocess import (
    train_test_split,
    batch_generator_with_transform,
    collate_fn,
    label_to_index_map,
    index_to_label_map,
    labels,
)
from datasets import Dataset
from itertools import islice
from transformers import ViTForImageClassification

model_name_or_path = "google/vit-base-patch16-224-in21k"


if __name__ in ("__main__", "run"):
    train_test = train_test_split()

    train_generator = batch_generator_with_transform(data=train_test.train_data[:1])
    collated = collate_fn(list(train_generator))

    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=index_to_label_map,
        label2id=label_to_index_map,
    )

    print(model(collated["pixel_values"]), return_tensors="pt")
