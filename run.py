from preprocess import (
    train_test_split,
    batch_generator_with_transform,
    collate_fn,
    label_to_index_map,
    index_to_label_map,
    labels,
)
from datasets import Dataset, load_metric
from itertools import islice
from transformers import ViTForImageClassification
from torchvision import models
import torch
from transformers import TrainingArguments, Trainer
import numpy as np

model_name_or_path = "google/vit-base-patch16-224-in21k"

metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


if __name__ in ("__main__", "run"):
    train_test = train_test_split()

    # train_generator = batch_generator_with_transform(data=train_test.train_data[:5])
    # collated = collate_fn(list(train_generator))

    print(labels, len(labels), index_to_label_map, label_to_index_map)

    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=index_to_label_map,
        label2id=label_to_index_map,
    )
    model.to("cuda")

    training_args = TrainingArguments(
        output_dir="./vit-metal",
        per_device_train_batch_size=128,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_test.train_data,
        eval_dataset=train_test.test_data,
    )

    results = trainer.train()
    trainer.save_model("./vit-metal")
    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    print(trainer.evaluate())
