from preprocess import train_test_split, batch_generator_with_transform, collate_fn
from datasets import Dataset
from itertools import islice

if __name__ in ("__main__", "run"):
    train_test = train_test_split()

    train_generator = batch_generator_with_transform(data=train_test.train_data)
    sliced = islice(train_generator, 0, 5)
    collated = collate_fn(list(sliced))

    print(collated)
