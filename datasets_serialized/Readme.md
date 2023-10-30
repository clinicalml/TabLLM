# Datasets Serialized

Here we provide HuggingFace datasets for the public datasets and the *Text* serialization.

You can load and inspect these datasets with:

```
from datasets import load_from_disk
dataset = load_from_disk('/root/TabLLM/datasets_serialized/heart')
dataset['note'][0]
dataset['label'][0]
```
