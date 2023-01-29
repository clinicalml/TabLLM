# TabLLM: Few-shot Classification of Tabular Data with Large Language Models

![Figure_Overview](https://user-images.githubusercontent.com/3011151/215147227-b384c811-71b2-44c3-8785-007dcb687575.jpg)

This repository contains the code to reproduce the results of the paper [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2210.10723) by Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag.

Reproducing the main results consists of three steps:

1. Creating textual serializations of the nine public tabular datasets
2. Train and evaluate TabLLM (use code from [t-few project](https://github.com/r-three/t-few)) on serialized datasets
3. Running the baseline models on the tabular datasets

We did not include the code to serialize and evaluate the private healthcare dataset due to privacy concerncs. Also, code for some additional experiments is not included. Feel free to contact us if you have any questions concerning these experiments.

## Preparing the Environment

We used conda to create a virtual environment using python 3.8:

```
conda create -n tabllm python==3.8
conda activate tabllm
```

Next, install the nececssary requirements.

```
conda isntall numpy scipy pandas scikit-learn
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install datasets transformerns sentencepiece protobuf-3.20.1 xgboost lightgbm tabpfn
```

## 1. Creating Serialized Datasets

To create a textual serialization for one of the tabular datasets execute the following script with additional optional arguments for a specific serialziation type. This will create a folder with a huggingface dataset in `datasets_serialized`:

```
create_external_datasets.py --dataset (car|income|diabetes|heart|bank|blood|calhousing|creditg|jungle) (--list) (--list (--tabletotext|--t0serialization|--values|--permuted|--shuffled))
```

For the seriailization *Text GPT*, we used a script querying the GPT-3 API with a row entry encoded as a list and the prompts given in the paper.

We provide the *Text* serializations in `datasets_serialized`. The other serializations are omnited here due to size constraints. The *Text* serialization achieved the best results in our experiments.

## 2. Train and Evaluate TabLLM on Serialized Datasets

We used the codebase of the [t-few project](https://github.com/r-three/t-few) for our experiments. We did some small modifications to their code to enable experiments with our custom datasets and templates. We included all changed files in the `t-few` folder. The script `few-shot-pretrained-100k.sh` runs all our TabLLM experiments for the different serializations. As a results an `exp_out` folder is created with the results. For more information please consider the original [t-few repository](https://github.com/r-three/t-few).

## 3. Running the Baseline Models

We tested TabLLM against several baselines. They use the standard non-serialized datasets. The hyperparameter ranges are given in the paper. You can specify the baseline models and datasets that you want to run in the code. To run a baseline model execute

```
evaluate_external_datasets.py
```

We hope these instructions help you to reproduce our results. Feel free to contact us if you have any questions!

## Citation

If you want to cite our work please use:

```
@article{hegselmann2022tabllm,
  title={TabLLM: Few-shot Classification of Tabular Data with Large Language Models},
  author={Hegselmann, Stefan and Buendia, Alejandro and Lang, Hunter and Agrawal, Monica and Jiang, Xiaoyi and Sontag, David},
  journal={arXiv preprint arXiv:2210.10723},
  year={2022}
}
```


We use the code of


```
@article{liu2022few,
  title={Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning},
  author={Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin},
  journal={arXiv preprint arXiv:2205.05638},
  year={2022}
}
```

```
@article{bach2022promptsource,
  title={Promptsource: An integrated development environment and repository for natural language prompts},
  author={Bach, Stephen H and Sanh, Victor and Yong, Zheng-Xin and Webson, Albert and Raffel, Colin and Nayak, Nihal V and Sharma, Abheesht and Kim, Taewoon and Bari, M Saiful and Fevry, Thibault and others},
  journal={arXiv preprint arXiv:2202.01279},
  year={2022}
}
```
