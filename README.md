# TabLLM: Few-shot Classification of Tabular Data with Large Language Models

![Figure_Overview](https://user-images.githubusercontent.com/3011151/215147227-b384c811-71b2-44c3-8785-007dcb687575.jpg)

This repository contains the code to reproduce the results of the paper [TabLLM: Few-shot Classification of Tabular Data with Large Language Models](https://arxiv.org/abs/2210.10723) by Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag.

## Update 10/30/2023: We Added Additional Instruction to the Readme

Since several issues were raised regarding the code, we decided to add some additional instructions to the readme. We now provide all steps to reproduce an entry of our final results table. Reproducing the remaining results mainly consists of changing the experimental parameters. Thanks for everyone who provided feedback!

## Overview

Reproducing the main results consists of three steps:

1. Creating textual serializations of the nine public tabular datasets
2. Train and evaluate TabLLM (use code from [t-few project](https://github.com/r-three/t-few)) on serialized datasets
3. Running the baseline models on the tabular datasets

We did not include the code to serialize and evaluate the private healthcare dataset due to privacy concerns. Also, code for some additional experiments is not included. Feel free to contact us if you have any questions concerning these experiments.

## Setting the Correct Paths

TabLLM and the [t-few project](https://github.com/r-three/t-few) use the path `/root/<project>` by default and we will assume that you cloned both repositories to this location for this readme, i.e., `/root/TabLLM` for TabLLM and `/root/t-few` for the [t-few](https://github.com/r-three/t-few) repository. It is very likely that you have to adapt those paths for your own setup. The easiest way is to replace all occurrences of `/root` with your own path. When you get an error running the code, please ensure that you set all paths correctly.


## Preparing the Environments

We used conda to create the necessary virtual environments. For the TabLLM environment, we used python 3.8:

```
conda create -n tabllm python==3.8
conda activate tabllm
```

Next, install the necessary requirements for TabLLM.


```
conda install numpy scipy pandas scikit-learn
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install datasets transformers sentencepiece protobuf xgboost lightgbm tabpfn
```


If you want to run training and inference for TabLLM, you also have to setup the environment for [t-few](https://github.com/r-three/t-few). You can follow their readme to setup the environment. We had some dependency issues when following their instructions. Here are the commands that worked for us (taken and adapted from their instructions):

```
conda create -n tfew python==3.7
conda activate tfew
pip install fsspec==2021.05.0
pip install --use-deprecated=legacy-resolver  -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install urllib3==1.26.6
pip install importlib-metadata==4.13.0
pip install scikit-learn
```


To ensure that the t-few project ist setup correctly, you can run the command given in their [repository](https://github.com/r-three/t-few):

```
export HF_HOME=~/.cache/huggingface
CUDA_VISIBLE_DEVICES=0 python -m src.pl_train -c t03b.json+rte.json -k save_model=False exp_name=first_exp
```


The result of the experiment should be stored in `/root/t-few/exp_out/first_exp`.

## 1. Creating Serialized Datasets

To create a textual serialization for one of the tabular datasets execute the following script with additional optional arguments for a specific serialization type. This will create a folder with a huggingface dataset in `datasets_serialized`:

```
create_external_datasets.py --dataset (car|income|diabetes|heart|bank|blood|calhousing|creditg|jungle) (--list) (--list (--tabletotext|--t0serialization|--values|--permuted|--shuffled))
```

For the serialization *Text GPT*, we used a script querying the GPT-3 API with a row entry encoded as a list and the prompts given in the paper.

We provide the *Text* serializations in `datasets_serialized`. The other serializations are omitted here due to size constraints. The *Text* serialization achieved the best results in our experiments.

## 2. Train and Evaluate TabLLM on Serialized Datasets

We used the codebase of the [t-few project](https://github.com/r-three/t-few) for our experiments. We made some small modifications to their code to enable experiments with our custom datasets and templates. We included all changed files in the `t-few` folder and you have to copy them over.

```
cp /root/TabLLM/t-few/bin/few-shot-pretrained-100k.sh  /root/t-few/bin/
cp /root/TabLLM/t-few/configs/* /root/t-few/configs/
cp /root/TabLLM/t-few/src/models/EncoderDecoder.py /root/t-few/src/models/
cp /root/TabLLM/t-few/src/data/* /root/t-few/src/data/
cp /root/TabLLM/t-few/src/scripts/get_result_table.py /root/t-few/src/scripts/
```

Please, check that you also set the paths correctly for the t-few project. In particular, you should check `/root/t-few/src/data/dataset_readers.py` to ensure that `DATASETS_OFFLINE` in line 75 points to `/root/TabLLM/datasets_serialized` and `yaml_dict = yaml.load(open(...))` in line 233 uses the path `/root/TabLLM/templates/templates_`.

The script `/root/t-few/bin/few-shot-pretrained-100k.sh` runs all our TabLLM experiments for the different serializations and stores them in `/root/t-few/exp_out`. To run the 4-shot heart experiment with the *Text* serialization using the T0-3B model, set the for-loops going over the different experimental settings in `/root/t-few/bin/few-shot-pretrained-100k.sh` to:

```
for model in 't03b'
do
  [...]
  for num_shot in 4
  do
    [...]
    for dataset in heart 
    do
      [...]
      for seed in 42 1024 0 1 32  # Keep this for-loop as it is
      do
        [...]
      done
    done
  done
done
```

Then, you can run the specified setup from the t-few folder `/root/t-few` via:

```
./bin/few-shot-pretrained-100k.sh
```

The result of the experiment should be stored in `/root/t-few/exp_out/t03b_heart_numshot4_seed*`. Note that we use no validation set, hence, in the code our test data is treated as validation (=pred) set. As a consequence, you can find the test performance for seed 42 in `/root/t-few/exp_out/t03b_heart_numshot4_seed42_ia3_pretrained100k/dev_scores.json`:

```
cat /root/t-few/exp_out/t03b_heart_numshot4_seed42_ia3_pretrained100k/dev_scores.json
{"AUC": 0.617825311942959, "PR": 0.6409831261754565, "micro_f1": 0.5869565217391305, "macro_f1": 0.5511042629686697, "accuracy": 0.5869565217391305, "num": 184, "num_steps": -1, "score_gt": 0.8486858865489131, "score_cand": 0.9136485224184783}
```

To collect the results of several runs, we slightly changed the `/root/t-few/src/scripts/get_result_table.py` script to report the mean AUC and standard deviation. For the above example, using the script looks as follows:

```
python /root/t-few/src/scripts/get_result_table.py -e t03b* -d heart
================================================================================
Find 5 experiments fit into t03b*
heart: 67.65 (12.87)
Save result to exp_out/summary.csv
```

This results corresponds to the entry "TabLLM (T0 3B + Text Template)" for the heart dataset for 4 training examples (shots) on page 21 in our [paper](https://arxiv.org/abs/2210.10723). To obtain the other experiments you have to adapt `/root/t-few/bin/few-shot-pretrained-100k.sh` accordingly. For more information, please also consider the original [t-few repository](https://github.com/r-three/t-few) or raise an issue.


## 3. Running the Baseline Models

We tested TabLLM against several baselines. They use the standard non-serialized datasets. The hyperparameter ranges are given in the paper. You can specify the baseline models and datasets that you want to run in the code. To run a baseline model execute

```
evaluate_external_datasets.py
```

We hope these instructions help you to reproduce our results. Feel free to contact us if you have any questions!

## Citation

If you want to cite our work please use:

```
@inproceedings{hegselmann2023tabllm,
  title={Tabllm: Few-shot classification of tabular data with large language models},
  author={Hegselmann, Stefan and Buendia, Alejandro and Lang, Hunter and Agrawal, Monica and Jiang, Xiaoyi and Sontag, David},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={5549--5581},
  year={2023},
  organization={PMLR}
}
```


We use the code of


```
@article{liu2022few,
  title={Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning},
  author={Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin A},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={1950--1965},
  year={2022}
}
```

```
@inproceedings{bach2022promptsource,
  title={PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts},
  author={Bach, Stephen and Sanh, Victor and Yong, Zheng Xin and Webson, Albert and Raffel, Colin and Nayak, Nihal V and Sharma, Abheesht and Kim, Taewoon and Bari, M Saiful and F{\'e}vry, Thibault and others},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  pages={93--104},
  year={2022}
}
```
