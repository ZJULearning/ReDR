# ReDR

Code for the ACL 2019 paper [Reinforced Dynamic Reasoning for Conversational Question Generation](https://www.aclweb.org/anthology/P19-1203). If you use this code as part of any published research, please cite the following paper. This code is based on the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).



## Introduction

This paper investigates a new task named Conversational Question Generation (CQG) which is to generate a question based on a passage and a conversation history (i.e., previous turns of question-answer pairs). Towards that end, we propose a new approach named Reinforced Dynamic Reasoning (ReDR) network, which is based on the general encoder-decoder framework but incorporates a reasoning procedure in a dynamic manner to better understand what has been asked and what to ask next about the passage.

## Required

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Running the Script

### 1. Download the dataset.

The dataset we used for training and testing our model is the Conversational Question Answering (CoQA) dataset [here](https://stanfordnlp.github.io/coqa/).

We download our data to a `data/` folder.

### 2. Preprocess the data.

```bash
python cqg_preprocess.py
```
This is to split the data into conversation histories, rationales, targets and answers. The processed data consists of parallel source (src) and target (tgt) data containing one sentence per line with tokens separated by a space.

```bash
python preprocess.py -train_src data/coqa-cqg-src.train.txt -train_history data/coqa-cqg-history.train.txt -train_ans data/coqa-cqg-ans.train.txt -train_tgt data/coqa-cqg-tgt.train.txt -valid_src data/coqa-cqg-src.dev.txt -valid_history data/coqa-cqg-history.dev.txt -valid_ans data/coqa-cqg-ans.dev.txt -valid_tgt data/coqa-cqg-tgt.dev.txt -save_data data/coqa-cqg --share_vocab --dynamic_dict  
```

This is to transform the data to indices, which means the system doesn't need to touch the words themselves. After running the preprocessing, the following files are generated:

* `coqa-cqg.train.0.pt`: serialized PyTorch file containing training data
* `coqa-cqg.valid.0.pt`: serialized PyTorch file containing validation data
* `coqa-cqg.vocab.pt`: serialized PyTorch file containing vocabulary data


### 3. Train and test the model of ReDR.
```
python train.py -data data/coqa-cqg -save_model out_model/
```
```
python generate.py  -src data/coqa-cqg-src.dev.txt -history data/coqa-cqg-history.dev.txt -tgt data/coqa-cqg-tgt.dev.txt -replace_unk -model out_model/_step_XXXXX.pt -output pred.txt
```

If you want to train on GPU, you need to set, as an example: CUDA_VISIBLE_DEVICES=1,3 -world_size 2 -gpu_ranks 0 1 to use (say) GPU 1 and 3 on this node only. 

## Reference

**"Reinforced Dynamic Reasoning for Conversational Question Generation"**
Boyuan Pan, Hao Li, Ziyu Yao, Deng Cai, Huan Sun. _ACL (2019)_ 

```
@inproceedings{pan2019reinforced,
  title={Reinforced Dynamic Reasoning for Conversational Question Generation},
  author={Pan, Boyuan and Li, Hao and Yao, Ziyu and Cai, Deng and Sun, Huan},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={2114--2124},
  year={2019}
}
```
