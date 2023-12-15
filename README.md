# NLP Fine Tuning Model of Llama2 on Google Colab.

## 1. Introduction

High-quality data is fundamental to produce a good model. The higher the quality, the better the model. In this notebook, we will create a dataset for fine-tuning LLMs.

![](https://i.imgur.com/IDNhAWH.png)

There are different types of datasets we can use to fine-tune LLMs:

1. **Instruction datasets**: inputs are instructions (e.g., questions) and outputs correspond to the expected responses (e.g., answers). _Example: Open-Orca._
2. **Raw completion**: this is a continuation of the pre-training objective (next token prediction). In this case, the trained model is not designed to be used as an assistant. _Example: MADLAD-400._
3. **Preference datasets**: these datasets are used with reinforcement learning to rank candidate responses. They can provide multiple answers to the same instruction, and help the model to select the best response. _Example: ultrafeedback_binarized._
4. **Others**: a fill-in-the-middle objective is very popular with code completion models (e.g., Codex from GitHub Copilot). Other datasets can be designed for classification, where the outputs correspond to the labels we want to predict (the model requires an additional classfication head in this case).

In practice, supervised fine-tuning only leverages the first type of dataset (Instruction Dataset). We can either **create our own** instruction dataset or **modify an existing one** to filter, improve, or enrich it.

# Fine-tune Llama 2 in Google Colab

### Note: This has been tested on free-tier Google Colab (T4 GPU)

## 1. Introduction

Base models like Llama 2 can **predict the next token** in a sequence. However, this does not make them particularly useful assistants since they don't reply to instructions. This is why we employ instruction tuning to align their answers with what humans expect. There are two main fine-tuning techniques:

- **Supervised Fine-Tuning** (SFT): Models are trained on a dataset of instructions and responses. It adjusts the weights in the LLM to minimize the difference between the generated answers and ground-truth responses, acting as labels.

- **Reinforcement Learning from Human Feedback** (RLHF): Models learn by interacting with their environment and receiving feedback. They are trained to maximize a reward signal (using [PPO](https://arxiv.org/abs/1707.06347)), which is often derived from human evaluations of model outputs.

In general, RLHF is shown to capture **more complex and nuanced** human preferences, but is also more challenging to implement effectively. Indeed, it requires careful design of the reward system and can be sensitive to the quality and consistency of human feedback. A possible alternative in the future is the [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO) algorithm, which directly runs preference learning on the SFT model.

In this project, we will perform SFT, but this raises a question: why does fine-tuning work in the first place? As highlighted in the [Orca paper](https://mlabonne.github.io/blog/notes/Large%20Language%20Models/orca.html), our understanding is that fine-tuning **leverages knowledge learned during the pretraining** process. In other words, fine-tuning will be of little help if the model has never seen the kind of data you're interested in. However, if that's the case, SFT can be extremely performant.

For example, the [LIMA paper](https://mlabonne.github.io/blog/notes/Large%20Language%20Models/lima.html) showed how you could outperform GPT-3 (DaVinci003) by fine-tuning a LLaMA (v1) model with 65 billion parameters on only 1,000 high-quality samples. The **quality of the instruction dataset is essential** to reach this level of performance, which is why a lot of work is focused on this issue (like [evol-instruct](https://arxiv.org/abs/2304.12244), Orca, or [phi-1](https://mlabonne.github.io/blog/notes/Large%20Language%20Models/phi1.html)). Note that the size of the LLM (65b, not 13b or 7b) is also fundamental to leverage pre-existing knowledge efficiently.

[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

## Fine-tuning Llama 2 model Options

Three options for supervised fine-tuning: full fine-tuning, [LoRA](https://arxiv.org/abs/2106.09685), and [QLoRA](https://arxiv.org/abs/2305.14314).

![](https://i.imgur.com/7pu5zUe.png)

In this section, we will fine-tune a Llama 2 model with 7 billion parameters on a T4 GPU with high RAM using Google Colab (2.21 credits/hour). Note that a T4 only has 16 GB of VRAM, which is barely enough to **store Llama 2-7b's weights** (7b × 2 bytes = 14 GB in FP16). In addition, we need to consider the overhead due to optimizer states, gradients, and forward activations (see [this excellent article](https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-models-memory) for more information).

To drastically reduce the VRAM usage, we must **fine-tune the model in 4-bit precision**, which is why we'll use QLoRA here. We can lose a bit of precision using QLoRA, however it is not that bad since we just train using T4 google colab.

NOTE: You can fine-tune a MINIMALISTIC implementation of LoRA called nanoLoRA with guidelines in [HERE](https://colab.research.google.com/drive/1QG1ONI3PfxCO2Zcs8eiZmsDbWPl4SftZ).

## HOW TO USE IT

### STEP 1:

Load the notebook in your google colab.

### STEP 2:

Configure your google colab with T4 GPU or higher GPU if available.

### STEP 3:

Google colab support for adding secret key.

Set your secret key with:

```
name: huggingface
key: (copy from your huggingface API profile)
```

### STEP 4:

Run the respective notebook.
