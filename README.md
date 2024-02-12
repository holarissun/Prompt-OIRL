# Prompt-OIRL
Code for paper [Query-Dependent Prompt Evaluation and Optimization with Offline Inverse Reinforcement Learning](https://arxiv.org/pdf/2309.06553.pdf).
 
#### Prompt-OIRL has been accepted at ICLR'2024. We look forward to talking with you in Vienna!
#### Prompt-OIRL has been selected as an oral presentation at the ENLSP workshop at NeurIPS'2023.

### Code and Offline Data
- Code and Offline Data for experiments using LLaMA2-7B is now released!

## Introduction

### Abstract 

> In this study, we aim to enhance the arithmetic reasoning ability of Large Language Models (LLMs) through zero-shot prompt optimization. We identify a previously overlooked objective of query dependency in such optimization and elucidate two ensuing challenges that impede the successful and economical design of prompt optimization techniques. One primary issue is the absence of an effective method to evaluate prompts during inference when the golden answer is unavailable. Concurrently, learning via interactions with the LLMs to navigate the expansive natural language prompting space proves to be resource-intensive.

> To address this, we introduce Prompt-OIRL, which harnesses offline inverse reinforcement learning to draw insights from offline prompting demonstration data. Such data exists as by-products when diverse prompts are benchmarked on open-accessible datasets. With Prompt-OIRL, the query-dependent prompt optimization objective is achieved by first learning an offline reward model. This model can evaluate any query-prompt pairs without accessing LLMs. Subsequently, a best-of-N strategy is deployed to recommend the optimal prompt. Our experimental evaluations across various LLM scales and arithmetic reasoning datasets underscore both the efficacy and economic viability of the proposed approach.

### Paper Preview

![Image](Prompt_OIRL_preview.png)

## Reproduction

### Code and Offline Data
- Code and Offline Data for experiments using LLaMA2-7B is now released!

To reproduce our results (e.g., using LLaMA2)

1. get license to use LLaMA-2 from https://ai.meta.com/llama/

2. get access to the SVAMP dataset: https://github.com/arkilpatel/SVAMP

3. get access to the GSM8K dataset: https://huggingface.co/datasets/gsm8k

4. run the code: from step 1 - step 5 to generate-, reorganize-, process- data, and then perform reward modeling (offline evaluation) and optimization.


## A Related Discussion on RLHF:
Prompt-OIRL addresses the prompting problems in LLMs using an RLAIF approach. For readers who are also interested in RLHF and RLAIF, and on the intersection between RL and LLM research, we would refer to our related perspective paper discussing RL in LLM research:
[RL in the Era of LLMs: What is Essential? What is Needed? RLHF, Prompting, and Beyond.](https://arxiv.org/pdf/2310.06147.pdf)





## TODOs
- [ ] code release for TigerBot-13B and GPT3.5-turbo
- [ ] re-organizing of the codebase


## BibTex Citation
If you would like to cite our code or paper, please use

```
@article{sun2023query,
  title={Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL},
  author={Sun, Hao and H{\"u}y{\"u}k, Alihan and van der Schaar, Mihaela},
  journal={arXiv e-prints},
  pages={arXiv--2309},
  year={2023}
}

@article{sun2023reinforcement,
  title={Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond},
  author={Sun, Hao},
  journal={arXiv preprint arXiv:2310.06147},
  year={2023}
}
