# 🪄 Prompt-OIRL: Learning to Prompt LLMs with Known Magic Prompts

### 💻 Implementation and 📒 tutorial for ICLR 2024 paper 

 ![Image](prompt-oirl-title.png)

- [Paper Link](https://arxiv.org/pdf/2309.06553.pdf)
- [Open Review Link](https://openreview.net/forum?id=N6o0ZtPzTg)

 
#### 🔥 News
- (2024.2) Code with GPT3.5 and TigerBot has been released.
- (2024.1) Prompt-OIRL has been accepted by ICLR'2024. We look forward to talking with you in Vienna!
- (2024.12) Prompt-OIRL has been presented at the NeurIPS conference. Thanks for all the invaluable feedback!
- (2023.10) Code with llama2 has been released.
- (2023.10) Prompt-OIRL has been featured in a positioning [paper](https://arxiv.org/pdf/2310.06147.pdf) as an example of **inverse alignment**.
- (2023.9) Prompt-OIRL has been selected as an **oral presentation** at the ENLSP workshop at NeurIPS'2023.

## Abstract

> In this study, we aim to enhance the arithmetic reasoning ability of Large Language Models (LLMs) through zero-shot prompt optimization. We identify a previously overlooked objective of query dependency in such optimization and elucidate two ensuing challenges that impede the successful and economical design of prompt optimization techniques. One primary issue is the absence of an effective method to evaluate prompts during inference when the golden answer is unavailable. Concurrently, learning via interactions with the LLMs to navigate the expansive natural language prompting space proves to be resource-intensive.
To address this, we introduce Prompt-OIRL, which harnesses offline inverse reinforcement learning to draw insights from offline prompting demonstration data. Such data exists as by-products when diverse prompts are benchmarked on open-accessible datasets. With Prompt-OIRL, the query-dependent prompt optimization objective is achieved by first learning an offline reward model. This model can evaluate any query-prompt pairs without accessing LLMs. Subsequently, a best-of-N strategy is deployed to recommend the optimal prompt. Our experimental evaluations across various LLM scales and arithmetic reasoning datasets underscore both the efficacy and economic viability of the proposed approach.

## Motivating Example

![Image](motivatingexample.png)
Figure 1. **No prompt is perfect that works for all queries**. The optimal prompt is query-dependent. Yet the seeking of such prompts can be costly and inefficient. 
    Prompt-OIRL optimizes prompt during inference time on a **query-dependent** level effectively and cost-efficiently.
(original chat logs with GPT4 for those motivating examples can be found at [Left](https://chat.openai.com/share/0f2d11b1-322a-4c47-a877-ad6fbace8179), [Right](https://chat.openai.com/share/15870a47-93c7-4b98-96c8-af0516c0c999))

## ⚙️ Reproduction

### Code and Offline Data
- Code and Offline Data for experiments using LLaMA2-7B is now released!

To reproduce our results (e.g., using LLaMA2)

1. get license to use LLaMA-2 from https://ai.meta.com/llama/

2. get access to the SVAMP dataset: https://github.com/arkilpatel/SVAMP

3. get access to the GSM8K dataset: https://huggingface.co/datasets/gsm8k

4. run the code: from step 1 - step 5 to generate-, reorganize-, process- data, and then perform reward modeling (offline evaluation) and optimization.


## 🚀 A Related Discussion on RLHF:
Prompt-OIRL addresses the prompting problems in LLMs using an RLAIF approach. For readers who are also interested in RLHF and RLAIF, and on the intersection between RL and LLM research, we would refer to our related perspective paper discussing RL in LLM research:
[RL in the Era of LLMs: What is Essential? What is Needed? RLHF, Prompting, and Beyond.](https://arxiv.org/pdf/2310.06147.pdf)




## 📚 BibTex Citation
If you would like to cite our code or paper, please use

```
@inproceedings{sun2023query,
  title={Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL},
  author={Sun, Hao and H{\"u}y{\"u}k, Alihan and van der Schaar, Mihaela},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}


@article{sun2023reinforcement,
  title={Reinforcement Learning in the Era of LLMs: What is Essential? What is needed? An RL Perspective on RLHF, Prompting, and Beyond},
  author={Sun, Hao},
  journal={arXiv preprint arXiv:2310.06147},
  year={2023}
}
