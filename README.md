# Learning to Steer: Input-dependent Steering for Multimodal LLMs

<div align="left">
<h3>
NeurIPS 2025
</h3>

<a href="https://jayneelparekh.github.io/learn-to-steer/"> <img alt="Blog Post" src="https://img.shields.io/badge/L2S-Project/Blog Page-810b81"> </a> [![arXiv](https://img.shields.io/badge/arXiv-2508.12815-b31b1b.svg)](https://arxiv.org/abs/2508.12815)

## Abstract/Overview
Steering has emerged as a practical approach to enable post-hoc guidance of LLMs towards enforcing a specific behavior. However, it remains largely underexplored for multimodal LLMs (MLLMs); furthermore, existing steering techniques, such as mean steering, rely on a single steering vector, applied independently of the input query. This paradigm faces limitations when the desired behavior is dependent on the example at hand. For example, a safe answer may consist in abstaining from answering when asked for an illegal activity, or may point to external resources or consultation with an expert when asked about medical advice. In this paper, we investigate a fine-grained steering that uses an input-specific linear shift. This shift is computed using contrastive input-specific prompting. However, the input-specific prompts required for this approach are not known at test time. Therefore, we propose to train a small auxiliary module to predict the input-specific steering vector. Our approach, dubbed as L2S (Learn-to-Steer), demonstrates that it reduces hallucinations and enforces safety in MLLMs, outperforming other static baselines.


## Installation

This repository is built on top of our [XL-VLMs](https://github.com/mshukor/xl-vlms) repository which contains code for our previous works on explainability and steering of multimodal LLMs. Feel free to refer to explore it if these works are also relevant for you.

### Environment 

### Datasets

### Models

Our experiments in the paper are on the following models:
* **LLaVA-v1.5-7b**
* **Qwen2-VL-7B-Instruct**

In general, owing to the parent [XL-VLMs](https://github.com/mshukor/xl-vlms) repo we support models from the `transformers` library.

## Main Experiments

### Feature Extraction

### Training L2S

### Inference 

### Evaluation




## Citations

If you find this repo useful, you can cite the work as follows:

```bibtex
@article{parekh2025learning,
  title={Learning to Steer: Input-dependent Steering for Multimodal LLMs},
  author={Parekh, Jayneel and Khayatan, Pegah and Shukor, Mustafa and Dapogny, Arnaud and Newson, Alasdair and Cord, Matthieu},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```
