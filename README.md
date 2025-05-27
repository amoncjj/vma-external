# Vocabulary Matching Attack on Large Language Models Hidden States

This repository provides an implementation of the *vocabulary matching attack* described in our paper:

> **"Hidden No More: Attacking and Defending Private Third-Party LLM Inference"**  
> *Arka Pal, Rahul Thomas, Louai Zahran, Erica Choi, Akilesh Potti, Micah Goldblum, ICML 2025 Main Proceedings*  
> [OpenReview link](https://openreview.net/forum?id=QfD9P9IIoz)

The attack is able to decode input tokens from permutations of LLM hidden states with accuracies in excess of 99%, across different LLM model families, different choices of layer, and different permutation types. This demonstrates the insecurity of providing access to permutations of LLM hidden in e.g. MPC protocols in the open-weights setting.

Key components of the proposed attack include:

- Support for different permutation types applied to the hidden states.
- Multiple matching strategies using L1 distance metrics.
- Full prompt reconstruction from only hidden states — no access to the original input.
---

## Overview

As LLMs grow too large for self-hosting by the majority of end users, there has been an increase in the provision of third party hosting services. This, however, raises the risk of privacy violations and potential for curtailed usage and censorship. SMPC (Secure Multi-Party Computation) methods have recently been applied to LLM inference, using cryptographic protocols to provide a mathematically-guaranteed method of conducting privacy-preserving inference. However, SMPC methods add significant overhead, and are on the order of 1000x slower than plaintext inference.
Recent schemes such as [PermLLM](https://arxiv.org/abs/2405.18744), [Secure Transformer Inference Protocol](https://arxiv.org/abs/2312.00025), and [Centaur](https://arxiv.org/abs/2412.10652) propose to significantly reduce this overhead by avoiding expensive cryptographic operations for calculating the non-linearities in LLMs. Instead, they propose to perform standard non-linearity calculations, but on permutations of the hidden states, exploiting the permutation-equivariant properties of these non-linearities.

The security of these schemes is based on the assumption that permuted intermediate hidden states are secure against decoding back to the original input. We argue that this trust is misplaced, by constructing a highly successful reversal attack against these permuted hidden states.

### Our results
The results demonstrate that our attack consistently achieves near-perfect prompt recovery across both Gemma and Llama models—even under permutation defenses (sequence, hidden, and factorized-2D). Further details of the permutation types and accompanying results are provided in our paper.

| Layer | Unpermuted |           | Sequence-Dim      |           | Hidden-Dim        |           | Factorized-2D     |           |
|-------|------------------------|-----------|-------------------|-----------|-------------------|-----------|-------------------|-----------|
|       | **Gemma**              | **Llama** | **Gemma**         | **Llama** | **Gemma**         | **Llama** | **Gemma**         | **Llama** |
| 1     | 100%                   | 100%      | 100%              | 99.7%     | 100%              | 100%      | 99.9%             | 98.4%     |
| 6     | 100%                   | 100%      | 99.8%             | 100%      | 100%              | 98.5%     | 99.5%             | 97.8%     |
| 11    | 100%                   | 100%      | 100%              | 100%      | 100%              | 99.2%     | 99.5%             | 98.9%     |
| 16    | 100%                   | 100%      | 100%              | 100%      | 99.9%             | 99.4%     | 99.2%             | 98.8%     |
| 21    | 100%                   | 99.9%     | 99.8%             | 100%      | 98.2%             | 98.9%     | 99.1%             | 98.0%     |
| 26    | 100%                   | 99.7%     | 99.8%             | 100%      | 98.0%             | 98.2%     | 99.0%             | 97.6%     |

---

## Features

- Handles `None`, `S` (sequence), `D` (hidden dimension), and `SD` (both sequence and hidden dimension) permutations
- Multiple matching strategies
- Supports KV caching to speed up repeated inference (see Appendix A of paper)
- Supports proposal model LLM for token ordering, improving the attack efficiency (see Appendix A of paper)

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ritual-net/vma-external
cd vma-external
```

Make sure you have:

* `torch` (tested on 2.3.1)
* `transformers` (tested on transformers v4.45)
* `notebook` (optional)

```bash
pip install torch==2.3.1 transformers==4.45.0 notebook
```

---

## Usage

Launch the notebook:

```bash
jupyter notebook vocab_matching_attack.ipynb
```

Then call:

```python
run_vocab_matching_attack(
    sentence="This is a secret",
    layer=6,
    dist_funct='l1_sort_permuted',
    perm_type='SD'
)
```

Note that the distance function used in the matching step of the attack depends on the type of permutation used. The following tables mentions which matching functions work with which permutation type.

| Permutation Type | l1 | l1_permuted | l1_sort | l1_sort_permuted |
|-------------|-----|-----|------|----------|
| `None` | ✅ | ✅ | ✅ | ✅ |
| `S` | | ✅ |  | ✅ |
| `D` | | | ✅ | ✅ |
| `SD` | | | | ✅ |

For replication of the results in our paper, use the following matching functions for each permutation choice (note that `SD` corresponds to Factorized-2D in the paper):

`None`: l1

`S`: l1_permuted

`D`: l1_sort

`SD`: l1_sort_permuted

---

## Notebook Structure

The notebook contains the following key components:

| Function                | Description                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------- |
| `permute_states`        | Applies permutation to the hidden states (sequence, dimension, or both)                         |
| `l1_dist_matching*`     | Matching functions based on L1 distance under various permutation assumptions                   |
| `argmin_except`         | Utility function to find best match ignoring previously matched tokens                          |
| `vocab_matching_attack` | Main attack loop that reconstructs token sequence using vocab embeddings                        |
| `run_vocab_matching_attack` | Entrypoint function. Runs full pipeline: tokenization → hidden states → permutation → attack    |

---

## Example

```python
decoded_tokens = run_vocab_matching_attack(
    sentence="The password is qwerty",
    layer=8,
    dist_funct='l1_sort_permuted',
    perm_type='SD',
    matching_eps=22,
    next_token_proposal=True,
    use_cache=True
)
```
---

## Citation

If you use this work, please cite our ICML 2025 paper:

```bibtex
@inproceedings{
title={Hidden No More: Attacking and Defending Private Third-Party {LLM} Inference},
author={Arka Pal and Rahul Thomas and Louai Zahran and Erica Choi and Akilesh Potti and Micah Goldblum},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=QfD9P9IIoz}
}
```

---

## License

This project is licensed under the BSD-3 License.