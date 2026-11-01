# PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2510.22264)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Models](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/datalyes)
[![Datasets](https://img.shields.io/badge/🤗-Datasets-blue.svg)](https://huggingface.co/datalyes)

Resources of the paper [PatenTEB](https://arxiv.org/abs/2510.22264) (Iliass Ayaou and Denis
Cavallucci, 2025): a benchmark of 15 patent text embedding tasks and 12 `patembed` embedding models
trained on it, from 67M to 344M parameters and up to 4,096 tokens of context.

## Overview

- 15 tasks in four families: retrieval, classification, clustering and paraphrase.
- 2,057,286 examples. The 319,320 test examples are released on Hugging Face. The 1,556,751 training
  and 181,215 validation examples are not released yet.
- 12 models on Hugging Face: 6 sizes, 3 long-context variants and 3 ablations.

Results reported in the paper:

- 0.654 overall score on PatenTEB, `patembed-large`.
- 0.494 V-measure on MTEB BigPatentClustering.v2, `patembed-base`. The best previous score was
  0.445 (SFR-Embedding-2_R).
- 0.377 NDCG@100 on DAPFAM cross-domain patent retrieval, `patembed-large`.

## Datasets

All 15 tasks are at [huggingface.co/datalyes](https://huggingface.co/datalyes). Each dataset is
gated with automatic approval: accept its conditions on its page, then log in before loading it.

| Family | Task | What it tests |
|---|---|---|
| Classification | [`class_bloom`](https://huggingface.co/datasets/datalyes/class_bloom) | Citation timing |
| Classification | [`class_nli_oldnew`](https://huggingface.co/datasets/datalyes/class_nli_oldnew) | Citation direction |
| Classification | [`class_text2ipc3`](https://huggingface.co/datasets/datalyes/class_text2ipc3) | IPC3 technology class |
| Clustering | [`clusters_ext_full_ipc`](https://huggingface.co/datasets/datalyes/clusters_ext_full_ipc) | Clusters by IPC |
| Clustering | [`clusters_inventor`](https://huggingface.co/datasets/datalyes/clusters_inventor) | Clusters by inventor |
| Symmetric retrieval | [`retrieval_IN`](https://huggingface.co/datasets/datalyes/retrieval_IN) | Same domain (identical IPC3) |
| Symmetric retrieval | [`retrieval_MIXED`](https://huggingface.co/datasets/datalyes/retrieval_MIXED) | Partial IPC3 overlap |
| Symmetric retrieval | [`retrieval_OUT`](https://huggingface.co/datasets/datalyes/retrieval_OUT) | Cross domain (disjoint IPC3) |
| Asymmetric retrieval | [`title2full`](https://huggingface.co/datasets/datalyes/title2full) | Title to full document |
| Asymmetric retrieval | [`problem2full`](https://huggingface.co/datasets/datalyes/problem2full) | Problem to full document |
| Asymmetric retrieval | [`problem2solution`](https://huggingface.co/datasets/datalyes/problem2solution) | Problem to solution |
| Asymmetric retrieval | [`effect2full`](https://huggingface.co/datasets/datalyes/effect2full) | Effect to full document |
| Asymmetric retrieval | [`effect2substance`](https://huggingface.co/datasets/datalyes/effect2substance) | Effect to substance |
| Paraphrase | [`para_problem`](https://huggingface.co/datasets/datalyes/para_problem) | Problem paraphrase |
| Paraphrase | [`para_solution`](https://huggingface.co/datasets/datalyes/para_solution) | Solution paraphrase |

| Family | Tasks | Test examples | Metric |
|---|---|---|---|
| Retrieval | 8 | 107,110 | NDCG@10 |
| Classification | 3 | 40,764 | Macro-F1 |
| Clustering | 2 | 134,064 | V-measure |
| Paraphrase | 2 | 37,382 | Pearson r |

## Models

All 12 models are at [huggingface.co/datalyes](https://huggingface.co/datalyes). Five are open, the
other seven are gated and need an approved access request.

| Model | Parameters | Dimension | Access |
|---|---|---|---|
| [`patembed-large`](https://huggingface.co/datalyes/patembed-large) | 344M | 1024 | open |
| [`patembed-base`](https://huggingface.co/datalyes/patembed-base) | 193M | 768 | open |
| [`patembed-base_small`](https://huggingface.co/datalyes/patembed-base_small) | 143M | 512 | open |
| [`patembed-small`](https://huggingface.co/datalyes/patembed-small) | 117M | 384 | gated |
| [`patembed-mini`](https://huggingface.co/datalyes/patembed-mini) | 92M | 256 | gated |
| [`patembed-nano`](https://huggingface.co/datalyes/patembed-nano) | 67M | 128 | gated |
| [`patembed-base_long_1024`](https://huggingface.co/datalyes/patembed-base_long_1024) | 149M | 768 | gated |
| [`patembed-base_long_2048`](https://huggingface.co/datalyes/patembed-base_long_2048) | 149M | 768 | gated |
| [`patembed-base_long_4096`](https://huggingface.co/datalyes/patembed-base_long_4096) | 149M | 768 | open |
| [`patembed-large_no_prompts`](https://huggingface.co/datalyes/patembed-large_no_prompts) | 344M | 1024 | open |
| [`patembed-large_all_ret_only`](https://huggingface.co/datalyes/patembed-large_all_ret_only) | 344M | 1024 | gated |
| [`patembed-large_all_no_classif`](https://huggingface.co/datalyes/patembed-large_all_no_classif) | 344M | 1024 | gated |

A `_long_N` variant reads up to N tokens. The last three are ablations: trained without task
prompts, on the retrieval tasks only, and without the classification tasks.

## Quick start

Load a test set (after accepting its conditions on Hugging Face and running `hf auth login`):

```python
from datasets import load_dataset

test = load_dataset("datalyes/class_bloom")["test"]
```

Encode texts. The models were trained with one prompt per task, listed in each model's
`config_sentence_transformers.json`. Pass the prompt of your task:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("datalyes/patembed-base")
embeddings = model.encode(
    ["A method for manufacturing semiconductor devices..."],
    prompt="encode document for ipc classification: ",
)
```

For the retrieval and paraphrase tasks, queries and documents have different prompts: encode each
side with its own.

## Paper

Iliass Ayaou and Denis Cavallucci. PatenTEB: A Comprehensive Benchmark and Model Family for Patent
Text Embedding. arXiv:2510.22264, 2025.

```bibtex
@misc{ayaou2025patentebcomprehensivebenchmarkmodel,
      title={PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding},
      author={Iliass Ayaou and Denis Cavallucci},
      year={2025},
      eprint={2510.22264},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.22264}
}
```

## License

The datasets and models are released under
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Contact

- Iliass Ayaou, postdoctoral researcher, ICube laboratory, INSA Strasbourg:
  iliass.ayaou@insa-strasbourg.fr
- Denis Cavallucci, professor, ICube laboratory, INSA Strasbourg: denis.cavallucci@insa-strasbourg.fr

ICube laboratory (UMR 7357), INSA Strasbourg, 24 Bd de la Victoire, 67000 Strasbourg, France.

## Related resources

- Raw patent data: [Lens.org](https://www.lens.org).
- [DAPFAM](https://doi.org/10.1016/j.array.2026.100720), the cross-domain patent retrieval benchmark
  (Array 29, 100720, 2026, preprint [arXiv:2506.22141](https://arxiv.org/abs/2506.22141)). Its 18
  retrieval tasks are part of MTEB
  ([embeddings-benchmark/mteb#2946](https://github.com/embeddings-benchmark/mteb/pull/2946)).
- [MTEB](https://github.com/embeddings-benchmark/mteb), the Massive Text Embedding Benchmark.
- [BERT for Patents](https://huggingface.co/anferico/bert-for-patents), a BERT model pretrained on
  patents.
- [Sentence Transformers](https://www.sbert.net/).
