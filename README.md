# PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2510.22264)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Models](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/datalyes)
[![Datasets](https://img.shields.io/badge/🤗-Datasets-blue.svg)](https://huggingface.co/datalyes)



[Paper](https://arxiv.org/abs/2510.22264) • [Models](https://huggingface.co/datalyes) • [Datasets](https://huggingface.co/datalyes)

</div>

---

## 📋 Overview

**PatenTEB** addresses a critical gap in patent text understanding by providing the first comprehensive benchmark specifically designed for patent embeddings, along with a family of state-of-the-art models.

### Key Highlights

- 🎯 **15 benchmark tasks** across retrieval, classification, paraphrase, and clustering
- 📊 **319K test examples** (publicly released) + 1.74M train/val (planned release)
- 🚀 **12 trained models** (67M-344M parameters) achieving SOTA on patentTEB and external benchmarks
- ✅ **All resources publicly available** under CC BY-NC-SA 4.0 license
- 🔄 **MTEB integration upcoming** (PR in progress)

### State-of-the-Art Results

- **0.654 Overall Score** on PatenTEB (15 tasks)
- **0.494 V-measure** on MTEB BigPatentClustering.v2 (new SOTA, previous: 0.445)
- **0.377 NDCG@100** on DAPFAM cross-domain patent retrieval

---

## 🔗 Resources Index

### 📦 Datasets (15 Tasks)

All datasets are available on HuggingFace: [huggingface.co/datalyes](https://huggingface.co/datalyes)

#### Classification Tasks (3)
- [`class_bloom`](https://huggingface.co/datasets/datalyes/class_bloom) - Citation timing classification
- [`class_nli_oldnew`](https://huggingface.co/datasets/datalyes/class_nli_oldnew) - Citation directionality  
- [`class_text2ipc3`](https://huggingface.co/datasets/datalyes/class_text2ipc3) - IPC3 technology classification

#### Clustering Tasks (2)
- [`clusters_ext_full_ipc`](https://huggingface.co/datasets/datalyes/clusters_ext_full_ipc) - IPC-based clustering
- [`clusters_inventor`](https://huggingface.co/datasets/datalyes/clusters_inventor) - Inventor-based clustering

#### Symmetric Retrieval Tasks (3)
- [`retrieval_IN`](https://huggingface.co/datasets/datalyes/retrieval_IN) - Same domain (identical IPC3)
- [`retrieval_MIXED`](https://huggingface.co/datasets/datalyes/retrieval_MIXED) - Mixed domain (partial IPC3 overlap)
- [`retrieval_OUT`](https://huggingface.co/datasets/datalyes/retrieval_OUT) - Cross domain (disjoint IPC3)

#### Asymmetric Retrieval Tasks (5)
- [`title2full`](https://huggingface.co/datasets/datalyes/title2full) - Title → Full document
- [`problem2full`](https://huggingface.co/datasets/datalyes/problem2full) - Problem → Full document
- [`problem2solution`](https://huggingface.co/datasets/datalyes/problem2solution) - Problem → Solution
- [`effect2full`](https://huggingface.co/datasets/datalyes/effect2full) - Effect → Full document
- [`effect2substance`](https://huggingface.co/datasets/datalyes/effect2substance) - Effect → Substance

#### Paraphrase Tasks (2)
- [`para_problem`](https://huggingface.co/datasets/datalyes/para_problem) - Problem paraphrase detection
- [`para_solution`](https://huggingface.co/datasets/datalyes/para_solution) - Solution paraphrase detection

### 🤖 Models (12 Models)

All models are available on HuggingFace: [huggingface.co/datalyes](https://huggingface.co/datalyes)

#### Core Models (6) - with prompts
- [`patembed-large`](https://huggingface.co/datalyes/patembed-large) - 344M params, 1024-dim (flagship)
- [`patembed-base`](https://huggingface.co/datalyes/patembed-base) - 193M params, 768-dim (recommended)
- [`patembed-base_small`](https://huggingface.co/datalyes/patembed-base_small) - 143M params, 512-dim
- [`patembed-small`](https://huggingface.co/datalyes/patembed-small) - 117M params, 384-dim
- [`patembed-mini`](https://huggingface.co/datalyes/patembed-mini) - 92M params, 256-dim
- [`patembed-nano`](https://huggingface.co/datalyes/patembed-nano) - 67M params, 128-dim

#### Long-Context Models (3) - with prompts
- [`patembed-base_long_1024`](https://huggingface.co/datalyes/patembed-base_long_1024) - 149M params, 1024 tokens
- [`patembed-base_long_2048`](https://huggingface.co/datalyes/patembed-base_long_2048) - 149M params, 2048 tokens
- [`patembed-base_long_4096`](https://huggingface.co/datalyes/patembed-base_long_4096) - 149M params, 4096 tokens

#### Ablation Models (3)
- [`patembed-large_no_prompts`](https://huggingface.co/datalyes/patembed-large_no_prompts) - Without prompts
- [`patembed-large_all_ret_only`](https://huggingface.co/datalyes/patembed-large_all_ret_only) - Retrieval tasks only
- [`patembed-large_all_no_classif`](https://huggingface.co/datalyes/patembed-large_all_no_classif) - No classification tasks


---

## 🚀 Quick Start

### Load a Dataset

```python
from datasets import load_dataset

# Load any of the 15 tasks
dataset = load_dataset("datalyes/class_bloom")
test_data = dataset['test']
```

### Use a Model

```python
from sentence_transformers import SentenceTransformer

# Load any patembed model (most use prompts automatically)
model = SentenceTransformer('datalyes/patembed-base')

# Encode patent texts
texts = ["A method for manufacturing semiconductor devices..."]
embeddings = model.encode(texts)
```

---

## 📄 Paper

**Title**: PatenTEB: A Comprehensive Benchmark and Model Family for Patent Text Embedding

**Authors**: Iliass Ayaou, Denis Cavallucci (ICUBE Laboratory, INSA Strasbourg)

**arXiv**: [2510.22264](https://arxiv.org/abs/2510.22264)

### Citation

If you use PatenTEB datasets or models, please cite our paper:

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

---

## 📊 Benchmark Details

### Dataset Statistics

| Split | Examples | Status |
|-------|----------|--------|
| **Test** | **319,320** | **✅ Released** |
| Train | 1,556,751 | 📅 Planned (future release) |
| Validation | 181,215 | 📅 Planned (future release) |
| **Total** | **2,057,286** | |

### Task Distribution

| Task Family | # Tasks | Test Examples | Metrics |
|-------------|---------|---------------|---------|
| Classification | 3 | 40,764 | Macro-F1 |
| Clustering | 2 | 134,064 | V-measure |
| Retrieval | 8 | 111,748 | NDCG@10 |
| Paraphrase | 2 | 32,744 | Pearson r |

### Model Performance Highlights

| Model | Params | PatenTEB Score | BigPatent V-measure | DAPFAM NDCG@100 |
|-------|--------|----------------|---------------------|-----------------|
| patembed-large | 344M | 0.654 | 0.458 | 0.377 |
| patembed-base | 193M | 0.645 | **0.494** (SOTA) | 0.369 |
| patembed-base_small | 143M | 0.639 | 0.492 | 0.363 |
| patembed-small | 117M | 0.625 | 0.485 | 0.353 |

---


## 📜 License

All PatenTEB resources (datasets, models, code) are released under:

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**

---

## 👥 Contact

**Authors**:
- **Iliass Ayaou** - PhD Candidate, ICUBE Laboratory
  - Email: iliass.ayaou@insa-strasbourg.fr
  - HuggingFace: [@datalyes](https://huggingface.co/datalyes)
  
- **Denis Cavallucci** - Professor, ICUBE Laboratory
  - Email: denis.cavallucci@insa-strasbourg.fr

**Institution**: ICUBE Laboratory (UMR 7357), INSA Strasbourg
- Address: 24 Bd de la Victoire, 67000 Strasbourg, France
- Web: [icube.unistra.fr](https://icube.unistra.fr/)

---

## 🙏 Acknowledgments

**Data Source**: Initial Raw Patent data sourced from [Lens.org](https://lens.org).
All the giants and predecessors in the open source and research community on whose shoulders this work stands, thank you.

**Related Resources**:
- **MTEB**: Massive Text Embedding Benchmark - [github.com/embeddings-benchmark/mteb](https://github.com/embeddings-benchmark/mteb)
- **DAPFAM**: Cross-domain patent retrieval benchmark - [arxiv.org/abs/2506.22141](https://arxiv.org/abs/2506.22141)  
- **BERT-for-Patents**: Domain-pretrained BERT - [huggingface.co/bert-for-patents](https://huggingface.co/bert-for-patents)
- **Sentence Transformers**: Framework - [sbert.net](https://www.sbert.net/)

---

<div align="center">

**PatenTEB** | Advancing Patent Text Understanding Through Comprehensive Evaluation

[Paper](https://arxiv.org/abs/2510.22264) • [Models](https://huggingface.co/datalyes) • [Datasets](https://huggingface.co/datalyes) • [GitHub](https://github.com/iliass-y/patenteb)



</div>
