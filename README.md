# NPS-FACL: A Fragment-Aware Chemical Language Model with Explainable Uncertainty Quantification

## Overview

The FACL model is a transformer-based framework designed to classify novel psychoactive substances (NPS) for forensic toxicology applications. It leverages fragment-aware tokenization and explainable uncertainty quantification to enhance NPS detection, offering a valuable tool for rapid substance identification.

## Model Architecture

The FACL model uses an NPS-aware tokenizer to process molecular structures, focusing on key substructures. The overview of the model pipeline is shown below:

![Model Architecture](figures/figure1_overview.pdf)

**Figure 1**: Overview of the FACL model pipeline, illustrating the NPS detection benchmark and fragment-aware tokenization process.

## Results

FACL outperforms baseline models in NPS classification, as demonstrated by performance comparisons and confusion matrices:

![Results](figures/figure4_result.pdf)

**Figure 2**: (a) Performance comparison of FACL against baseline models; (b) Confusion matrices of FACL and baseline models.

## Dataset

The dataset is available from the author (liupf7@mail2.sysu.edu.cn) upon reasonable request. An example dataset, `data/benchmark.csv`, is provided in the repository for reference.
