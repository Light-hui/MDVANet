# MDVANet
MDVANet:Enhanced representation learning with multi-dimensional variations aggregation for time series prediction

## Overview
This is the official repository for MDVANet. As the paper is currently under review, we only provide a model demo at this stage. The full implementation code will be released publicly upon acceptance of the paper.

## Description
To facilitate understanding, we elaborate on the parameter settings involved in this paper as follows.
- On 3D Parameter Design.
  - Taking the ETTh dataset (collected hourly) as an example, we set the input sequence length to 720. The small window length can be set to 24 or 48 (corresponding to one day or its multiples). Taking 48 as an example, to ensure the product of the three dimensions equals the input sequence length, we set the number of large windows to 3 (month-level scale) and the number of small windows within each large window to 5 (10 days of data, weekly-like scale).
- On Intrinsic Period Modeling
  - We first explored an automatic period detection approach based on TimesNet, which extracts dominant periods from frequency-domain components. However, we observed that this strategy leads to performance degradation on datasets with weak periodicity.
  - To enhance the model’s generalization across diverse datasets, we then investigated a manual period specification strategy for datasets lacking clear periodic patterns. Inspired by SparseTSF, we set a default period of 2 for such datasets and increased the weight of 3D feature fusion in the aggregation module. Experimental results demonstrate that our model still achieves competitive performance on these weakly periodic datasets, verifying its robust generalization ability.

## Supplement
For any further questions, please feel free to leave comments here or contact us via the information provided in the paper.
