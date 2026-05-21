# Variable Selection in Non-Parametric Regression
## Under the supervision of Prof. Subhra Sankar Dhar

## Overview
This repository contains the implementation of a novel two-stage feature selection methodology for multiple non-parametric regression, specifically applied to breast cancer diagnostics. This also contains the project report and the presentation files.

* **Developed a two-stage feature selection algorithm in R** for high-dimensional non-parametric regression using the **Akaike Information Criterion (AIC), Nadaraya-Watson, and Local Linear Kernel Smoothing**
* **Engineered** a custom Leave-One-Out Cross-Validation (LOOCV) pipeline leveraging parallel computing (`doParallel`) to optimize runtime, utilizing a **Focused Information Criterion (FIC) inspired backward elimination** framework
* **Achieved** a **96.49%** out-of-sample classification accuracy, **outperforming a computationally heavy 25-variable control model** on the **Wisconsin Breast Cancer Dataset** from the UCI Machine Learning Repository

---

## Dataset
* The methodology is evaluated on the **Breast Cancer Wisconsin (Diagnostic) Dataset**
* **Sample Size:** $n = 569$ patients (357 benign, 212 malignant)
* **Covariates:** $p = 30$ continuous variables derived from 10 morphological characteristics of cell nuclei (e.g., radius, texture, perimeter) computed as Mean, Standard Error, and Worst

---

## Methodology

High-dimensional non-parametric models inherently suffer from the "curse of dimensionality," where expanding covariate spaces lead to data sparsity, high variance, and predictive failure. This project tackles this via a two-stage approach:

### Stage 1: Preliminary Parametric Screening
To rapidly filter redundant noise, a forward-stepwise logistic regression is applied using the **Akaike Information Criterion (AIC)**:
$$\text{AIC} = -2\ln(\hat{L}) + 2k$$ 
The lighter penalty of AIC (compared to BIC) preserves the necessary flexibility for the subsequent non-parametric stage, successfully reducing the space to an optimal subset size of $k=6$

### Stage 2: Non-Parametric Variable Importance
Following parametric reduction, an FIC-inspired marginal screening evaluates the predictive power of each surviving variable independently
1.  **Local Linear Estimation:** The marginal relationship between the response and each focus covariate $X_j$ is evaluated using the Local Linear estimator, chosen for its design-adaptive nature and automatic boundary correction bias of $O(h^2)$.
2.  **Bandwidth Optimization:** For each focus variable, the bandwidth $h_j$ is optimally selected via Data-Driven Least Squares Cross-Validation (LSCV)
3.  **Backward Elimination:** Covariates are ranked by their unbiased generalization error computed via LOOCV Mean Squared Error:
    $$MSE_{CV}(j)$$ = $$\frac{1}{n} \sum_{i=1}^{n}(Y_i - \hat{m} (X_{ij}; h_{j}^{*}))^2$$
    The variable with the highest error (least unique predictive information) is permanently eliminated 

---

## 📈 Key Results
* **Optimal Subset:** The methodology successfully isolated 5 structurally dominant features (e.g., `perimeter_worst`, `concavity_worst`)
* **Performance:** The parsimonious 5-variable model achieved an accuracy of **96.49%**
* **Negative Control:** A model trained on the discarded 25 variables suffered from extreme multicollinearity and yielded a lower accuracy of 94.74%, empirically validating the methodology's ability to combat the dimensionality collapse

---

## Tech Stack
* **Language:** R 
* **Libraries:** `np`: Kernel smoothing and bandwidth selection
    * `foreach` & `doParallel`: Parallel processing for LOOCV pipelines
    * `MASS`: Core statistical modeling

---

## Usage
1. Clone the repository.
2. Install the required R packages:
   ```R
   install.packages(c("MASS", "np", "foreach", "doParallel"))
3. Run final ```code.R``` to execute the full pipeline, from data ingestion to non-parametric elimination and final hold-out testing.  
