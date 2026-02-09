# Design Specification for Floating-Point Error Bounding in GEMM
**Variance-Estimation based Algorithm-Based Fault Tolerance (V-ABFT)**

[中文版](README.md) | **English**

| Document Version | 1.2  |  |  |
| :--- | :--- | :--- | :--- |
| **Target Scenario** | GEMM operator verification in LLM training/inference | **Hardware Target** | Ascend 910B |

---

## 1. Background & Objectives

### 1.1 Problem Definition
In large-scale matrix multiplication (GEMM) for deep learning (especially LLMs), hardware soft errors (bit-flips) are difficult to distinguish from inherent floating-point round-off errors. As the demand for larger model scales and faster training/inference grows, the use of low-precision floating-point numbers (e.g., BF16, FP8) and large matrices (e.g., $K > 8192$) has become increasingly common, posing severe challenges to traditional **threshold-based verification algorithms**.

*   **Challenges**:
    *   As precision decreases (BF16/FP8) and matrix dimensions increase ($K > 8192$), the magnitude of accumulated round-off errors may exceed the deviation caused by single-event upsets (SEUs), causing traditional fixed-threshold verification to fail (missed detections or false positives).
    *   Under low precision, accumulation errors are significantly amplified with increasing computation depth, and relying solely on C-matrix threshold detection becomes increasingly unreliable.
    *   Moreover, since the cost of false positives is often higher (requiring chip replacement/recomputation), we aim to maximize the detection rate under the premise of zero false positives.
*   **Current State**: The original A-ABFT scheme:
    *   Its computation logic is not suitable for the SIMD programming model.
    *   While its estimates are significantly better than traditional SEA-ABFT (by 1-2 orders of magnitude), they still differ from actual round-off errors by 2-4 orders of magnitude, growing with matrix size.
    *   A-ABFT provides estimates based on vector inner product operations, meaning the error is composed of checksum and row-sum errors separately, which increases computation overhead and is the reason for loose thresholds.

    The baseline scheme (based on A-ABFT), while simplifying some computations, still has excessively high computation overhead ($O(N^2)$ level, significant for small matrices), and due to its simplifications, cannot pass diverse data tests.

### 1.2 Design Objectives
This design proposes a **Variance-Estimation based dynamic threshold algorithm (V-ABFT)**, aiming to achieve the following:
1.  **High Accuracy**: Under FP8/BF16 low precision, precisely fit the round-off error upper bound through blocking, variance estimation, and confidence intervals.
2.  **Low Overhead**: Keep the verification computation overhead within the masking range of the GEMM computation pipeline, with theoretical extra FLOPs significantly lower than A-ABFT.
3.  **Pipeline Affinity**: Optimize the heterogeneous collaboration between AIV (vector unit) and AIC (matrix unit).

---

## 2. Theoretical Framework

### 2.1 Variance Estimation

#### 2.1.1 Extreme Value Theory

To avoid the high cost of computing variance over the full data, this design employs **Extreme Value Theory (EVT)** to infer the data distribution's variance from the row/column maximum values.

Assuming matrix elements $x$ are independently and identically distributed (i.i.d.) with mean $\mu$ and variance $\sigma^2$. For a sequence of length $n$, its maximum $m$ approximately satisfies:

$$ m \approx \mu + \sigma \sqrt{2 \ln n} $$

From this, the **fast estimation formula** for standard deviation $\sigma$ is:

$$ \sigma_{est} \approx \frac{m - \mu}{\sqrt{2 \ln n}} $$

> **Engineering Significance**: By efficiently computing `Max` and `Mean` on the AIV unit, we can obtain variance estimates without computing expensive sums of squares.
> **Limitation**: Strong distributional assumptions; cannot adapt to the diverse matrix distributions in LLM training/inference, such as truncated matrices.

#### 2.1.2 Upper Bound Estimation Method

Given a sequence $x_1, x_2, ..., x_n$ with maximum $m$, minimum $l$, and mean $\mu$, its variance $\sigma^2$ satisfies:

$$ \sigma^2 \leq (m - \mu)(\mu - l) $$

Equality holds when the sequence only takes the maximum/minimum values. Thus, we can quickly provide an upper bound on variance using three quantities.

> **Advantage**: Distribution-free, more robust.
> **Limitation**: Overestimates variance for normally distributed data, which increases Type II errors.


### 2.2 "Black-Box" Accumulation Model for Floating-Point Operations

Our starting point is rigorous floating-point error analysis.

Define the unit roundoff $u$:
*   **FP32**: $u = 2^{-24} \approx 5.96 \times 10^{-8}$
*   **BF16**: $u = 2^{-8} \approx 3.90 \times 10^{-3}$
*   **FP8 (E4M3)**: $u = 2^{-4} = 6.25 \times 10^{-2}$
*   **FP8 (E5M2)**: $u = 2^{-3} = 1.25 \times 10^{-1}$

The basic arithmetic model satisfies:

$$ fl(x \cdot y) = (x \cdot y)(1 + \delta u), \quad |\delta| \le 1$$

$$ fl(x + y) = (x + y)(1 + \delta u), \quad |\delta| \le 1 $$

On mixed-precision hardware (e.g., Tensor Core / Cube Unit), the accumulation order in matrix multiplication is often determined by the hardware microarchitecture (e.g., blocked accumulation, tree reduction), and intermediate precision conversions may be involved (FP16 $\to$ FP32 $\to$ FP16, high precision within AIC). Therefore, we treat floating-point operations as a "black box" with bounded errors.
For an accumulation sequence of length $K$, considering the pipeline depth and uncertainty in accumulation order, we model the cumulative round-off error as $\prod_{i=1}^s(1+\delta_i u)$, where $s$ is the effective accumulation depth. For notational convenience, we omit the subscript of $\delta$, treating it as a random variable fluctuating in $[-1, 1]$, and write the error as $(1+\delta u)^s$. We use $fl(...)$ to denote the actual floating-point result of the operations in parentheses.

### 2.3 Algebraic Expression of Verification Error

The core logic of ABFT is to compare the "row sum of the computation result" with the "product of pre-computed row sums of input matrices".
For input matrices $A \in R^{M \times K}, B \in R^{K \times N}$, define the verification error $E$ as the absolute value of the check difference:

$$
E = \left| fl\left( \sum_n fl \left(\sum_k A_{mk} B_{kn} \right) \right) - fl\left( \sum_k A_{mk} fl\left( \sum_n B_{kn} \right) \right) \right|
$$

Using the floating-point model above to expand, the error arises from the difference in accumulation depths $s_1$ (multiply then accumulate then sum) and $s_2$ (sum then multiply) between the two computation paths:

$$
\begin{aligned}
E &= \left| \sum_n \sum_k A_{mk} B_{kn} (1+\delta u)^{s_1} - \sum_n \sum_k A_{mk} B_{kn} (1+\delta u)^{s_2} \right| \\
  &= \left| \sum_n \sum_k A_{mk} B_{kn} \cdot \underbrace{((1+\delta u)^{s_1} - (1+\delta u)^{s_2})}_{\text{Effective Error Factor } e_{kn}} \right| \\
  &:= \left| \sum_n \sum_k e_{kn} A_{mk} B_{kn} \right|
\end{aligned}
$$

> **Note**: $e_k$ represents the **composite error coefficient** determined jointly by the path difference and machine precision.

### 2.4 Statistical Expansion Based on Variance

Directly applying the triangle inequality ($|a+b| \le |a|+|b|$) leads to overly loose thresholds (worst-case bound), which cannot detect small errors under low precision. Therefore, we introduce a statistical model.

Assuming the elements of $A, B$ are independently distributed, we decompose using mean $\mu$ and standard deviation $\sigma$:

$$ A_{mk} = \mu_{Am} + \sigma_{Am} \cdot a_{mk}, \quad a_{mk} \sim F_a $$

$$ B_{kn} = \mu_{Bk} + \sigma_{Bk} \cdot b_{kn}, \quad b_{kn} \sim F_b $$

> Note: Here $F_a, F_b$ are two distributions with unit variance, not necessarily unit normal.
According to the updated variable definitions ($A$ uses row statistics $\mu_{Am}$, $B$ uses row statistics $\mu_{Bk}$), the subscripts and summation logic in the derivation need to be strictly synchronized.

Expanding $E$:

$$
\begin{aligned}
E &= \left| \sum_n \sum_k e_{kn} (\mu_{Am} + \sigma_{Am} a_{mk})(\mu_{Bk} + \sigma_{Bk} b_{kn}) \right| \\
  &= \left| \sum_k (\mu_{Am} + \sigma_{Am} a_{mk}) \underbrace{\left(e_{kn} \sum_n (\mu_{Bk} + \sigma_{Bk} b_{kn}) \right)}_{\text{Sum over } N \text{ columns (Row } k \text{ of B)}} \right|
\end{aligned}
$$

By the Central Limit Theorem (CLT), summing over the $N$ columns of $B$. Note that $\mu_{Bk}$ and $\sigma_{Bk}$ are statistics of row $k$ of $B$, so they are constants when summing over $n$. Let $\alpha_k=\frac{\sum_n e_{kn}}{N}$, $\beta_k=\sqrt{\frac{\sum_n e_{kn}^2}{N}}$, then:

$$ \sum_n e_{kn} \mu_{Bk} = N \alpha_k \mu_{Bk} $$

$$ \sum_n e_{kn} \sigma_{Bk} b_{kn}  = \sqrt{N} \sigma_{Bk} \beta_k b'_{k} $$

where $b'_k \sim F_b'$ is a unit (variance) variable representing the random fluctuation of row $k$ of $B$.

Substituting back, the error $E$ becomes an accumulation over the $K$ dimension:

$$
E = \left| \sum_k^K \left[ (\mu_{Am} + \sigma_{Am} a_{mk}) \cdot (N \alpha_k \mu_{Bk} + \sqrt{N} \sigma_{Bk}\beta_k b'_{k}) \right] \right|
$$

Expanding the four product terms (note $\mu_{Am}, \sigma_{Am}$ do not vary with $k$ and can be extracted outside the summation):

$$
\begin{aligned}
E = \Bigg| \bigg( & \underbrace{N \mu_{Am} \sum_k \alpha_k \mu_{Bk}}_{\text{(1) Bias Term}} + \underbrace{\sqrt{N} \mu_{Am} \sum_k \sigma_{Bk} \beta_k b'_k}_{\text{(2) Random B Term}} \\
& + \underbrace{N \sigma_{Am} \sum_k \alpha_k\mu_{Bk} a_{mk}}_{\text{(3) Random A Term}} + \underbrace{\sqrt{N} \sigma_{Am} \sum_k \sigma_{Bk} \beta_k a_{mk} b'_k}_{\text{(4) Interaction Term}} \bigg) \Bigg|
\end{aligned}
$$

#### Physical Interpretation

1.  **Term ① (Bias Term)**: $N \mu_{Am} (\sum \mu_{Bk})$. This is the primary DC component, determined by the row mean of $A$ and the column-sum mean of $B$.
2.  **Term ② (Random B)**: $\sqrt{N} \mu_{Am} \sqrt{K} \dots$. Caused by within-row fluctuations of matrix $B$, growing with $\sqrt{K}$.
3.  **Term ③ (Random A)**: $N \sigma_{Am} \sqrt{K} \dots$. Caused by within-row fluctuations of matrix $A$, growing with $\sqrt{K}$.
4.  **Term ④ (Interaction)**: $\sqrt{N} \sigma_{Am} \sqrt{K} \dots$, dominant when both A and B have zero mean, growing with $\sqrt{K}$.

Then applying the triangle inequality,

$$
\begin{aligned}
E \leq \underbrace{\left| \sum_k \alpha_k N \mu_{Am} \mu_{Bk} \right|}_{\text{Bias Term (DC)}} + \underbrace{\left| \sum_k \beta_k \sqrt{N} \mu_{Am} \sigma_{Bk} b'_{k}  +  \sum_k \alpha_k N \sigma_{Am} a_{mk} \mu_{Bk} \right|}_{\text{Linear Random Term (Primary Noise)}} + \underbrace{\left| \sum_k \beta_k \sqrt{N} \sigma_{Am} a_{mk} \sigma_{Bk} b'_{k} \right|}_{\text{Interaction Term (Secondary Noise)}}
\end{aligned}
$$

To obtain an engineering-usable threshold, we introduce a uniform upper bound $e_{max} \ge max\{|\alpha_k|, |\beta_k|\}$ and extract it:
For the random terms, using the variance property of sums of independent random variables ($Var(\sum X_i) = \sum Var(X_i)$), and taking $4\sigma$ (approximately 99.9% confidence for normal distributions) as the safety margin:

$$
\begin{aligned}
Bound &\lesssim e_{max} \left( \underbrace{N|\mu_{Am}| \sum_k |\mu_{Bk}|}_{\text{Deterministic Bound}} + 4\sqrt{\underbrace{N \sum_k \mu_{Am}^2 \sigma_{Bk}^2}_{\text{Var of Term 2}} + \underbrace{N^2 \sigma_{Am}^2 \sum_k \mu_{Bk}^2}_{\text{Var of Term 3}}} + 4\underbrace{\sqrt{N}\sigma_{Am} \sqrt{\sum_k \sigma_{Bk}^2}}_{\text{SD of Term 4}} \right)
\end{aligned}
$$

> Note: Terms 2 and 3 are often independent variables, so we combine them, while Term 4 has some coupling with Terms 2 and 3, so it is bounded separately. Additionally, when a and b are independent, the variance of $A_{mk} b'_{k}$ is 1, but if they are positively correlated, the variance > 1, in which case a larger coefficient may be needed.

## 3. Final Engineering Threshold Formula

Through the above statistical derivation, the error upper bound no longer follows linear accumulation but obeys variance composition rules. We decompose the error bound $T_{bound}$ into two parts: **Mean Drift (DC Component)** and **Random Fluctuation (AC Component)**.

$$
T_{bound} \approx e_{max} \cdot \left( \underbrace{K \cdot N \cdot |\mu_A \mu_B|}_{\text{Bias Term}} + \underbrace{\sqrt{K} \cdot \Phi(\mu, \sigma, N)}_{\text{Variance Term}} \right)
$$

In engineering implementation, to avoid computing the complex $\Phi$ function, we use the extreme value theory from Section 2.1 to replace $\sigma$ with $(Max - Mean)/\sqrt{2ln(n)}$, or use the upper bound estimate to replace $\sigma$ with $\sqrt{(Max - Mean)(Mean - Min)}$. For $e_{max}$, we currently adopt $3u$. Hardware-specific empirical tests can be conducted later.
The final **computable threshold** is:

$$
\text{Threshold} = 3u \left( \underbrace{n|\mu_{Am}| \sum_k |\mu_{Bk}|}_{\text{Deterministic Bound}} + 4\sqrt{\underbrace{N \sum_k \mu_{Am}^2 \sigma_{Bk}^2}_{\text{Var of Term 2}} + \underbrace{N^2 \sigma_{Am}^2 \sum_k \mu_{Bk}^2}_{\text{Var of Term 3}}} + 4\underbrace{\sqrt{N}\sigma_{Am} \sqrt{\sum_k \sigma_{Bk}^2}}_{\text{SD of Term 4}} \right)
$$

> Note: Current rough estimation method for $e_{max}$: Generate matrices A, B with distribution $N(1,1)$, compute the relative error between the checksum and C matrix row sum at machine precision, repeat for 100k experiments, and take the maximum as $e_{max}$. More refined empirical tests can be conducted for different hardware. On the 910B chip, experimental values for BF16, FP32, FP16 are approximately 7.76e-03, 2.13e-06, 9.77e-04, so $e_{max}$ is set to 8e-03, 2.2e-06, 1e-03 respectively.
> Experiments show that under FP32, the estimated $e_{max}$ grows with matrix scale, while under low precision, $e_{max}$ has little relationship with matrix scale, possibly related to the hardware accumulator design.
> Compared to the original A-ABFT implementation, our threshold under FP32 is only one order of magnitude above the error (~20x) and does not grow with matrix size.

### 3.1 Sampling-Based Acceleration

Currently, the mean and variance estimation calculations account for a significant portion of the threshold overhead. To accelerate computation, we allow a certain level of false positives and perform sampling, followed by secondary confirmation when errors are detected. This reduces the additional computation overhead of the threshold.

Specifically, we can sample across stride columns to compute the mean and variance. For example, for the row mean $\mu_{Am}$ of matrix $A$, we can compute the mean and variance using only data with indices satisfying $i\text{ mod } 32 < 16$.

However, the impact of sampling ratio on the statistical estimation of matrices A and B is heterogeneous. Matrix A has a block row length of $K=1024$, while matrix B has a block row length of $N=256$. Intuitively, if the same sampling ratio is used, the confidence levels of A and B statistics will differ significantly. Therefore, we use different sampling ratios to balance the confidence of A and B matrix statistics, or only sample matrix A. To reduce false positive triggers, we can also increase the threshold by a certain proportion.

## 4. Discussion on Baseline: A-ABFT

The baseline scheme is built on **Mixed Precision Floating-point Error Analysis**.

It mainly models the following scenario: using **FP32** (high precision) to verify a **BF16** (low precision) matrix multiplication result, and computing the theoretically allowable maximum error.

We define the following symbols:

*   $A, B$: Input matrices. $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}$.
*   $C$: Result matrix, $C = A \times B$.
*   $N$: Number of columns of matrix $B$.
*   $K$: Inner dimension of the matrix multiplication.
*   $\epsilon_{high} = 2^{-23}$: Machine epsilon for FP32.
*   $\epsilon_{low} = 2^{-8}$: Machine epsilon for BF16.

The total error bound $E_{total}$ returned by the function consists of four parts:

$$ E_{total} = E_{\text{sum-C}} + E_{\text{ele-round}} + E_{check_1} + E_{check_2} $$

Below is the mathematical formula and physical meaning of each term:

#### $E_{\text{sum-C}}$ : Cumulative Summation Error of C

**Formula:**

$$ E_{\text{sum-C}} = \left( \sqrt{\frac{1}{8} \sum_{i=1}^{N} i^2} \right) \cdot \max(|C|) \cdot \epsilon_{high} $$


#### $E_{\text{ele-round}}$ : Cumulative Element Quantization Error of C

**Formula:**

$$ E_{\text{ele-round}} = \epsilon_{low}\sqrt{N}\max(|C|) $$

*   **Explanation**:
    *   $\max(|C|)$ is the vector obtained by taking the maximum absolute value per row of matrix C.
    *   $\sqrt{N}$ represents the standard deviation growth when summing $N$ independent random errors.
    *   This is the **dominant term** as it involves the low precision $\epsilon_{low}$.

#### $E_{check_1}$ : Multiplication Error and Accumulation of A and Be

**Formula:**

$$ \delta_1 = \epsilon_{high} \left( \sqrt{\frac{1}{8} \sum_{i=1}^{N} i^2 } \right) \max(|B|) $$

$$ E_{\text{prop}} = |A| \times \delta_1 $$

*   **Explanation**:
    *   $\max(|B|)$ is the vector obtained by taking the maximum absolute value per row of matrix B. $|A|$ is the matrix obtained by taking the absolute value of each element of A.
    *   $\delta_1$ is the cumulative error when computing the row sum of $B$ (same structure as the first term).

#### $E_{check_2}$ : Secondary Error from A Acting on Be Error

**Formula:**

$$ E_{\text{matmul-diff}} \approx \epsilon_{high}\left( \sqrt{\frac{1}{8} \sum_{k=1}^{K} k^2+\frac{K}{12}} \right)  \max(\max(|B|))\max(|A|) $$

*   **Explanation**:
    *   $\max(\max(|B|))$ is the scalar maximum of all absolute values in matrix B. $\max(|A|)$ is the vector obtained by taking the maximum absolute value per row of matrix A.
    *   It reflects the rounding noise under FP32 precision during $K$ multiply-accumulate operations.

Main limitations of this scheme:

1.  **Strong dependence on verification precision**: Assumes high-precision (FP32) verification, which requires sacrificing performance on actual hardware. In this case, Term 2 dominates the error bound. However, Term 2 was simplified and does not strictly follow A-ABFT's derivation, making it inaccurate.
2.  **Conservatism of the error model**: Uses worst-case analysis for summation operations, ignoring statistical distribution characteristics, resulting in overly loose thresholds for Terms 1, 3, 4, making it difficult to detect small errors. In practice, if FP16 is used for verification, Terms 1, 3, 4 dominate the error bound, yielding a 0% detection rate.
3.  **Neglect of A*B computation error**: Does not account for the rounding error of the low-precision matrix multiplication itself on verification results, leading to underestimated error bounds.
4.  **Implicit assumptions in A-ABFT**: The A-ABFT authors implicitly assumed in their derivation that the exponent and mantissa parts of intermediate data are independent. We are skeptical of this: for example, data uniformly distributed in [0.9, 1.1] — if the exponent is 1, the mantissa is in [1.0, 1.1]; if the exponent is 0.5, the mantissa is in [1.8, 2). The authors did not discuss whether this correlation affects the final error distribution.

For more details, see the experimental section.

## 5. Experiments: Detection Rate & False Positive Rate

False positive rates (FP8 measurement currently unavailable) and detection rates under different precisions. For V-ABFT, we set $e_{max}$ to 8e-03, 2.2e-06, 1e-03 for BF16, FP32, FP16 respectively, with variance term coefficient set to 2.5.

We primarily focus on errors occurring in the exponent bits, since mantissa bit-flips cause relatively small errors (single-element numerical impact <33.4%). Additionally, large models contain numerous norm/softmax operations that suppress the impact of small numerical errors. Similarly, 1→0 flips in exponent bits cause values to decrease, which has a relatively small numerical impact (<100%) and is not the main focus of ABFT algorithms. Therefore, we primarily focus on 0→1 flips. However, for experimental completeness, we also measure the detection rate of 1→0 flips.

### 5.1 Simulated Data

We generated multiple sets of simulated data, with elements of matrices $A$ and $B$ following these distributions:

1.  Normal distribution $N(1e\text{-}6,1)$
2.  Normal distribution $N(1,1)$
3.  Uniform distribution $U(-1,1)$
4.  Truncated normal distribution $N(0,1)$ truncated to $[-1,1]$


Scenarios 1 and 4 are the most common cases; scenarios 2 and 3 are used to test algorithm robustness. Both algorithms are tested on matrices of size $(M,K,N) = (128, 1024, 256)$ (adapted for Ascend chips), with single-bit 0→1 soft errors injected at bits 7-15 under BF16 precision.

> BF16 format: [Sign(1) | Exponent(8) | Mantissa(7)]
> FP32 format: [Sign(1) | Exponent(8) | Mantissa(23)]
> FP16 format: [Sign(1) | Exponent(5) | Mantissa(10)]
> The last bit of the mantissa is bit 0; the first bit of the exponent is bit 7.
> Note 1: We do not focus on 1→0 flips because they cause values to decrease, which has less impact on model training and is not the main focus of ABFT algorithms.
> Note 2: A bit-flip at bit 7 is equivalent to multiplying the number by $2^{2^0}=2$; at bit 8, by $2^{2^1}=4$; then 16, 256, etc. The growth is doubly exponential.

#### 5.1.1 False Positive Rate


| Algorithm | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **V-ABFT (BF16) (%)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **V-ABFT (FP32) (%)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **V-ABFT (FP16) (%)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **A-ABFT (%)** | 0.0000 | 84.8250 | 0.0000 | 0.0000 |

Experiment scale: 100k trials
> Note 1: The A-ABFT mixed-precision verification scheme can only be used to detect low-precision matrix multiplication results, so only the BF16 false positive rate was tested.
> Note 2: V-ABFT achieves zero false positive rate across all distributions, while A-ABFT performs poorly under high-mean normal distribution with a false positive rate as high as 84.825%. The reason lies in the neglect of A*B computation errors. Over-reliance on the final C matrix results, but the C matrix error itself is very large, causing verification failure.

#### 5.1.2 Detection Rate

For detection rate, we tested each bit position for bit-flips. Results are shown in the tables below:

V-ABFT (BF16) (%)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **7th** | 0.0064 | 0.0000 | 19.6558 | 10.8967 |
| **8th** | 36.6953 | 69.5500 | 46.8472 | 36.4867 |
| **9th** | 73.4750 | 100.0000 |  75.0310 | 99.3833 |
| **10th** | 99.9860 | -       | 99.8603 | 99.9567 |
| **11th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **12th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **13th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **14th** | 100.0000 | -        | 100.0000 | 100.0000 |
| **15th** | 4.4033 | 5.5100 | 42.3433 | 56.7233 |

V-ABFT (FP32) (%)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **23th** | 99.9367 | 100.0000 | 99.9633 | 99.9800 |
| **24th** | 99.9833 | 100.0000 | 99.9767 | 99.9867 |
| **25th** | 99.9967 | 100.0000 |  100.0000 | 99.9967 |
| **26th** | 99.9967 | 100.0000 | 100.0000 | 100.0000 |
| **27th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **28th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **29th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **30th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **31th** | 99.9667 | 100.0000 | 99.9833 | 99.9967 |

V-ABFT (FP16) (%)

| Bit | Normal $N(10^{-6},1)\times10^{-2}$ | Normal $N(1,1)\times10^{-2}$ | Uniform $U(-1,1)\times10^{-2}$ | Truncated Normal $N(0,1)\times10^{-2}$ |
|------|------------------------|------------------|-------------------|----------------------|
| **10th** | 67.0467 | -    | 77.2533 | - |
| **11th** | 88.6567 | -    | 92.2367 | - |
| **12th** | 80.4893 | 100.0000 |  100.0000 | 100.0000 |
| **13th** | 100.0000 | -    | 100.0000 | - |
| **14th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **15th** | 80.1267 | 100.0000 | 92.2933 | 100.0000 |

A-ABFT (BF16) (%)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **7th** | 87.1876 | * | 87.0669 | 80.1737 |
| **8th** | 96.9240 | * | 96.9887 | 94.5370 |
| **9th** | 99.5554 | * |  99.5269 | 99.2126 |
| **10th** | 99.9790 | * | 99.9930 | 100.0000 |
| **11th** | 100.0000 | * | 100.0000 | 100.0000 |
| **12th** | 100.0000 | * | 100.0000 | 100.0000 |
| **13th** | 100.0000 | * | 100.0000 | 100.0000 |
| **14th** | 53.9394 | * | 48.6708 | 55.6291 |
| **15th** | 93.1479 | * | 92.0962 | 91.1012 |

> Note 1: Cells with `-` indicate that type of flip cannot be injected; `*` indicates the false positive rate for that column is too high, making the detection rate meaningless.
> Note 2: FP16 data is scaled by 1e-2 because FP16's limited representable range causes matrix multiplication results to overflow to infinity.

Experiment scale: >10k trials (per bit position)

Under BF16, V-ABFT can reliably detect the top 5 exponent bit-flip errors and detect the 6th bit error with high probability. A-ABFT can detect all errors with high probability, but the cost is:
**1.** Requires high-precision verification (cannot switch to low precision);
**2.** Under high-mean distributions, the false positive rate is too high, rendering the detection rate meaningless.

In real production environments, the input matrices for matrix multiplication may be mixtures of various distributions, where V-ABFT's robustness is stronger.

#### 5.1.3 1→0 Flip Detection Rate

Results for 1→0 flips are shown below:

V-ABFT (BF16) (%)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **7th** | 0.0000 | 0.0000 | 0.0000 | 0.0400 |
| **8th** | 36.6953 | -     | 5.5800 | 11.0600 |
| **9th** | 0.0000 | -      |  2.0023 | 36.8740 |
| **10th** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **11th** | 0.0000 | -      | 0.0000 | 0.0000 |
| **12th** | 0.0000 | -      | 0.0000 | 0.0000 |
| **13th** | 0.0000 | -      | 0.0000 | 0.0000 |
| **14th** | 0.0000 | 0.0000 | 13.3900 | 28.6900 |
| **15th** | 4.4900 | -      | 42.3400 | 56.9300 |

V-ABFT (FP32) (%)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **23th** | 99.8700 | 100.0000 | 99.9500 | 99.9600 |
| **24th** | 99.8900 | - | 100.0000 | 99.9600 |
| **25th** | 99.9600 | - |  99.7343 | 99.8176 |
| **26th** | 99.0605 | 100.0000 | 100.0000 | 100.0000 |
| **27th** | 98.1978 | - | 99.7464 | 99.8748 |
| **28th** | 98.3240 | - | 99.7839 | 99.6716 |
| **29th** | 98.1148 | - | 99.7841 | 99.8750 |
| **30th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **31th** | 99.9600 | - | 99.9900 | 99.9900 |

V-ABFT (FP16) (%)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **10th** | 42.3333 | 99.5800  | 66.3667 | 100.0000 |
| **11th** | 77.7667 | 100.0000 | 64.9667 | 100.0000 |
| **12th** | 67.7333 | -        | 100.0000| -     |
| **13th** | 100.0000| 100.0000 |  -      | 100.0000 |
| **14th** | -       | -        | -       | -     |
| **15th** | 79.7333 | -        | 92.5667 | -     |

> Note: Cells with `-` indicate that type of flip cannot be injected.

Experiment scale: >10k trials (per bit position)

#### 5.1.4 Performance at Larger Scales

The standalone threshold testing algorithm (decoupled from the GEMM operator) cannot split the K dimension, and typical matrix dimensions are 4096×4096. Therefore, we tested false positive rates (all 0) and detection rates (BF16) at sizes (128, 4096, 256) and (4096, 4096, 4096).

V-ABFT Detection Rate (%) (128, 4096, 256)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **7th** | 0.0000 | 0.0000 | 9.5776 | 36.7597 |
| **8th** | 13.0348 | 92.5000 | 82.1467 | 69.3462 |
| **9th** | 39.8570 | 100.0000 |  81.8534 | 97.4612 |
| **10th** | 99.9846 | 100.0000 | 99.9949 | 99.9860 |
| **11th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **12th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **13th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **14th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |

V-ABFT Detection Rate (%) (4096, 4096, 4096)

| Bit | Normal $N(10^{-6},1)$ | Normal $N(1,1)$ | Uniform $U(-1,1)$ | Truncated Normal $N(0,1)$ |
|------|------------------------|------------------|-------------------|----------------------|
| **7th** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **8th** | 0.0000 | -      | 0.0000 | 26.4159 |
| **9th** | 0.0000 | 0.0000 |  3.4595 | 67.5388 |
| **10th** | 96.4110 | -       | 99.9383 | 100.0000 |
| **11th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **12th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **13th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |
| **14th** | 100.0000 | 100.0000 | 100.0000 | 100.0000 |

Experiment scale: >5k trials (per bit position)

### 5.2 Real Model Data

We also collected matrix data from actual large model training/inference for false positive rate testing.
The tested models include Llama-7B, GPT-2, and ViT-B/32. V-ABFT achieved zero false positive rate in all tests.

Data scale:

##### Llama-7B

All A-ABFT false positive matrices (111 matrices) were retested with V-ABFT, all achieving zero false positive rate.

##### GPT-2

5,379 matrix multiplication verifications, all with zero false positive rate.

##### ViT-B/32

Fine-tuning task, 50 epochs, 1% of matrix multiplications were sampled, totaling 5,937 matrix multiplication verifications.
