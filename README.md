# Poverty Prediction Challenge

## 📋 Challenge Description (Official)

### Problem Definition

The goal of this challenge is to predict poverty outcomes using household survey data.  
Specifically, the task is to predict:

1. **Household-level consumption**  
   - Daily per capita consumption (2017 USD PPP)

2. **Population poverty rates**  
   - Percentage of the population living strictly below **19 predefined poverty thresholds**

---

### Dataset Details

The dataset consists of household-level survey responses, including:
- Demographics
- Education
- Utilities
- Food consumption and assets

**Training surveys**
- 100000
- 200000
- 300000

**Test surveys**
- 400000
- 500000
- 600000

---

### Performance Metric

Model performance is evaluated using a blended metric:

\[
\text{Score} = 0.9 \cdot \text{W-MAPE}_{rates} + 0.1 \cdot \text{MAPE}_{cons}
\]

Where:
- **MAPE\_cons** — Mean Absolute Percentage Error for household consumption
- **W-MAPE\_rates** — Weighted MAPE for poverty rates  
  (higher weight for thresholds near the 40% baseline percentile)

---

## 🧠 Solution Overview

This solution combines **machine learning techniques with domain-informed economic feature engineering** to improve robustness across heterogeneous surveys.

Instead of relying only on raw survey metrics, the model explicitly incorporates **consumption diversity as an economic proxy for welfare**.

### Key Ideas

#### 1. Economic Feature: Consumption (Dietary) Diversity

A central feature of this solution is a **consumption diversity indicator**, motivated by economic theory:

- Households with higher welfare typically consume a **wider variety of goods**
- Poorer households tend to concentrate spending on a narrow subset of essentials

**Implementation**
- For each household, the number of distinct items consumed within the recall window is counted
- This feature (`consumed_yes_count`) acts as a strong proxy for disposable income
- Empirically, it improves generalization across surveys more than many raw categorical indicators

This allows the model to learn **structural economic signals**, not just survey-specific correlations.

---

#### 2. Survey-Specific Modeling (Domain Shift Aware)

Household surveys differ substantially across:
- Geography
- Timing
- Sampling design
- Consumption baskets

To address this **domain shift**:
- Separate models are trained for each training survey
- Predictions are combined using survey-level similarity rather than naive pooling

---

#### 3. CDF Matching & Similarity-Based Ensembling

For each survey:
- A poverty **Cumulative Distribution Function (CDF)** is estimated
- Test surveys are matched to training surveys based on CDF similarity
- Final predictions are generated via a **distance-weighted ensemble**

This reduces sensitivity to:
- Survey-specific noise
- Leaderboard overfitting
- Random threshold effects

---

#### 4. Logical Post-Processing

To ensure valid outputs:
- Poverty rates are forced to be **monotonically increasing** across thresholds using cumulative constraints
- Mean-shift calibration aligns predicted and reference distributions

---

### Why This Works

- Combines **economic intuition + statistical learning**
- Reduces reliance on leaderboard luck
- Produces logically consistent poverty distributions
- Improves robustness under limited survey coverage

This approach consistently outperforms metric-only or pooled-survey baselines.

---

## 📁 Project Structure

```text
├── data/                       # Input CSVs (from DrivenData)
├── result/                     # Final model outputs
│   ├── predicted_household_consumption.csv
│   └── predicted_poverty_distribution.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Feature engineering
│   ├── trainer.py              # Survey-specific training
│   ├── inference.py            # CDF matching & ensembling
│   └── utils.py                # Softmax, CDF utilities
├── main.py                     # End-to-end pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/poverty-prediction-challenge.git
cd poverty-prediction-challenge
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Download the data

Obtain the datasets from **DrivenData** and place all CSV files into the `data/` directory.

---

### 4. Run the full pipeline

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train survey-specific models
- Generate consumption predictions
- Convert predictions into poverty distributions
- Save final submission files to the `result/` directory

---

## 📤 Outputs

After running the pipeline, the following files will be created:

- `result/predicted_household_consumption.csv`
- `result/predicted_poverty_distribution.csv`

These files are ready for direct submission.

---

## ✅ Notes

- Poverty rate predictions are guaranteed to be **monotonic across thresholds**
- The pipeline is fully deterministic given the same random seed
- Designed for clarity, reproducibility, and leaderboard optimization

---

## 📜 License

This project is provided for educational and research purposes.
