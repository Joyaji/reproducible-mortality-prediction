# MIMIC-III Synthetic Data Generator - Complete Documentation

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Main Features](#main-features)
   - 13 Implemented Features
3. [Usage](#usage)
4. [Configuration Parameters](#configuration-parameters)
5. [Validation and Quality](#validation-and-quality)
6. [Related Files](#related-files)

---

## 🎯 Overview

The MIMIC-III synthetic data generator creates realistic Intensive Care Unit (ICU) data for training and validating in-hospital mortality prediction models. The generator replicates clinical, temporal, and statistical characteristics of real ICU data, incorporating physiological patterns, individual variability, and realistic artifacts.

### **Dataset Specifications:**

- **Total Episodes:** 24,327 ICU admissions
- **Features:** 15 clinical variables (vital signs + lab tests)
- **Time Window:** 48 hours of observation per episode
- **Mortality Rate:** 20.8% (realistic imbalance)
- **Data Splits:**
  - Training: 16,972 episodes (69.9%)
  - Validation: 3,740 episodes (14.8%)
  - Test: 3,615 episodes (15.2%)

### **13 Main Implemented Features:**

1. ✅ **Circadian Patterns (24h)** - Natural variations throughout the day
2. ✅ **Sleep Simulation** - Vital signs reduction during 11pm-6am
3. ✅ **Meal Effects** - Glucose and HR spikes post-meals
4. ✅ **Advanced Temporal Variability** - Noise, jitter, dropout, artifacts
5. ✅ **Realistic Missingness** - MCAR, MAR, MNAR with temporal patterns
6. ✅ **Multivariate Correlations** - Hierarchical covariance matrix
7. ✅ **Documentation Quality** - Variation by shift and day of week
8. ✅ **Individual Variability** - Unique baselines, 5 progression patterns
9. ✅ **Stochastic Correlations** - Anti-overfitting, variable correlations
10. ✅ **Sophisticated Mortality Model** - Multivariate, non-linear, stochastic
11. ✅ **Label Noise and Ambiguity** - Clinical uncertainty, censoring, timing
12. ✅ **Dataset Shift** - Holdout domain for generalization evaluation
13. ✅ **Demographic Diversity** - Balancing, data augmentation, diverse conditions

---

## 🌟 Main Features

### **1. Circadian Patterns (24h)**

Simulates natural variations throughout the day:

```python
CIRCADIAN_PARAMS = {
    'Heart_Rate': {
        'peak_hour': 14,      # Peak at 2pm
        'trough_hour': 3,     # Minimum at 3am
        'amplitude': 0.08     # ±8% variation
    },
    'Temperature': {
        'peak_hour': 18,      # Peak at 6pm
        'trough_hour': 4,     # Minimum at 4am
        'amplitude': 0.5      # ±0.5°C
    }
}
```

**Effect:** HR varies naturally between day and night.

### **2. Sleep Simulation (11pm-6am)**

```python
SLEEP_PARAMS = {
    'start_hour': 23,
    'duration_hours': 7,
    'effects': {
        'Heart_Rate': -0.15,      # -15% during sleep
        'Respiratory_Rate': -0.10, # -10%
        'MAP': -0.08              # -8%
    }
}
```

**Effect:** Vital signs reduce during sleep period.

### **3. Meal Effects**

```python
MEAL_PARAMS = {
    'meal_times': [7, 12, 18],  # Breakfast, lunch, dinner
    'duration_hours': 2,
    'effects': {
        'Glucose': +0.30,  # +30% glucose post-meal
        'Heart_Rate': +0.05 # +5% HR
    }
}
```

### **4. Advanced Temporal Variability**

#### **Signal-Specific Noise (Realistic SNR):**

```python
SIGNAL_NOISE_PARAMS = {
    'Heart_Rate': 0.02,        # ±2% noise
    'SpO2': 0.005,             # ±0.5%
    'Temperature': 0.008,      # ±0.8%
    'Lactate': 0.05            # ±5%
}
```

#### **Time Warping/Jitter:**

- ±10% variation in temporal spacing
- Simulates measurement irregularity

#### **Temporal Dropout:**

- 10% randomly missing measurements
- Simulates equipment failures

#### **Artifacts:**

- 3% probability of isolated spikes
- Simulates patient movement

### **5. Realistic Missingness (MCAR/MAR/MNAR)**

```python
MISSINGNESS_PARAMS = {
    'mcar_rate': 0.30,  # 30% completely at random
    'mar_rate': 0.50,   # 50% dependent on observed severity
    'mnar_rate': 0.20   # 20% dependent on unobserved values
}
```

**Temporal Patterns:**

- +30% missing at night (11pm-7am)
- +15% missing during shift changes
- Critical patients have **LESS** missing (more monitoring)

### **6. Realistic Correlations**

#### **Multivariate Covariance Matrix:**

```python
CORRELATIONS = {
    'vitals': {
        'HR_RR': 0.45,      # HR ↔ RR: 0.45
        'MAP_HR': -0.30,    # MAP ↔ HR: -0.30
        'SpO2_RR': -0.35    # SpO2 ↔ RR: -0.35
    },
    'vitals_labs': {
        'HR_Lactate': 0.50,     # HR ↔ Lactate: 0.50
        'MAP_Lactate': -0.55    # MAP ↔ Lactate: -0.55
    },
    'labs': {
        'Creatinine_BUN': 0.80,  # Cr ↔ BUN: 0.80
        'Lactate_HCO3': -0.65    # Lactate ↔ HCO3: -0.65
    }
}
```

#### **Hierarchical Generation:**

1. Demographics → Condition
2. Condition → Vitals
3. Vitals → Labs

### **7. Documentation Quality (Shift Variation)**

```python
DOCUMENTATION_PATTERNS = {
    'by_shift': {
        'day':    {'completeness': 0.95, 'accuracy': 0.97},  # 7am-3pm
        'evening': {'completeness': 0.90, 'accuracy': 0.94}, # 3pm-11pm
        'night':   {'completeness': 0.85, 'accuracy': 0.90}  # 11pm-7am
    },
    'by_weekday': {
        'weekday': {'completeness': 0.93, 'accuracy': 0.96},
        'weekend': {'completeness': 0.87, 'accuracy': 0.92}
    }
}
```

**Effect:** Night shift and weekends have fewer recorded measurements.

### **8. Individual Variability (Anti-Overfitting)**

```python
INDIVIDUAL_VARIABILITY = {
    'baseline_offsets': {
        'Heart_Rate': 5,        # ±5 bpm per patient
        'MAP': 8,               # ±8 mmHg
        'Temperature': 0.3      # ±0.3°C
    },
    'deterioration_rate': {
        'mean': 0.3,
        'std': 0.1              # Each patient deteriorates differently
    },
    'progression_patterns': [
        'exponential',  # Rapid deterioration
        'sigmoid',      # S-curve
        'linear',       # Constant
        'step',         # Abrupt change
        'oscillating'   # Back and forth
    ]
}
```

**Goal:** Prevent all sepsis patients from having EXACTLY +25 bpm.

### **9. Stochastic Correlations (Anti-Overfitting)**

```python
STOCHASTIC_CORRELATIONS = {
    'correlation_variability': 0.10,  # ±10% in correlations
    'example': {
        'HR_RR_base': 0.45,
        'HR_RR_range': [0.35, 0.55]  # Varies per patient
    }
}
```

**Goal:** Prevent correlation always being 0.45 (memorizable).

### **10. Stochastic and Multivariate Mortality Model**

```python
MORTALITY_RISK_MODEL = {
    'base_features': {
        'age': 0.03,           # +3% risk per year
        'lactate': 0.25,       # +25% per mmol/L
        'map': -0.02,          # -2% per mmHg
        'spo2': -0.05          # -5% per %
    },
    'interactions': {
        'shock': ['map', 'hr', 'lactate'],      # Septic shock
        'sepsis': ['temp', 'wbc', 'lactate'],   # Sepsis
        'respiratory': ['spo2', 'rr', 'fio2']   # Respiratory failure
    },
    'nonlinear_terms': {
        'age_squared': 0.0005,
        'lactate_exp': 0.15,
        'spo2_inverse': -2.0
    },
    'temporal_effects': {
        'early_mortality': 0.3,    # First 12h
        'late_recovery': -0.2      # After 36h
    },
    'stochasticity': {
        'noise_std': 0.15,         # Gaussian noise
        'individual_variation': 0.2 # Individual variation
    }
}
```

**Goal:** Avoid trivial patterns like "if A > X then death".

### **11. Label Noise and Clinical Ambiguity**

Simulates the reality that clinical labels are not 100% deterministic:

```python
LABEL_NOISE_PARAMS = {
    'label_flip': {
        'false_positive': 0.01,  # 1% (0→1)
        'false_negative': 0.02   # 2% (1→0)
    },
    'censoring': {
        'transfer': 0.05,        # 5% transfer
        'ama': 0.02,             # 2% AMA discharge
        'lost_followup': 0.01    # 1% lost follow-up
    },
    'ambiguity': {
        'borderline': 0.08,      # 8% borderline cases
        'palliative': 0.03,      # Palliative care
        'withdrawal': 0.02       # Withdrawal of support
    },
    'timing_uncertainty': {
        'death': 6,              # ±6h uncertainty
        'discharge': 4,          # ±4h
        'documentation_delay': 0.15  # 15% delay
    }
}
```

**Goal:** Reflect clinical reality where labels have uncertainty and ambiguity.

### **12. Dataset Shift (Holdout Domain)**

```python
DATASET_SHIFT = {
    'holdout_fraction': 0.15,  # 15% of data
    'measurement_bias': {
        'Heart_Rate': +3,      # +3 bpm
        'MAP': -5,             # -5 mmHg
        'SpO2': -1,            # -1%
        'Lactate': +0.3        # +0.3 mmol/L
    },
    'population_shift': {
        'age_offset': +5,      # +5 years mean
        'male_ratio': +0.05,   # +5% males
        'severity': +0.10      # +10% more severe
    },
    'temporal_shift': {
        'measurements': -0.15,  # -15% measurements
        'missing': +0.20,       # +20% missing
        'los': -0.10           # -10% LOS
    }
}
```

**Goal:** Evaluate generalization and detect memorization of artifacts.

### **13. Demographic Diversity and Balancing**

```python
DIVERSITY_AND_BALANCING = {
    'age_distribution': {
        'young': 0.15,        # 18-40 years (15%)
        'middle': 0.30,       # 41-60 years (30%)
        'elderly': 0.40,      # 61-75 years (40%)
        'very_elderly': 0.15  # 76+ years (15%)
    },
    'target_mortality': 0.208,  # 20.8% (realistic)
    'data_augmentation': {
        'positive_cases': 0.50,  # +50% positive cases
        'perturbations': {
            'Heart_Rate': 5,     # ±5 bpm
            'MAP': 5,            # ±5 mmHg
            'Lactate': 0.5       # ±0.5 mmol/L
        }
    },
    'condition_diversity': {
        'sepsis': 0.40,          # 40% sepsis
        'heart_failure': 0.25,   # 25% heart failure
        'aki': 0.20,             # 20% acute kidney injury
        'respiratory': 0.10,     # 10% respiratory failure
        'multi_organ': 0.05      # 5% multi-organ failure
    },
    'conditional_probabilities': {
        'elderly_sepsis': 1.80,      # Elderly: +80% sepsis
        'elderly_hf': 2.20,          # Elderly: +120% HF
        'young_trauma': 1.50         # Young: +50% trauma
    }
}
```

**Goal:** Create balanced dataset with realistic demographic and clinical diversity.

---

## 🔧 Usage

### **Data Generation:**

```bash
# Generate complete dataset (24,327 episodes)
python scripts/generate_synthetic_mimic3.py

# Process data (create patient structure)
python scripts/process_synthetic_data.py
```

### **Complete Regeneration:**

```bash
# Regenerate data + train models + generate visualizations
./scripts/regenerate_data.sh
```

---

## ⚙️ Configuration Parameters

All parameters are centralized in `config/constants.py`:

```python
from config.constants import (
    NORMAL_RANGES,              # Normal vital sign ranges
    PATHOLOGICAL_RANGES,        # Pathological ranges
    CONDITION_IMPACTS,          # Condition impacts
    CIRCADIAN_PARAMS,           # Circadian patterns
    SLEEP_PARAMS,               # Sleep effects
    MEAL_PARAMS,                # Meal effects
    SIGNAL_NOISE_PARAMS,        # Signal noise
    TEMPORAL_VARIABILITY_PARAMS, # Temporal variability
    MISSINGNESS_PARAMS,         # Missing patterns
    CORRELATIONS,               # Correlations
    DOCUMENTATION_PATTERNS,     # Documentation quality
    MORTALITY_RISK_MODEL,       # Mortality model
    INDIVIDUAL_VARIABILITY,     # Individual variability
    STOCHASTIC_CORRELATIONS     # Stochastic correlations
)
```

---

## ✅ Validation and Quality

### **Quality Metrics:**

1. **Realistic Correlations:**
   - HR ↔ RR: 0.45 ± 0.10 ✅
   - MAP ↔ Lactate: -0.55 ± 0.10 ✅

2. **Distributions:**
   - HR: 60-100 bpm (normal), 40-180 bpm (pathological) ✅
   - Mortality: 20.8% ✅

3. **Temporal Patterns:**
   - Circadian: ±8% HR variation ✅
   - Sleep: -15% HR during 11pm-6am ✅

4. **Anti-Overfitting:**
   - Individual variability: ±5 bpm baseline ✅
   - Stochastic correlations: ±10% ✅
   - 5 different progression patterns ✅

### **Model Performance:**

Deep Learning model (LSTM + Focal Loss + Calibration) trained on this dataset:

- **AUROC:** 0.8638 ✅ ("Good" category for clinical use)
- **AUPRC:** 0.6396 ✅
- **Recall:** 96.81% ✅ (high sensitivity - detects 96.8% of deaths)
- **F1-Score:** 0.5114 ✅
- **Improvement vs Baseline:** +22.7% AUROC, +40.1% AUPRC

---

## 📚 Related Files

### **Scripts:**

- **Main Generator:** `scripts/generate_synthetic_mimic3.py` (84K)
- **Data Processor:** `scripts/process_synthetic_data.py` (13K)
- **Complete Regeneration:** `scripts/regenerate_data.sh`

### **Configuration:**

- **Constants:** `config/constants.py` - All centralized parameters

### **Training:**

- **Deep Learning:** `src/train_dl.py`
- **Baseline:** `src/train_baseline.py`
- **Utilities:** `src/calibration_utils.py`, `src/data_loader.py`

### **Results:**

- **Generated Dataset:** `data/in-hospital-mortality/`
- **Models:** `models/best_model_calibrated.keras`, `models/baseline_model.pkl`
- **Visualizations:** `results/plots/` (14 charts)
- **Reports:** `results/TECHNICAL_REPORT.md`

---
