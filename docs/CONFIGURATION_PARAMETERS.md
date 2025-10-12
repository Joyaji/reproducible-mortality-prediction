# Configuration Parameters - constants.py

Complete documentation of configurable parameters for the MIMIC-III synthetic data generator.

**File:** `config/constants.py`

---

## 📋 Table of Contents

1. [Normal and Pathological Value Ranges](#value-ranges)
2. [Clinical Condition Impacts](#condition-impacts)
3. [Circadian Patterns](#circadian-patterns)
4. [Sleep Simulation](#sleep-simulation)
5. [Meal Effects](#meal-effects)
6. [Noise and Entropy](#noise-and-entropy)
7. [Temporal Variability](#temporal-variability)
8. [Missingness](#missingness)
9. [Correlations](#correlations)
10. [Documentation Quality](#documentation-quality)
11. [Mortality Model](#mortality-model)
12. [Individual Variability](#individual-variability)
13. [Stochastic Correlations](#stochastic-correlations)
14. [Diversity and Balancing](#diversity-and-balancing)
15. [Dataset Shift](#dataset-shift)

---

## 1. Normal and Pathological Value Ranges

### **NORMAL_RANGES**

Defines normal values for each vital sign and lab test.

```python
NORMAL_RANGES = {
    # Vital Signs
    'Heart_Rate': (60, 100),          # bpm
    'Respiratory_Rate': (12, 20),     # rpm
    'MAP': (70, 100),                 # mmHg (Mean Arterial Pressure)
    'SpO2': (95, 100),                # % (Oxygen Saturation)
    'Temperature': (36.5, 37.5),      # °C
    
    # Lab Tests
    'Glucose': (70, 140),             # mg/dL
    'Lactate': (0.5, 2.0),            # mmol/L
    'Creatinine': (0.6, 1.2),         # mg/dL
    'BUN': (7, 20),                   # mg/dL (Blood Urea Nitrogen)
    'WBC': (4.5, 11.0),               # 10³/μL (White Blood Cells)
    'Hemoglobin': (12.0, 16.0),       # g/dL
    'Platelets': (150, 400),          # 10³/μL
    'Sodium': (135, 145),             # mEq/L
    'Potassium': (3.5, 5.0),          # mEq/L
    'HCO3': (22, 28)                  # mEq/L (Bicarbonate)
}
```

**How to use:**

- Values are generated within these ranges for healthy patients
- Deviations indicate pathology

### **PATHOLOGICAL_RANGES**

Defines pathological ranges (extreme possible values).

```python
PATHOLOGICAL_RANGES = {
    'Heart_Rate': (40, 180),
    'Respiratory_Rate': (6, 40),
    'MAP': (40, 140),
    'SpO2': (70, 100),
    'Temperature': (35.0, 41.0),
    'Glucose': (40, 400),
    'Lactate': (0.2, 15.0),
    'Creatinine': (0.3, 8.0),
    'BUN': (5, 100),
    'WBC': (1.0, 30.0),
    'Hemoglobin': (6.0, 20.0),
    'Platelets': (20, 600),
    'Sodium': (120, 160),
    'Potassium': (2.5, 7.0),
    'HCO3': (10, 40)
}
```

**How to use:**

- Extreme values for critical patients
- Realistic physiological limits

---

## 2. Clinical Condition Impacts

### **CONDITION_IMPACTS**

Defines how each clinical condition affects vital signs and labs.

```python
CONDITION_IMPACTS = {
    'sepsis': {
        'Heart_Rate': +25,        # Tachycardia
        'Temperature': +1.5,      # Fever
        'Respiratory_Rate': +8,   # Tachypnea
        'WBC': +10.0,             # Leukocytosis
        'Lactate': +3.0,          # Hyperlactatemia
        'MAP': -15                # Hypotension
    },
    'heart_failure': {
        'Heart_Rate': +15,
        'Respiratory_Rate': +6,
        'SpO2': -5,
        'BUN': +15,
        'Creatinine': +0.8
    },
    'aki': {  # Acute Kidney Injury
        'Creatinine': +2.5,
        'BUN': +30,
        'Potassium': +1.5,
        'HCO3': -5
    },
    'respiratory_failure': {
        'SpO2': -15,
        'Respiratory_Rate': +12,
        'Heart_Rate': +20,
        'Lactate': +2.0
    },
    'multi_organ_failure': {
        'Heart_Rate': +30,
        'MAP': -25,
        'SpO2': -20,
        'Lactate': +5.0,
        'Creatinine': +3.0,
        'WBC': +15.0
    }
}
```

**How to use:**

- Values are **added** to normal baselines
- Multiple conditions can be combined
- Example: Patient with sepsis will have HR = baseline + 25 bpm

---

## 3. Circadian Patterns

### **CIRCADIAN_PARAMS**

Defines natural variations throughout the day (24h cycle).

```python
CIRCADIAN_PARAMS = {
    'Heart_Rate': {
        'peak_hour': 14,      # Peak at 2pm (afternoon)
        'trough_hour': 3,     # Minimum at 3am (early morning)
        'amplitude': 0.08     # ±8% variation
    },
    'Respiratory_Rate': {
        'peak_hour': 16,
        'trough_hour': 4,
        'amplitude': 0.06     # ±6%
    },
    'MAP': {
        'peak_hour': 10,
        'trough_hour': 2,
        'amplitude': 0.10     # ±10%
    },
    'Temperature': {
        'peak_hour': 18,      # Peak at 6pm (evening)
        'trough_hour': 4,     # Minimum at 4am (early morning)
        'amplitude': 0.5      # ±0.5°C (absolute)
    }
}
```

**How it works:**

- Sinusoidal pattern with 24h period
- `amplitude`: For numeric features = % variation, for temperature = absolute °C
- Simulates natural circadian rhythm

**Example:**

```python
# HR at 2pm (peak): baseline * (1 + 0.08) = baseline * 1.08
# HR at 3am (minimum): baseline * (1 - 0.08) = baseline * 0.92
```

---

## 4. Sleep Simulation

### **SLEEP_PARAMS**

Defines sleep effect on vital signs.

```python
SLEEP_PARAMS = {
    'start_hour': 23,         # Sleep start: 11pm
    'duration_hours': 7,      # Duration: 7 hours (11pm-6am)
    'effects': {
        'Heart_Rate': -0.15,      # -15% during sleep
        'Respiratory_Rate': -0.10, # -10%
        'MAP': -0.08,             # -8%
        'Temperature': -0.3       # -0.3°C
    }
}
```

**How it works:**

- Reduction of vital signs during sleep period
- Negative values = reduction
- Simulates resting state

**Example:**

```python
# HR during sleep (11pm-6am): baseline * (1 - 0.15) = baseline * 0.85
# HR awake: baseline (no change)
```

---

## 5. Meal Effects

### **MEAL_PARAMS**

Defines meal effects on vital signs and glucose.

```python
MEAL_PARAMS = {
    'meal_times': [7, 12, 18],  # Breakfast (7am), Lunch (12pm), Dinner (6pm)
    'duration_hours': 2,         # Effect duration: 2h
    'effects': {
        'Glucose': +0.30,        # +30% glucose post-meal
        'Heart_Rate': +0.05      # +5% HR (digestion)
    }
}
```

**How it works:**

- Glucose spike after meals
- Effect lasts 2 hours
- Simulates metabolic response

---

## 6. Noise and Entropy

### **ENTROPY_PARAMS**

Defines entropy and unpredictability in signals.

```python
ENTROPY_PARAMS = {
    'base_entropy': 0.15,        # 15% base entropy
    'temporal_correlation': 0.7,  # 70% temporal correlation
    'random_events': {
        'probability': 0.05,      # 5% chance per timestep
        'magnitude': 0.20         # ±20% variation
    }
}
```

**How it works:**

- `base_entropy`: Base random noise
- `temporal_correlation`: How much current value depends on previous (0-1)
- `random_events`: Stochastic events (e.g., patient movement)

### **SIGNAL_NOISE_PARAMS**

Defines signal-specific noise (SNR - Signal-to-Noise Ratio).

```python
SIGNAL_NOISE_PARAMS = {
    'Heart_Rate': 0.02,        # ±2% noise
    'Respiratory_Rate': 0.03,  # ±3%
    'MAP': 0.04,               # ±4%
    'SpO2': 0.005,             # ±0.5%
    'Temperature': 0.008,      # ±0.8%
    'Glucose': 0.06,           # ±6%
    'Lactate': 0.05,           # ±5%
    'Creatinine': 0.04,        # ±4%
    'BUN': 0.05,               # ±5%
    'WBC': 0.08,               # ±8%
    'Hemoglobin': 0.03,        # ±3%
    'Platelets': 0.10,         # ±10%
    'Sodium': 0.02,            # ±2%
    'Potassium': 0.03,         # ±3%
    'HCO3': 0.04               # ±4%
}
```

**How to use:**

- Values represent % of Gaussian noise
- Noise is applied to each measurement
- Simulates measurement variability

---

## 7. Temporal Variability

### **TEMPORAL_VARIABILITY_PARAMS**

Defines advanced temporal variations.

```python
TEMPORAL_VARIABILITY_PARAMS = {
    'time_warping': {
        'enabled': True,
        'max_shift': 0.10      # ±10% variation in spacing
    },
    'dropout': {
        'rate': 0.10,          # 10% missing measurements
        'burst_probability': 0.02  # 2% chance of burst (multiple missing)
    },
    'artifacts': {
        'spike_probability': 0.03,  # 3% chance of spike
        'spike_magnitude': 2.0      # 2x normal value
    }
}
```

**How it works:**

- `time_warping`: Varies spacing between measurements
- `dropout`: Simulates equipment failures
- `artifacts`: Isolated spikes (movement, disconnection)

---

## 8. Missingness

### **MISSINGNESS_PARAMS**

Defines missing data patterns (MCAR, MAR, MNAR).

```python
MISSINGNESS_PARAMS = {
    'mcar_rate': 0.30,  # 30% Missing Completely At Random
    'mar_rate': 0.50,   # 50% Missing At Random (dependent on severity)
    'mnar_rate': 0.20,  # 20% Missing Not At Random
    
    'temporal_patterns': {
        'night_increase': 0.30,     # +30% missing at night (11pm-7am)
        'shift_change': 0.15,       # +15% at shift changes
        'weekend_increase': 0.10    # +10% on weekends
    },
    
    'severity_dependence': {
        'critical': -0.20,  # Critical patients: -20% missing (more monitoring)
        'stable': +0.15     # Stable patients: +15% missing
    }
}
```

**Types of Missingness:**

- **MCAR:** Completely at random
- **MAR:** Depends on observed variables (severity)
- **MNAR:** Depends on unobserved variables

---

## 9. Correlations

### **CORRELATIONS**

Defines correlations between variables.

```python
CORRELATIONS = {
    'vitals': {
        'HR_RR': 0.45,      # HR ↔ RR: positive correlation
        'MAP_HR': -0.30,    # MAP ↔ HR: negative correlation
        'SpO2_RR': -0.35,   # SpO2 ↔ RR: negative correlation
        'Temp_HR': 0.25     # Temp ↔ HR: positive correlation
    },
    'vitals_labs': {
        'HR_Lactate': 0.50,     # HR ↔ Lactate: 0.50
        'MAP_Lactate': -0.55,   # MAP ↔ Lactate: -0.55
        'SpO2_Lactate': -0.40,  # SpO2 ↔ Lactate: -0.40
        'RR_Lactate': 0.35      # RR ↔ Lactate: 0.35
    },
    'labs': {
        'Creatinine_BUN': 0.80,  # Cr ↔ BUN: strong correlation
        'Lactate_HCO3': -0.65,   # Lactate ↔ HCO3: negative
        'WBC_Temp': 0.40,        # WBC ↔ Temp: positive
        'Glucose_Lactate': 0.30  # Glucose ↔ Lactate: positive
    }
}
```

**How it works:**

- Values between -1 and +1
- Positive: variables increase together
- Negative: one increases, other decreases
- Generation uses multivariate covariance matrix

---

## 10. Documentation Quality

### **DOCUMENTATION_PATTERNS**

Defines variation in documentation quality by shift.

```python
DOCUMENTATION_PATTERNS = {
    'by_shift': {
        'day': {
            'completeness': 0.95,  # 95% of measurements recorded
            'accuracy': 0.97       # 97% accuracy
        },
        'evening': {
            'completeness': 0.90,
            'accuracy': 0.94
        },
        'night': {
            'completeness': 0.85,  # Worst: less staff
            'accuracy': 0.90
        }
    },
    'by_weekday': {
        'weekday': {
            'completeness': 0.93,
            'accuracy': 0.96
        },
        'weekend': {
            'completeness': 0.87,  # Worst: reduced team
            'accuracy': 0.92
        }
    }
}
```

**Shifts:**

- **Day:** 7am-3pm (best documentation)
- **Evening:** 3pm-11pm (good)
- **Night:** 11pm-7am (worst - less staff)

**Effects:**

- `completeness`: Probability of measurement being recorded
- `accuracy`: Value precision (additional noise if low)

---

## 11. Mortality Model

### **MORTALITY_RISK_MODEL**

Defines how to calculate mortality risk.

```python
MORTALITY_RISK_MODEL = {
    'base_features': {
        'age': 0.03,           # +3% risk per year
        'lactate': 0.25,       # +25% per mmol/L
        'map': -0.02,          # -2% per mmHg (high MAP protects)
        'spo2': -0.05,         # -5% per % (high SpO2 protects)
        'creatinine': 0.15,    # +15% per mg/dL
        'wbc': 0.02            # +2% per 10³/μL
    },
    'interactions': {
        'shock': {
            'features': ['map', 'hr', 'lactate'],
            'weight': 0.30     # Septic shock
        },
        'sepsis': {
            'features': ['temp', 'wbc', 'lactate'],
            'weight': 0.25
        },
        'respiratory': {
            'features': ['spo2', 'rr'],
            'weight': 0.20
        }
    },
    'nonlinear_terms': {
        'age_squared': 0.0005,     # Risk increases exponentially with age
        'lactate_exp': 0.15,       # High lactate = exponential risk
        'spo2_inverse': -2.0       # Low SpO2 = very high risk
    },
    'temporal_effects': {
        'early_mortality': {
            'window': 12,          # First 12h
            'weight': 0.30         # +30% risk
        },
        'late_recovery': {
            'window': 36,          # After 36h
            'weight': -0.20        # -20% risk (survived critical period)
        }
    },
    'stochasticity': {
        'noise_std': 0.15,         # Gaussian noise
        'individual_variation': 0.2 # Individual variation
    }
}
```

**How it works:**

1. Calculate base score with linear features
2. Add interactions (e.g., shock = low MAP + high HR + high Lactate)
3. Add non-linear terms
4. Apply temporal effects
5. Add stochastic noise
6. Convert to probability with sigmoid

---

## 12. Individual Variability

### **INDIVIDUAL_VARIABILITY**

Defines variation between patients (anti-overfitting).

```python
INDIVIDUAL_VARIABILITY = {
    'baseline_offsets': {
        'Heart_Rate': 5,        # ±5 bpm unique per patient
        'MAP': 8,               # ±8 mmHg
        'Temperature': 0.3,     # ±0.3°C
        'Respiratory_Rate': 2   # ±2 rpm
    },
    'deterioration_rate': {
        'mean': 0.3,
        'std': 0.1              # Rate varies between patients
    },
    'condition_sensitivity': {
        'mean': 1.0,
        'std': 0.2              # Individual response to conditions
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

**Goal:**

- Prevent all patients from being identical
- Each patient has unique baseline
- Diverse progression patterns

---

## 13. Stochastic Correlations

### **STOCHASTIC_CORRELATIONS**

Defines variation in correlations (anti-overfitting).

```python
STOCHASTIC_CORRELATIONS = {
    'correlation_variability': 0.10,  # ±10% in correlations
    'covariance_noise': 0.10          # Noise in covariance matrix
}
```

**How it works:**

- Base correlation: HR ↔ RR = 0.45
- With variability: HR ↔ RR varies between 0.35-0.55 per patient
- Prevents memorization of fixed correlations

---

## 14. Diversity and Balancing

### **DIVERSITY_AND_BALANCING**

Defines demographic distribution and balancing.

```python
DIVERSITY_AND_BALANCING = {
    'age_distribution': {
        'young': 0.15,        # 18-40 years (15%)
        'middle': 0.30,       # 41-60 years (30%)
        'elderly': 0.40,      # 61-75 years (40%)
        'very_elderly': 0.15  # 76+ years (15%)
    },
    'target_mortality': 0.208,  # 20.8%
    'data_augmentation': {
        'positive_cases': 0.50,  # +50% positive cases
        'perturbations': {
            'Heart_Rate': 5,
            'MAP': 5,
            'Lactate': 0.5
        }
    },
    'condition_diversity': {
        'sepsis': 0.40,
        'heart_failure': 0.25,
        'aki': 0.20,
        'respiratory': 0.10,
        'multi_organ': 0.05
    }
}
```

---

## 15. Dataset Shift

### **DATASET_SHIFT**

Defines holdout domain with different distribution.

```python
DATASET_SHIFT = {
    'holdout_fraction': 0.15,  # 15% of data
    'measurement_bias': {
        'Heart_Rate': +3,      # +3 bpm bias
        'MAP': -5,
        'SpO2': -1,
        'Lactate': +0.3
    },
    'population_shift': {
        'age_offset': +5,      # Older population
        'male_ratio': +0.05,
        'severity': +0.10
    },
    'temporal_shift': {
        'measurements': -0.15,  # Fewer measurements
        'missing': +0.20,       # More missing
        'los': -0.10           # Shorter length of stay
    }
}
```

**Goal:**

- Evaluate model generalization
- Detect overfitting to specific artifacts

---

## 🔧 How to Modify Parameters

### **Example 1: Increase Noise**

```python
# In config/constants.py
SIGNAL_NOISE_PARAMS = {
    'Heart_Rate': 0.05,  # Increase from 0.02 to 0.05 (5% noise)
    # ...
}
```

### **Example 2: Change Mortality Rate**

```python
DIVERSITY_AND_BALANCING = {
    'target_mortality': 0.25,  # Increase from 20.8% to 25%
    # ...
}
```

### **Example 3: Add New Condition**

```python
CONDITION_IMPACTS = {
    # ...
    'pneumonia': {
        'Temperature': +2.0,
        'Respiratory_Rate': +10,
        'SpO2': -10,
        'WBC': +8.0
    }
}
```

---

## 📚 References

- **File:** `config/constants.py`
- **Generator:** `scripts/generate_synthetic_mimic3.py`
- **General Documentation:** `docs/SYNTHETIC_DATA_GENERATOR.md`

---
