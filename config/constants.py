"""
Constants for MIMIC-III synthetic data generation
"""

# Normal ranges (for validation)
NORMAL_RANGES = {
    'Heart Rate': (60, 100),
    'Systolic blood pressure': (90, 140),
    'Diastolic blood pressure': (60, 90),
    'Respiratory Rate': (12, 20),
    'Temperature': (36.5, 37.5),
    'SpO2': (95, 100),
    'Lactate': (0.5, 2.0),
    'Creatinine': (0.7, 1.3),
    'White blood cell count': (4.0, 11.0),
    'Hemoglobin': (12.0, 16.0),
    'Glucose': (70, 110),
    'Potassium': (3.5, 5.0),
    'Sodium': (135, 145),
    'Bicarbonate': (22, 28),
    'Blood Urea Nitrogen': (7, 20)
}

# Pathological ranges (for generation - allows extreme values)
PATHOLOGICAL_RANGES = {
    'Heart Rate': (40, 200),
    'Systolic blood pressure': (60, 220),
    'Diastolic blood pressure': (40, 140),
    'Respiratory Rate': (8, 40),
    'Temperature': (35.0, 41.0),
    'SpO2': (70, 100),
    'Lactate': (0.5, 20.0),
    'Creatinine': (0.5, 10.0),
    'White blood cell count': (2.0, 50.0),
    'Hemoglobin': (6.0, 18.0),
    'Glucose': (40, 600),
    'Potassium': (2.5, 7.0),
    'Sodium': (120, 160),
    'Bicarbonate': (10, 40),
    'Blood Urea Nitrogen': (5, 150)
}

# Condition impacts (multiplicators)
CONDITION_IMPACTS = {
    'sepsis': {
        'Lactate': 2.5,
        'White blood cell count': 1.8,
        'Heart Rate': 1.3,
        'Systolic blood pressure': 0.8,
        'Temperature': 1.15,
        'Creatinine': 1.4
    },
    'aki': {  # Acute Kidney Injury
        'Creatinine': 2.0,
        'Potassium': 1.3,
        'Blood Urea Nitrogen': 2.2,
        'Bicarbonate': 0.8
    },
    'heart_failure': {
        'Heart Rate': 1.2,
        'Systolic blood pressure': 0.85,
        'SpO2': 0.92,
        'Blood Urea Nitrogen': 1.3
    },
    'copd': {
        'SpO2': 0.88,
        'Respiratory Rate': 1.25,
        'Bicarbonate': 1.15
    },
    'diabetes': {
        'Glucose': 1.8,
        'Creatinine': 1.2
    }
}

# Circadian parameters (variation over the day)
CIRCADIAN_PARAMS = {
    'Heart Rate': {
        'peak_hour': 15,      # 3 PM
        'trough_hour': 3,     # 3 AM
        'amplitude': 0.10     # ±10%
    },
    'Systolic blood pressure': {
        'peak_hour': 10,      # 10 AM
        'trough_hour': 2,     # 2 AM
        'amplitude': 0.08     # ±8%
    },
    'Diastolic blood pressure': {
        'peak_hour': 10,      # 10 AM
        'trough_hour': 2,     # 2 AM
        'amplitude': 0.08     # ±8%
    },
    'Temperature': {
        'peak_hour': 16,      # 4 PM
        'trough_hour': 4,     # 4 AM
        'amplitude': 0.5      # ±0.5°C (absolute)
    },
    'Respiratory Rate': {
        'peak_hour': 14,      # 2 PM
        'trough_hour': 3,     # 3 AM
        'amplitude': 0.05     # ±5%
    }
}

# Sleep parameters
SLEEP_PARAMS = {
    'start_hour': 23,
    'duration_hours': 7,
    'reductions': {
        'Heart Rate': 0.15,           # -15%
        'Systolic blood pressure': 0.12,  # -12%
        'Diastolic blood pressure': 0.12, # -12%
        'Respiratory Rate': 0.08      # -8%
    }
}

# Meal parameters
MEAL_PARAMS = {
    'times': [7, 12, 18],      # 7 AM, 12 PM, 6 PM
    'effect_duration': 1.5,    # hours
    'effects': {
        'Glucose': 0.40,       # +40%
        'Heart Rate': 0.08,    # +8%
        'Lactate': 0.12        # +12%
    }
}

# Entropy parameters
ENTROPY_PARAMS = {
    'temporal_noise': 0.15,      # ±15% temporal noise
    'measurement_noise': 0.05,   # ±5% measurement noise
    'spike_probability': 0.10,   # 10% chance of spike
    'event_probability': 0.25,   # 25% chance of stochastic event
    'chaos_probability': 0.00    # Disabled
}

# Signal-specific noise parameters (SNR realistic)
SIGNAL_NOISE_PARAMS = {
    # Vital signals - noise as % of value
    'Heart Rate': 0.02,                    # ±2% (high precision)
    'Systolic blood pressure': 0.03,       # ±3% (NIBP has more noise)
    'Diastolic blood pressure': 0.03,      # ±3%
    'Respiratory Rate': 0.04,              # ±4% (more variable)
    'Temperature': 0.005,                  # ±0.5% (very precise)
    'SpO2': 0.01,                          # ±1% (saturation has low noise)
    
    # Lab tests - noise as % of value
    'Lactate': 0.05,                       # ±5%
    'Creatinine': 0.03,                    # ±3%
    'White blood cell count': 0.08,        # ±8% (cell count more variable)
    'Hemoglobin': 0.02,                    # ±2%
    'Glucose': 0.04,                       # ±4%
    'Potassium': 0.03,                     # ±3%
    'Sodium': 0.02,                        # ±2%
    'Bicarbonate': 0.04,                   # ±4%
    'Blood Urea Nitrogen': 0.05            # ±5%
}

# Temporal variability parameters
TEMPORAL_VARIABILITY_PARAMS = {
    'time_dropout_prob': 0.10,             # 10% of timesteps without measurement
    'time_jitter_range': 0.10,             # ±10% variation in temporal spacing
    'artifact_probability': 0.03,          # 3% chance of artifact (isolated spike)
    'artifact_magnitude_range': (0.2, 0.5) # Artifacts of 20-50% of value
}

# Missingness parameters (MCAR/MAR/MNAR)
MISSINGNESS_PARAMS = {
    # Missingness mechanisms
    'mechanisms': {
        'MCAR': 0.30,   # 30% Missing Completely At Random
        'MAR': 0.50,    # 50% Missing At Random (dependent on observed variables)
        'MNAR': 0.20    # 20% Missing Not At Random (dependent on non-observed values)
    },
    
    # CLINICAL REALITY: Patients with more severity have MORE measurements
    # (opposite of what many generators do)
    'severity_effect': {
        'direction': 'inverse',  # 'inverse' = more severe → less missing
        'baseline_prob': 0.15,   # Base missing probability (stable patient)
        'min_prob': 0.03,        # Minimum missing (critical patient)
        'max_prob': 0.25,        # Maximum missing (stable patient/recovering)
        'steepness': 3.0         # Steepness of the sigmoid (controls transition)
    },
    
    # Missingness specific by measurement type
    'measurement_specific': {
        # Vital signals - continuous monitoring in severe patients
        'vital_signs': {
            'baseline_missing': 0.08,
            'severity_multiplier': 0.5  # Severe patients have 50% less missing
        },
        # Labs - collected more frequently in severe patients
        'labs': {
            'baseline_missing': 0.20,
            'severity_multiplier': 0.3  # Severe patients have 70% less missing
        }
    },
    
    # Temporal patterns of missingness
    'temporal_patterns': {
        'night_increase': 1.3,      # +30% missing at night (23h-6h)
        'weekend_increase': 1.2,    # +20% missing at weekends
        'handoff_increase': 1.15    # +15% missing at handoffs (7h, 15h, 23h)
    }
}

# Realistic correlations between variables
# Based on MIMIC-III literature and clinical physiology
CORRELATIONS = {
    # Correlations between vital signals
    'vital_signs': {
        # Pairs of variables and expected correlation
        ('Heart Rate', 'Respiratory Rate'): 0.45,      # Tachycardia ↔ Tachypnea
        ('Heart Rate', 'Systolic blood pressure'): -0.30,  # FC ↑ → PA ↓ (shock)
        ('Systolic blood pressure', 'Diastolic blood pressure'): 0.85,  # PA systolic ↔ diastolic
        ('Heart Rate', 'Temperature'): 0.35,           # Fever → Tachycardia
        ('SpO2', 'Respiratory Rate'): -0.40,           # Hypoxemia → Tachypnea
        ('Systolic blood pressure', 'SpO2'): 0.25,     # Hypotension ↔ Hypoxemia
    },
    
    # Correlations between labs
    'labs': {
        ('Creatinine', 'Blood Urea Nitrogen'): 0.80,   # Renal function
        ('Lactate', 'Bicarbonate'): -0.65,             # Lactic acidosis
        ('Sodium', 'Potassium'): 0.15,                 # Electrolytes
        ('Hemoglobin', 'White blood cell count'): -0.20,  # Anemia ↔ Leucocitose
        ('Glucose', 'Lactate'): 0.30,                  # Metabolismo
    },
    
    # Correlations between vital signals and labs
    'vital_lab': {
        ('Heart Rate', 'Lactate'): 0.50,               # Tachycardia ↔ Hyperlactatemia
        ('Systolic blood pressure', 'Lactate'): -0.55, # Hypotension ↔ Lactate ↑
        ('Systolic blood pressure', 'Creatinine'): -0.35,  # Hypotension → AKI
        ('Temperature', 'White blood cell count'): 0.40,   # Fever ↔ Leucocitosis
        ('SpO2', 'Lactate'): -0.45,                    # Hypoxemia ↔ Lactate ↑
    }
}

# Hierarchical conditional generation
# Order: Demographics → Condition → Vitals → Labs
CONDITIONAL_GENERATION = {
    # Conditional probabilities: P(Condition | Demographics)
    'condition_given_demographics': {
        # P(sepsis | age, sex)
        'sepsis': {
            'age_effect': {
                'baseline': 0.15,
                'age_>70': 1.8,      # +80% in elderly
                'age_<40': 0.6       # -40% in young
            },
            'sex_effect': {
                'M': 1.1,            # +10% in men
                'F': 0.9
            }
        },
        'aki': {
            'age_effect': {
                'baseline': 0.10,
                'age_>70': 2.0,
                'age_<40': 0.5
            },
            'sex_effect': {
                'M': 1.2,
                'F': 0.8
            }
        },
        'heart_failure': {
            'age_effect': {
                'baseline': 0.12,
                'age_>70': 2.2,
                'age_<40': 0.3
            },
            'sex_effect': {
                'M': 1.3,
                'F': 0.7
            }
        }
    },
    
    # Conditional distributions: P(Vitals | Condition)
    # Mean and standard deviation for each condition
    'vitals_given_condition': {
        'sepsis': {
            'Heart Rate': {'mean_shift': +25, 'std_multiplier': 1.5},
            'Systolic blood pressure': {'mean_shift': -20, 'std_multiplier': 1.3},
            'Diastolic blood pressure': {'mean_shift': -10, 'std_multiplier': 1.2},
            'Respiratory Rate': {'mean_shift': +6, 'std_multiplier': 1.4},
            'Temperature': {'mean_shift': +1.5, 'std_multiplier': 1.2},
            'SpO2': {'mean_shift': -5, 'std_multiplier': 1.3}
        },
        'aki': {
            'Heart Rate': {'mean_shift': +10, 'std_multiplier': 1.2},
            'Systolic blood pressure': {'mean_shift': -10, 'std_multiplier': 1.1},
            'Diastolic blood pressure': {'mean_shift': -5, 'std_multiplier': 1.1},
            'Respiratory Rate': {'mean_shift': +2, 'std_multiplier': 1.1},
            'Temperature': {'mean_shift': 0, 'std_multiplier': 1.0},
            'SpO2': {'mean_shift': -2, 'std_multiplier': 1.1}
        },
        'heart_failure': {
            'Heart Rate': {'mean_shift': +15, 'std_multiplier': 1.3},
            'Systolic blood pressure': {'mean_shift': -15, 'std_multiplier': 1.2},
            'Diastolic blood pressure': {'mean_shift': -8, 'std_multiplier': 1.2},
            'Respiratory Rate': {'mean_shift': +4, 'std_multiplier': 1.3},
            'Temperature': {'mean_shift': 0, 'std_multiplier': 1.0},
            'SpO2': {'mean_shift': -8, 'std_multiplier': 1.4}
        },
        'healthy': {
            'Heart Rate': {'mean_shift': 0, 'std_multiplier': 1.0},
            'Systolic blood pressure': {'mean_shift': 0, 'std_multiplier': 1.0},
            'Diastolic blood pressure': {'mean_shift': 0, 'std_multiplier': 1.0},
            'Respiratory Rate': {'mean_shift': 0, 'std_multiplier': 1.0},
            'Temperature': {'mean_shift': 0, 'std_multiplier': 1.0},
            'SpO2': {'mean_shift': 0, 'std_multiplier': 1.0}
        }
    },
    
    # P(Labs | Condition, Vitals)
    'labs_given_condition': {
        'sepsis': {
            'Lactate': {'mean_shift': +3.0, 'std_multiplier': 2.0},
            'White blood cell count': {'mean_shift': +8.0, 'std_multiplier': 1.8},
            'Creatinine': {'mean_shift': +0.8, 'std_multiplier': 1.5},
            'Blood Urea Nitrogen': {'mean_shift': +15, 'std_multiplier': 1.6},
            'Bicarbonate': {'mean_shift': -5, 'std_multiplier': 1.3},
            'Glucose': {'mean_shift': +40, 'std_multiplier': 1.5}
        },
        'aki': {
            'Lactate': {'mean_shift': +1.0, 'std_multiplier': 1.3},
            'White blood cell count': {'mean_shift': +2.0, 'std_multiplier': 1.2},
            'Creatinine': {'mean_shift': +2.5, 'std_multiplier': 2.0},
            'Blood Urea Nitrogen': {'mean_shift': +30, 'std_multiplier': 2.0},
            'Bicarbonate': {'mean_shift': -3, 'std_multiplier': 1.2},
            'Potassium': {'mean_shift': +0.8, 'std_multiplier': 1.4}
        },
        'heart_failure': {
            'Lactate': {'mean_shift': +0.5, 'std_multiplier': 1.2},
            'White blood cell count': {'mean_shift': +1.0, 'std_multiplier': 1.1},
            'Creatinine': {'mean_shift': +0.5, 'std_multiplier': 1.3},
            'Blood Urea Nitrogen': {'mean_shift': +10, 'std_multiplier': 1.4},
            'Sodium': {'mean_shift': -3, 'std_multiplier': 1.2}
        },
        'healthy': {
            # No shifts for healthy patients
        }
    }
}

# Covariance matrix for multivariate generation
# Simplified: use correlations above to build matrix
COVARIANCE_STRUCTURE = {
    'use_correlated_generation': True,
    'correlation_strength': 0.7,  # Scale factor for correlations (0-1)
}

# Noisy labels and clinical ambiguity
# REALITY: Labels are not 100% deterministic
LABEL_NOISE_PARAMS = {
    # Label noise (flip of labels)
    'mortality_label_noise': {
        'flip_0_to_1': 0.01,  # 1% false positives (survived but marked as dead)
        'flip_1_to_0': 0.02,  # 2% false negatives (died but marked as survived)
        'reasons_0_to_1': [
            'Documentation error',
            'Death after transfer not registered',
            'Confusion with another patient'
        ],
        'reasons_1_to_0': [
            'Death in another hospital after transfer',
            'Palliative care (expected death but not registered)',
            'Coding error'
        ]
    },
    
    # Censoring (patients with uncertain outcome)
    'censoring': {
        'transfer_rate': 0.05,        # 5% transferred to another hospital
        'discharge_ama_rate': 0.02,   # 2% discharge against medical advice
        'lost_followup_rate': 0.01,   # 1% loss of follow-up
        
        # Conditional mortality probabilities after censoring
        'transfer_mortality_prob': 0.30,      # 30% die after transfer
        'discharge_ama_mortality_prob': 0.15, # 15% die after AMA discharge
        'lost_followup_mortality_prob': 0.20  # 20% die (unknown)
    },
    
    # Clinical ambiguity (borderline cases)
    'clinical_ambiguity': {
        'borderline_cases_rate': 0.08,  # 8% borderline cases
        'borderline_scenarios': {
            'comfort_care': {
                'probability': 0.35,  # 35% of borderline cases
                'description': 'Transition to palliative care',
                'label_uncertainty': 0.5  # 50% chance of flip
            },
            'withdrawal_of_care': {
                'probability': 0.25,  # 25% of borderline cases
                'description': 'Withdrawal of care',
                'label_uncertainty': 0.4
            },
            'brain_death': {
                'probability': 0.20,  # 20% of borderline cases
                'description': 'Morte cerebral (tempo de registro varia)',
                'label_uncertainty': 0.3
            },
            'transfer_to_hospice': {
                'probability': 0.20,  # 20% of borderline cases
                'description': 'Transferência para hospice',
                'label_uncertainty': 0.6
            }
        }
    },
    
    # Timing uncertainty (incerteza temporal)
    'timing_uncertainty': {
        'death_time_jitter_hours': 6,  # ±6h in registered death time
        'discharge_time_jitter_hours': 4,  # ±4h in registered discharge time
        'documentation_delay_prob': 0.15,  # 15% have documentation delay
        'documentation_delay_hours': (2, 24)  # 2-24h delay
    }
}

# Realistic documentation patterns
DOCUMENTATION_PATTERNS = {
    # Quality varies by shift
    'by_shift': {
        'day': {'completeness': 0.95, 'accuracy': 0.97},    # Day: best documentation
        'evening': {'completeness': 0.90, 'accuracy': 0.94}, # Evening: good
        'night': {'completeness': 0.85, 'accuracy': 0.90}    # Night: worse
    },
    
    # Quality varies by day of the week
    'by_weekday': {
        'weekday': {'completeness': 0.93, 'accuracy': 0.96},
        'weekend': {'completeness': 0.87, 'accuracy': 0.92}  # Weekend: worse
    }
}

# Mortality risk model (STOCHASTIC and MULTIVARIATE)
# AVOID simple patterns (e.g: "if A > X then death")
MORTALITY_RISK_MODEL = {
    # Logistic model: P(death) = sigmoid(score)
    # score = non-linear combination of multiple features
    
    # Base weights by feature (linear)
    'feature_weights': {
        # Demographics
        'age': 0.03,           # +3% per year above 50
        'male': 0.15,          # +15% if male
        
        # Vital signs (non-linear)
        'heart_rate': {
            'low': 0.40,       # HR < 50: +40%
            'high': 0.35,      # HR > 120: +35%
            'optimal': -0.10   # HR 60-80: -10%
        },
        'systolic_bp': {
            'low': 0.50,       # PA < 90: +50%
            'high': 0.20,      # PA > 160: +20%
            'optimal': -0.15   # PA 110-130: -15%
        },
        'spo2': {
            'critical': 0.60,  # SpO2 < 85: +60%
            'low': 0.30,       # SpO2 85-92: +30%
            'optimal': -0.10   # SpO2 > 95: -10%
        },
        'temperature': {
            'hypothermia': 0.45,  # Temp < 36: +45%
            'fever': 0.25,        # Temp > 38.5: +25%
            'hyperthermia': 0.50  # Temp > 40: +50%
        },
        
        # Labs (non-linear)
        'lactate': {
            'normal': -0.05,      # Lac < 2: -5%
            'elevated': 0.30,     # Lac 2-4: +30%
            'high': 0.60,         # Lac > 4: +60%
        },
        'creatinine': {
            'normal': -0.05,      # Cr < 1.5: -5%
            'elevated': 0.25,     # Cr 1.5-3: +25%
            'high': 0.50,         # Cr > 3: +50%
        },
        'wbc': {
            'low': 0.30,          # WBC < 4: +30%
            'high': 0.25,         # WBC > 15: +25%
            'very_high': 0.40     # WBC > 25: +40%
        }
    },
    
    # Feature interactions (non-linear)
    'feature_interactions': {
        # Shock (low PA + high HR + high lactate)
        'shock': {
            'features': ['systolic_bp', 'heart_rate', 'lactate'],
            'conditions': ['low', 'high', 'high'],
            'weight': 0.80,  # +80% if all conditions present
            'partial_weight': 0.30  # +30% if 2 of 3 present
        },
        
        # Sepsis (Temp abnormal + WBC abnormal + Lactate high)
        'sepsis_like': {
            'features': ['temperature', 'wbc', 'lactate'],
            'conditions': ['fever', 'high', 'high'],
            'weight': 0.70,
            'partial_weight': 0.25
        },
        
        # Respiratory failure (low SpO2 + high RR)
        'respiratory_failure': {
            'features': ['spo2', 'respiratory_rate'],
            'conditions': ['low', 'high'],
            'weight': 0.60,
            'partial_weight': 0.20
        },
        
        # Renal failure + hypotension
        'cardiorenal': {
            'features': ['creatinine', 'systolic_bp'],
            'conditions': ['high', 'low'],
            'weight': 0.55,
            'partial_weight': 0.20
        },
        
        # Frailty (age + multiple comorbidities)
        'frailty': {
            'features': ['age', 'creatinine', 'heart_rate'],
            'conditions': ['high', 'elevated', 'high'],
            'weight': 0.50,
            'partial_weight': 0.15
        }
    },
    
    # Non-linear terms (quadratic, exponential)
    'nonlinear_terms': {
        'age_squared': {
            'feature': 'age',
            'transform': 'quadratic',
            'weight': 0.0005  # Effect increases quadratically
        },
        'lactate_exponential': {
            'feature': 'lactate',
            'transform': 'exponential',
            'weight': 0.15,
            'scale': 0.3  # exp(0.3 * lactate)
        },
        'spo2_inverse': {
            'feature': 'spo2',
            'transform': 'inverse',
            'weight': -2.0  # -2.0 / spo2
        }
    },
    
    # Temporal effects (progression)
    'temporal_effects': {
        'deterioration_rate': 0.02,  # +2% per day in ICU
        'early_mortality_boost': 0.30,  # +30% in first 24h
        'late_recovery_bonus': -0.20   # -20% after 7 days (survivors)
    },
    
    # Stochasticity (noise in model)
    'stochasticity': {
        'base_noise': 0.15,        # ±15% Gaussian noise in score
        'individual_variation': 0.10,  # Individual variation (random effect)
        'measurement_error': 0.05   # Measurement error in features
    },
    
    # Baseline (intercept)
    'baseline_risk': -3.2,  # Logit(-3.0) ≈ 4.7% baseline (adjusted for target 18%)
    
    # Limits
    'min_probability': 0.001,  # Minimum 0.1%
    'max_probability': 0.95    # Maximum 95%
}

# Demographic diversity and class balancing
DIVERSITY_AND_BALANCING = {
    # Target mortality (increase positive cases)
    'target_mortality_rate': 0.18,  # 18% (vs ~10-12% natural)
    
    # Demographic stratification (ensure diversity)
    'demographic_strata': {
        # Age × Gender × Condition
        'age_groups': {
            'young': (18, 45),      # 15% of patients
            'middle': (45, 65),     # 30% of patients
            'elderly': (65, 80),    # 40% of patients
            'very_old': (80, 95)    # 15% of patients
        },
        'age_distribution': [0.15, 0.30, 0.40, 0.15],
        
        # Ensure representation of all groups
        'min_per_stratum': 50,  # Minimum 50 patients per stratum
    },
    
    # Condition diversity (positive cases)
    'condition_diversity': {
        'sepsis': 0.40,           # 40% of those who die
        'heart_failure': 0.25,    # 25% of those who die
        'aki': 0.20,              # 20% of those who die
        'respiratory': 0.10,      # 10% of those who die
        'multi_organ': 0.05       # 5% of those who die (most severe)
    },
    
    # Data augmentation (for positive cases)
    'augmentation': {
        'enabled': True,
        'augment_positive_only': True,  # Only positive cases
        'augmentation_factor': 1.5,     # 50% more cases via augmentation
        
        # Small perturbations (preserve realism)
        'perturbations': {
            # Vitals
            'heart_rate': {'std': 5, 'max_delta': 15},      # ±5 bpm (max ±15)
            'systolic_bp': {'std': 5, 'max_delta': 15},     # ±5 mmHg
            'diastolic_bp': {'std': 3, 'max_delta': 10},    # ±3 mmHg
            'respiratory_rate': {'std': 2, 'max_delta': 5}, # ±2 rpm
            'temperature': {'std': 0.3, 'max_delta': 0.8},  # ±0.3°C
            'spo2': {'std': 2, 'max_delta': 5},             # ±2%
            
            # Labs
            'lactate': {'std': 0.5, 'max_delta': 1.5},      # ±0.5 mmol/L
            'creatinine': {'std': 0.3, 'max_delta': 0.8},   # ±0.3 mg/dL
            'wbc': {'std': 2.0, 'max_delta': 5.0},          # ±2 × 10³/μL
            'hemoglobin': {'std': 0.5, 'max_delta': 1.5},   # ±0.5 g/dL
            'glucose': {'std': 15, 'max_delta': 40},        # ±15 mg/dL
            'potassium': {'std': 0.2, 'max_delta': 0.5},    # ±0.2 mEq/L
            'sodium': {'std': 2, 'max_delta': 5},           # ±2 mEq/L
            'bicarbonate': {'std': 1.5, 'max_delta': 4},    # ±1.5 mEq/L
            'bun': {'std': 5, 'max_delta': 15}              # ±5 mg/dL
        },
        
        # Timing perturbations
        'timing': {
            'admission_jitter_days': 7,      # ±7 days in admission date
            'los_jitter_hours': 12,          # ±12h in ICU time
            'measurement_jitter_minutes': 30 # ±30 min in timestamps
        }
    },
    
    # Intelligent oversampling (rare cases)
    'intelligent_oversampling': {
        'enabled': True,
        'oversample_rare_combinations': True,
        
        # Rare combinations to oversample
        'rare_combinations': {
            'young_sepsis': {
                'age_range': (18, 45),
                'condition': 'sepsis',
                'target_count': 100,
                'boost_factor': 2.0
            },
            'elderly_aki': {
                'age_range': (75, 95),
                'condition': 'aki',
                'target_count': 150,
                'boost_factor': 1.8
            },
            'female_heart_failure': {
                'sex': 'F',
                'condition': 'heart_failure',
                'target_count': 120,
                'boost_factor': 1.6
            }
        }
    }
}

# Dataset Shift and Holdout Domain
# Simulate distribution changes to evaluate generalization
DATASET_SHIFT = {
    # Enable holdout domain generation
    'enabled': True,
    'holdout_fraction': 0.15,  # 15% of data as holdout domain
    
    # Tipos de shift
    'shift_types': {
        # 1. Systematic measurement bias (measurement bias)
        'measurement_bias': {
            'enabled': True,
            'vitals': {
                'Heart Rate': {'bias': +3, 'std_increase': 1.2},      # +3 bpm, +20% std
                'Systolic blood pressure': {'bias': -5, 'std_increase': 1.15},  # -5 mmHg
                'SpO2': {'bias': -1, 'std_increase': 1.1},            # -1%
                'Temperature': {'bias': +0.2, 'std_increase': 1.1}    # +0.2°C
            },
            'labs': {
                'Lactate': {'bias': +0.3, 'std_increase': 1.2},       # +0.3 mmol/L
                'Creatinine': {'bias': +0.1, 'std_increase': 1.15},   # +0.1 mg/dL
                'Glucose': {'bias': +5, 'std_increase': 1.1}          # +5 mg/dL
            }
        },
        
        # 2. Population shift (demografia diferente)
        'population_shift': {
            'enabled': True,
            'age_shift': +5,           # Population 5 years older on average
            'male_ratio_shift': +0.05, # +5% men
            'severity_shift': +0.10    # +10% more severe
        },
        
        # 3. Temporal shift (clinical practices different)
        'temporal_shift': {
            'enabled': True,
            'measurement_frequency_change': 0.85,  # 15% fewer measurements
            'missing_rate_change': 1.20,           # +20% missing
            'los_change': 0.90                     # -10% ICU time
        },
        
        # 4. Protocol shift (different treatment protocols)
        'protocol_shift': {
            'enabled': True,
            'early_intervention_prob': 0.70,  # 70% early intervention (vs 50%)
            'aggressive_treatment_prob': 0.60 # 60% aggressive treatment (vs 40%)
        },
        
        # 5. Calibration drift (calibration drift)
        'calibration_drift': {
            'enabled': True,
            'drift_rate': 0.02,  # 2% drift per "month"
            'affected_features': ['SpO2', 'Temperature', 'Glucose']
        }
    },
    
    # Shift combination (multiple shifts can occur)
    'shift_combination': {
        'single_shift': 0.40,      # 40% only 1 type of shift
        'two_shifts': 0.35,        # 35% combination of 2 shifts
        'three_or_more': 0.25      # 25% combination of 3+ shifts
    },
    
    # Shift intensity (mild, moderate, severe)
    'shift_intensity': {
        'mild': {'probability': 0.50, 'multiplier': 0.5},      # 50% mild shift
        'moderate': {'probability': 0.35, 'multiplier': 1.0},  # 35% moderate shift
        'severe': {'probability': 0.15, 'multiplier': 1.5}     # 15% severe shift
    }
}

# Individual variability (anti-overfitting)
INDIVIDUAL_VARIABILITY = {
    'enabled': True,
    
    # Unique baseline offsets per patient
    'baseline_offsets': {
        'Heart Rate': {'std': 5},                    # ±5 bpm
        'Systolic blood pressure': {'std': 8},       # ±8 mmHg
        'Diastolic blood pressure': {'std': 5},      # ±5 mmHg
        'Respiratory Rate': {'std': 2},              # ±2 rpm
        'Temperature': {'std': 0.3},                 # ±0.3°C
        'SpO2': {'std': 2},                          # ±2%
        'Lactate': {'std': 0.5},                     # ±0.5 mmol/L
        'Creatinine': {'std': 0.2},                  # ±0.2 mg/dL
        'White blood cell count': {'std': 2.0},      # ±2 × 10³/μL
        'Glucose': {'std': 15},                      # ±15 mg/dL
    },
    
    # Individual deterioration rate
    'deterioration_rate': {
        'mean': 0.3,
        'std': 0.1  # 0.3 ± 0.1
    },
    
    # Condition sensitivity
    'condition_sensitivity': {
        'mean': 1.0,
        'std': 0.2  # 1.0 ± 0.2
    },
    
    # Temporal progression patterns
    'progression_patterns': {
        'exponential': {
            'probability': 0.25,
            'rate_range': (0.05, 0.15)
        },
        'sigmoid': {
            'probability': 0.20,
            'inflection_range': (12, 36),
            'steepness_range': (0.1, 0.5)
        },
        'linear': {
            'probability': 0.25,
            'rate_range': (0.2, 0.4)
        },
        'step': {
            'probability': 0.15,
            'step_time_range': (6, 42),
            'step_magnitude_range': (0.2, 0.6)
        },
        'oscillating': {
            'probability': 0.15,
            'base_rate_range': (0.2, 0.4),
            'amplitude': 0.1,
            'period': 6
        }
    }
}

# Stochastic correlations (anti-overfitting)
STOCHASTIC_CORRELATIONS = {
    'enabled': True,
    'correlation_noise_std': 0.10,  # Noise in base correlations
    
    # Correlations with variability
    'vitals': {
        ('Heart Rate', 'Respiratory Rate'): {
            'base': 0.45,
            'std': 0.10  # 0.45 ± 0.10
        },
        ('Systolic blood pressure', 'Heart Rate'): {
            'base': -0.30,
            'std': 0.08
        },
        ('Systolic blood pressure', 'SpO2'): {
            'base': 0.25,
            'std': 0.08
        }
    },
    
    'labs': {
        ('Creatinine', 'Blood Urea Nitrogen'): {
            'base': 0.80,
            'std': 0.10
        },
        ('Lactate', 'Bicarbonate'): {
            'base': -0.65,
            'std': 0.10
        }
    },
    
    'vitals_labs': {
        ('Heart Rate', 'Lactate'): {
            'base': 0.50,
            'std': 0.12
        },
        ('Systolic blood pressure', 'Lactate'): {
            'base': -0.55,
            'std': 0.12
        }
    }
}

