#!/usr/bin/env python3
"""
Synthetic MIMIC-III Data Generator

Improvements over v2:
- Circadian patterns (24h)
- Sleep simulation (11pm-6am)
- Meal effects (7am/12pm/6pm)
- Realistic noise and entropy
- Stochastic events
- Quantified impacts by condition
- Centralized configuration

Advanced Temporal Variability:
- Signal-specific noise (realistic SNR ±0.5-8%)
- Time warping/jitter (±10% spacing variation)
- Temporal dropout (10% missing measurements)
- Isolated artifacts (3% spike probability)

Realistic Missingness (MCAR/MAR/MNAR):
- MCAR: 30% completely random
- MAR: 50% dependent on observed severity
- MNAR: 20% dependent on unobserved values
- CLINICAL REALITY: Sicker patients have MORE measurements (less missing)
- Temporal patterns: +30% missing at night, +15% at shift changes

Realistic Correlations:
- Hierarchical generation: Demographics → Condition → Vitals → Labs
- Multivariate covariance matrix for vitals (HR ↔ RR: 0.45, BP ↔ HR: -0.30)
- Vitals-labs correlations (HR ↔ Lactate: 0.50, BP ↔ Lactate: -0.55)
- Labs correlations (Cr ↔ BUN: 0.80, Lactate ↔ HCO3: -0.65)
- P(Condition | Age, Sex): Elderly have +80% sepsis, +120% heart failure

Label Noise and Clinical Ambiguity:
- Label flip: 1% false positives (0→1), 2% false negatives (1→0)
- Censoring: 5% transfer, 2% AMA discharge, 1% loss to follow-up
- Ambiguity: 8% borderline cases (palliative, withdrawal of support, etc.)
- Timing uncertainty: ±6h death, ±4h discharge, 15% documentation delay
- REALITY: Labels are not 100% deterministic

Stochastic and Multivariate Mortality Model:
- AVOIDS trivial patterns (not "if A > X then death")
- Non-linear combination of multiple features
- Interactions: shock (BP+HR+Lac), sepsis (Temp+WBC+Lac), etc.
- Non-linear terms: age², lactate^exp, SpO2^-1
- Temporal effects: deterioration, early mortality, late recovery
- Stochasticity: Gaussian noise, individual variation, measurement error

Demographic Diversity and Balancing:
- Stratified age distribution: 15% young, 30% middle-aged, 40% elderly, 15% very elderly
- Target mortality: 18% (increases positive cases)
- Data augmentation: +50% positive cases with small perturbations
- Perturbations: ±5 bpm HR, ±5 mmHg BP, ±0.5 mmol/L Lac (preserves realism)
- Condition diversity: sepsis (40%), HF (25%), AKI (20%), resp (10%), multi-organ (5%)

Dataset Shift (Holdout Domain):
- 15% of data as holdout domain (different distribution)
- Measurement bias: HR +3 bpm, BP -5 mmHg, SpO2 -1%, Lac +0.3 mmol/L
- Population shift: +5 years mean age, +5% male, +10% more severe
- Temporal shift: -15% measurements, +20% missing, -10% LOS
- Variable intensity: 50% mild, 35% moderate, 15% severe
- OBJECTIVE: Evaluate generalization and detect artifact memorization

Documentation Quality (Shift Variation):
- Day (7am-3pm): 95% completeness, 97% accuracy (best)
- Evening (3pm-11pm): 90% completeness, 94% accuracy
- Night (11pm-7am): 85% completeness, 90% accuracy (worst)
- Weekend: 87% completeness, 92% accuracy (reduced)
- REALITY: Documentation varies with staff availability

Individual Variability (Anti-Overfitting):
- Unique baseline offsets per patient: ±5 bpm HR, ±8 mmHg BP, ±0.3°C Temp
- Individual deterioration rate: 0.3 ± 0.1 (each patient different)
- Condition sensitivity: 1.0 ± 0.2 (individual response)
- 5 progression patterns: exponential, sigmoid, linear, step, oscillating
- AVOIDS: All sepsis patients having EXACTLY +25 bpm

Stochastic Correlations (Anti-Overfitting):
- Correlations vary per patient: HR↔RR = 0.45 ± 0.10 (not fixed)
- Noise in covariance matrix: ±10% in base correlations
- AVOIDS: Correlation always 0.45 (memorizable)
- CREATES: Correlation varies between 0.35-0.55 (realistic)

Objective: Reduce extreme overfitting from v2

Author: Andre Lehdermann Silveira
Date: October 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add root directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import constants
from config.constants import (
    NORMAL_RANGES, PATHOLOGICAL_RANGES,
    CONDITION_IMPACTS, CIRCADIAN_PARAMS,
    SLEEP_PARAMS, MEAL_PARAMS, ENTROPY_PARAMS,
    SIGNAL_NOISE_PARAMS, TEMPORAL_VARIABILITY_PARAMS,
    MISSINGNESS_PARAMS, CORRELATIONS, CONDITIONAL_GENERATION,
    COVARIANCE_STRUCTURE, LABEL_NOISE_PARAMS, DOCUMENTATION_PATTERNS,
    MORTALITY_RISK_MODEL, DIVERSITY_AND_BALANCING, DATASET_SHIFT,
    INDIVIDUAL_VARIABILITY, STOCHASTIC_CORRELATIONS
)

# Configuration
np.random.seed(42)
N_PATIENTS = 5000
OUTPUT_DIR = "data/mimic3_synthetic_v3/raw"

# Create directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SYNTHETIC MIMIC-III DATA GENERATOR")
print("=" * 70)
print(f"Generating data for {N_PATIENTS} patients...")
print()


# ============================================================================
# AUXILIARY FUNCTIONS
# ============================================================================

def get_hour_from_timestamp(hours_since_admit):
    """Convert hours since admission to hour of day (0-23)"""
    # Assume admission at 12pm (noon)
    admit_hour = 12
    current_hour = (admit_hour + hours_since_admit) % 24
    return current_hour


def apply_circadian_pattern(value, feature_name, hour_of_day):
    """
    Apply circadian pattern to a value
    
    Args:
        value: Base value
        feature_name: Feature name
        hour_of_day: Hour of day (0-23)
    
    Returns:
        Value with circadian pattern applied
    """
    if feature_name not in CIRCADIAN_PARAMS:
        return value
    
    params = CIRCADIAN_PARAMS[feature_name]
    peak_hour = params['peak_hour']
    trough_hour = params['trough_hour']
    amplitude = params['amplitude']
    
    # Calculate circadian phase (peak at peak_hour, minimum at trough_hour)
    phase = (hour_of_day - trough_hour) / 24 * 2 * np.pi
    circadian_factor = 0.5 + 0.5 * np.cos(phase)
    
    # Apply amplitude
    if feature_name == 'Temperature':
        # For temperature, amplitude is absolute
        adjustment = (circadian_factor - 0.5) * 2 * amplitude
        return value + adjustment
    else:
        # For other features, amplitude is relative
        adjustment = (circadian_factor - 0.5) * 2 * amplitude
        return value * (1 + adjustment)


def apply_sleep_effect(value, feature_name, hour_of_day):
    """
    Apply sleep effect
    
    Args:
        value: Base value
        feature_name: Feature name
        hour_of_day: Hour of day (0-23)
    
    Returns:
        Value with sleep effect applied
    """
    sleep_start = SLEEP_PARAMS['start_hour']
    sleep_duration = SLEEP_PARAMS['duration_hours']
    sleep_end = (sleep_start + sleep_duration) % 24
    
    # Check if in sleep period
    if sleep_end > sleep_start:
        is_sleeping = sleep_start <= hour_of_day < sleep_end
    else:  # Sleep crosses midnight
        is_sleeping = hour_of_day >= sleep_start or hour_of_day < sleep_end
    
    if not is_sleeping:
        return value
    
    # Calculate sleep depth (deeper in the middle)
    if sleep_end > sleep_start:
        sleep_midpoint = sleep_start + sleep_duration / 2
        time_to_midpoint = abs(hour_of_day - sleep_midpoint)
    else:
        sleep_midpoint = (sleep_start + sleep_duration / 2) % 24
        if hour_of_day >= sleep_start:
            time_to_midpoint = abs(hour_of_day - sleep_midpoint)
            if sleep_midpoint < sleep_start:
                time_to_midpoint = abs((hour_of_day - 24) - sleep_midpoint)
        else:
            time_to_midpoint = abs(hour_of_day - sleep_midpoint)
    
    sleep_depth = 1.0 - min(1.0, time_to_midpoint / (sleep_duration / 2))
    
    # Apply reduction
    if feature_name in SLEEP_PARAMS['reductions']:
        reduction = SLEEP_PARAMS['reductions'][feature_name]
        return value * (1 - reduction * sleep_depth)
    
    return value


def apply_meal_effect(value, feature_name, hour_of_day):
    """
    Apply meal effect
    
    Args:
        value: Base value
        feature_name: Feature name
        hour_of_day: Hour of day (0-23)
    
    Returns:
        Value with meal effect applied
    """
    if feature_name not in MEAL_PARAMS['effects']:
        return value
    
    meal_times = MEAL_PARAMS['times']
    effect_duration = MEAL_PARAMS['effect_duration']
    
    # Verify meal proximity
    max_effect = 0.0
    for meal_time in meal_times:
        time_since_meal = (hour_of_day - meal_time) % 24
        
        if time_since_meal < effect_duration:
            # Exponential decay
            effect = np.exp(-3.0 * time_since_meal / effect_duration)
            max_effect = max(max_effect, effect)
    
    # Apply effect
    if max_effect > 0:
        increase = MEAL_PARAMS['effects'][feature_name]
        return value * (1 + increase * max_effect)
    
    return value


def add_entropy(value, feature_name):
    """
    Add entropy (noise) to the value
    
    Args:
        value: Base value
        feature_name: Feature name
    
    Returns:
        Value with entropy applied
    """
    # Temporal noise (use abs to avoid negative scale)
    temporal_noise = ENTROPY_PARAMS['temporal_noise']
    noise = np.random.normal(0, temporal_noise * abs(value))
    value_with_noise = value + noise
    
    # Measurement noise (use abs to avoid negative scale)
    measurement_noise = ENTROPY_PARAMS['measurement_noise']
    measurement_error = np.random.normal(0, measurement_noise * abs(value))
    value_with_noise += measurement_error
    
    # Occasional spikes
    if np.random.random() < ENTROPY_PARAMS['spike_probability']:
        spike_direction = np.random.choice([-1, 1])
        spike_magnitude = np.random.uniform(0.15, 0.30)
        value_with_noise *= (1 + spike_direction * spike_magnitude)
    
    return value_with_noise


def add_signal_specific_noise(value, feature_name):
    """
    Add Gaussian noise specific by signal with SNR realistically
    
    Args:
        value: Base value
        feature_name: Feature name
    
    Returns:
        Value with specific signal noise applied
    """
    if feature_name not in SIGNAL_NOISE_PARAMS:
        return value
    
    # Noise proportional to value (SNR)
    noise_scale = SIGNAL_NOISE_PARAMS[feature_name]
    noise = np.random.normal(0, noise_scale * abs(value))
    
    return value + noise


def add_temporal_artifact(value, feature_name):
    """
    Add temporal artifacts (isolated peaks) with low probability
    
    Args:
        value: Base value
        feature_name: Feature name
    
    Returns:
        Value with temporal artifact applied (or original value)
    """
    artifact_prob = TEMPORAL_VARIABILITY_PARAMS['artifact_probability']
    
    if np.random.random() < artifact_prob:
        # Isolated peak artifact
        artifact_direction = np.random.choice([-1, 1])
        min_mag, max_mag = TEMPORAL_VARIABILITY_PARAMS['artifact_magnitude_range']
        artifact_magnitude = np.random.uniform(min_mag, max_mag)
        
        return value * (1 + artifact_direction * artifact_magnitude)
    
    return value


def should_dropout_measurement():
    """
    Determine if a measurement should be dropped (missing)
    
    Returns:
        True if should drop, False otherwise
    """
    dropout_prob = TEMPORAL_VARIABILITY_PARAMS['time_dropout_prob']
    return np.random.random() < dropout_prob


def apply_time_jitter(base_hour, max_jitter_hours=1.0):
    """
    Apply temporal jitter (variation in spacing between records)
    
    Args:
        base_hour: Base hour
        max_jitter_hours: Maximum jitter in hours
    
    Returns:
        Hour with jitter applied
    """
    jitter_range = TEMPORAL_VARIABILITY_PARAMS['time_jitter_range']
    jitter = np.random.uniform(-jitter_range * max_jitter_hours, 
                                jitter_range * max_jitter_hours)
    return base_hour + jitter


def calculate_severity_score(vital_values, lab_values, condition, died, progress=0.0):
    """
    Calculate severity score type SOFA simplified
    
    Args:
        vital_values: Dict with vital signal values
        lab_values: Dict with lab values
        condition: Patient condition
        died: If patient died
        progress: Temporal progress (0-1)
    
    Returns:
        Severity score (0-10, higher = more severe)
    """
    score = 0.0
    
    # Respiratory component (SpO2)
    if 'SpO2' in vital_values:
        spo2 = vital_values['SpO2']
        if spo2 < 90:
            score += 2.0
        elif spo2 < 95:
            score += 1.0
    
    # Cardiovascular component (PA and HR)
    if 'Systolic blood pressure' in vital_values:
        sbp = vital_values['Systolic blood pressure']
        if sbp < 90:
            score += 2.0
        elif sbp < 100:
            score += 1.0
    
    if 'Heart Rate' in vital_values:
        hr = vital_values['Heart Rate']
        if hr > 120 or hr < 50:
            score += 1.0
    
    # Renal component (Creatinine)
    if 'Creatinine' in lab_values:
        cr = lab_values['Creatinine']
        if cr > 3.5:
            score += 2.0
        elif cr > 2.0:
            score += 1.0
    
    # Hepatic/metabolic component (Lactate)
    if 'Lactate' in lab_values:
        lac = lab_values['Lactate']
        if lac > 4.0:
            score += 2.0
        elif lac > 2.0:
            score += 1.0
    
    # Condition severity
    condition_severity = {
        'sepsis': 2.5,
        'aki': 2.0,
        'heart_failure': 1.5,
        'copd': 1.0,
        'diabetes': 0.5,
        'healthy': 0.0
    }
    score += condition_severity.get(condition, 0.0)
    
    # Progression adjustment (if died, worsens with time)
    if died:
        score += 3.0 * progress
    
    # Normalize to 0-10
    return min(10.0, max(0.0, score))


def sigmoid(x):
    """Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-x))


def calculate_missingness_probability(severity_score, measurement_type, hour_of_day, mechanism='MAR'):
    """
    Calculate missingness probability based on multiple factors
    
    Args:
        severity_score: Severity score (0-10)
        measurement_type: 'vital_signs' or 'labs'
        hour_of_day: Hour of day (0-23)
        mechanism: 'MCAR', 'MAR', or 'MNAR'
    
    Returns:
        Missingness probability (0-1)
    """
    
    # MCAR: Completely random (does not depend on anything)
    if mechanism == 'MCAR':
        return MISSINGNESS_PARAMS['measurement_specific'][measurement_type]['baseline_missing']
    
    # Base probability
    base_prob = MISSINGNESS_PARAMS['measurement_specific'][measurement_type]['baseline_missing']
    
    # MAR: Depends on observed severity (REALIDADE: MORE severe = LESS missing)
    if mechanism == 'MAR':
        severity_params = MISSINGNESS_PARAMS['severity_effect']
        
        # Normalize severity (0-10) to (-1, 1)
        normalized_severity = (severity_score - 5.0) / 5.0
        
        # Inverse sigmoid: more severe (high severity) → less missing
        # p_missing = max_prob - (max_prob - min_prob) * sigmoid(steepness * severity)
        steepness = severity_params['steepness']
        min_prob = severity_params['min_prob']
        max_prob = severity_params['max_prob']
        
        severity_factor = sigmoid(steepness * normalized_severity)
        prob = max_prob - (max_prob - min_prob) * severity_factor
        
        # Specific adjustment by measurement type
        severity_mult = MISSINGNESS_PARAMS['measurement_specific'][measurement_type]['severity_multiplier']
        prob = prob * (1.0 - severity_mult * (severity_score / 10.0))
    
    # MNAR: Depends on unobserved values (e.g., extreme values have more missing)
    elif mechanism == 'MNAR':
        # For MNAR, extreme values (very severe) may have more missing
        # (equipment failure, patient too unstable to measure)
        if severity_score > 8.0:
            prob = base_prob * 1.5  # +50% missing in extreme cases
        else:
            prob = base_prob
    else:
        prob = base_prob
    
    # Temporal adjustments
    temporal_params = MISSINGNESS_PARAMS['temporal_patterns']
    
    # Night (11pm-6am): more missing
    if 23 <= hour_of_day or hour_of_day < 6:
        prob *= temporal_params['night_increase']
    
    # Shift changes (7am, 3pm, 11pm): more missing
    if hour_of_day in [7, 15, 23]:
        prob *= temporal_params['handoff_increase']
    
    # Limit between 0 and 1
    return min(0.95, max(0.01, prob))


def should_be_missing(severity_score, measurement_type, hour_of_day):
    """
    Determine if a measurement should be missing
    
    Args:
        severity_score: Severity score (0-10)
        measurement_type: 'vital_signs' or 'labs'
        hour_of_day: Hour of day (0-23)
    
    Returns:
        True if should be missing, False otherwise
    """
    # Choose missingness mechanism
    mechanisms = MISSINGNESS_PARAMS['mechanisms']
    mechanism = np.random.choice(
        list(mechanisms.keys()),
        p=list(mechanisms.values())
    )
    
    # Calculate probability
    prob = calculate_missingness_probability(
        severity_score, measurement_type, hour_of_day, mechanism
    )
    
    # Decision
    return np.random.random() < prob


def sample_condition_given_demographics(age, sex, died):
    """
    Sample condition based on demographics (hierarchical generation)
    P(Condition | Age, Sex, Died)
    
    Args:
        age: Patient age
        sex: Sex ('M' or 'F')
        died: If patient died (boolean)
    
    Returns:
        Condition: 'sepsis', 'aki', 'heart_failure', or 'healthy'
    """
    if not died:
        return 'healthy'
    
    # Calculate conditional probabilities
    probs = {}
    for condition in ['sepsis', 'aki', 'heart_failure']:
        if condition not in CONDITIONAL_GENERATION['condition_given_demographics']:
            continue
            
        params = CONDITIONAL_GENERATION['condition_given_demographics'][condition]
        
        # Base probability
        prob = params['age_effect']['baseline']
        
        # Age adjustment
        if age > 70:
            prob *= params['age_effect']['age_>70']
        elif age < 40:
            prob *= params['age_effect']['age_<40']
        
        # Sex adjustment
        prob *= params['sex_effect'].get(sex, 1.0)
        
        probs[condition] = prob
    
    # Normalize probabilities
    total = sum(probs.values())
    if total > 0:
        probs = {k: v/total for k, v in probs.items()}
    else:
        probs = {'sepsis': 0.5, 'aki': 0.3, 'heart_failure': 0.2}
    
    # Sample
    conditions = list(probs.keys())
    probabilities = list(probs.values())
    
    return np.random.choice(conditions, p=probabilities)


def build_correlation_matrix(feature_names, correlation_dict, correlation_strength=0.7):
    """
    Build correlation matrix from correlation pairs
    
    Args:
        feature_names: List of feature names
        correlation_dict: Dict with pairs (feat1, feat2): correlation
        correlation_strength: Scale factor (0-1)
    
    Returns:
        Correlation matrix (numpy array)
    """
    n = len(feature_names)
    corr_matrix = np.eye(n)
    
    # Fill correlations
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            if i == j:
                continue
            
            # Search for correlation
            corr = 0.0
            if (feat1, feat2) in correlation_dict:
                corr = correlation_dict[(feat1, feat2)]
            elif (feat2, feat1) in correlation_dict:
                corr = correlation_dict[(feat2, feat1)]
            
            # Apply strength
            corr *= correlation_strength
            
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    
    # Ensure matrix is positive semi-definite
    # Simple method: add small value to diagonal if necessary
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    if eigenvalues.min() < 0:
        corr_matrix += np.eye(n) * (abs(eigenvalues.min()) + 0.01)
    
    return corr_matrix


def generate_correlated_values(base_means, base_stds, feature_names, correlation_dict):
    """
    Generate correlated values using multivariate normal distribution
    
    Args:
        base_means: Dict {feature: mean}
        base_stds: Dict {feature: std}
        feature_names: List of features
        correlation_dict: Dict with correlations
    
    Returns:
        Dict {feature: value}
    """
    if not COVARIANCE_STRUCTURE['use_correlated_generation']:
        # Independent generation (fallback)
        return {
            feat: np.random.normal(base_means[feat], base_stds[feat])
            for feat in feature_names
        }
    
    # Build correlation matrix
    corr_strength = COVARIANCE_STRUCTURE['correlation_strength']
    corr_matrix = build_correlation_matrix(feature_names, correlation_dict, corr_strength)
    
    # Build covariance matrix
    means = np.array([base_means[feat] for feat in feature_names])
    stds = np.array([base_stds[feat] for feat in feature_names])
    
    # Cov = D * Corr * D, where D is diagonal of stds
    D = np.diag(stds)
    cov_matrix = D @ corr_matrix @ D
    
    # Sample from multivariate normal
    try:
        values = np.random.multivariate_normal(means, cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback if matrix is not positive definite
        values = np.array([np.random.normal(base_means[feat], base_stds[feat]) 
                          for feat in feature_names])
    
    return {feat: val for feat, val in zip(feature_names, values)}


def generate_vitals_given_condition(condition):
    """
    Generate vital signs conditioned on patient condition
    P(Vitals | Condition)
    
    Args:
        condition: Patient condition
    
    Returns:
        Dict com valores de sinais vitais
    """
    vital_names = ['Heart Rate', 'Systolic blood pressure', 'Diastolic blood pressure',
                   'Respiratory Rate', 'Temperature', 'SpO2']
    
    # Get shifts and multipliers from condition
    condition_params = CONDITIONAL_GENERATION['vitals_given_condition'].get(
        condition, CONDITIONAL_GENERATION['vitals_given_condition']['healthy']
    )
    
    # Calculate base means and stds
    base_means = {}
    base_stds = {}
    
    for vital in vital_names:
        if vital in NORMAL_RANGES:
            min_val, max_val = NORMAL_RANGES[vital]
            base_mean = (min_val + max_val) / 2
            base_std = (max_val - min_val) / 6  # ~99.7% dentro do range
        else:
            base_mean = 75.0
            base_std = 10.0
        
        # Apply condition shifts
        if vital in condition_params:
            params = condition_params[vital]
            base_mean += params['mean_shift']
            base_std *= params['std_multiplier']
        
        base_means[vital] = base_mean
        base_stds[vital] = base_std
    
    # Generate correlated values
    values = generate_correlated_values(
        base_means, base_stds, vital_names, 
        CORRELATIONS['vital_signs']
    )
    
    return values


def generate_labs_given_condition_and_vitals(condition, vital_values):
    """
    Generate labs conditioned on condition and vital signs
    P(Labs | Condition, Vitals)
    
    Args:
        condition: Patient condition
        vital_values: Dict with vital sign values
    
    Returns:
        Dict with lab values
    """
    lab_names = ['Lactate', 'Creatinine', 'White blood cell count', 'Hemoglobin',
                 'Glucose', 'Potassium', 'Sodium', 'Bicarbonate', 'Blood Urea Nitrogen']
    
    # Get condition shifts
    condition_params = CONDITIONAL_GENERATION['labs_given_condition'].get(
        condition, CONDITIONAL_GENERATION['labs_given_condition']['healthy']
    )
    
    # Calculate base means and stds
    base_means = {}
    base_stds = {}
    
    for lab in lab_names:
        if lab in NORMAL_RANGES:
            min_val, max_val = NORMAL_RANGES[lab]
            base_mean = (min_val + max_val) / 2
            base_std = (max_val - min_val) / 6
        else:
            base_mean = 100.0
            base_std = 15.0
        
        # Apply condition shifts
        if lab in condition_params:
            params = condition_params[lab]
            base_mean += params['mean_shift']
            base_std *= params['std_multiplier']
        
        base_means[lab] = base_mean
        base_stds[lab] = base_std
    
    # Generate correlated values (labs among themselves)
    values = generate_correlated_values(
        base_means, base_stds, lab_names,
        CORRELATIONS['labs']
    )
    
    # Adjust labs based on vitals (vital-lab correlations)
    for (vital_name, lab_name), corr in CORRELATIONS['vital_lab'].items():
        if vital_name in vital_values and lab_name in values:
            # Adjustment proportional to correlation
            vital_val = vital_values[vital_name]
            vital_mean = (NORMAL_RANGES[vital_name][0] + NORMAL_RANGES[vital_name][1]) / 2
            vital_std = (NORMAL_RANGES[vital_name][1] - NORMAL_RANGES[vital_name][0]) / 6
            
            # Z-score do vital
            vital_z = (vital_val - vital_mean) / vital_std if vital_std > 0 else 0
            
            # Ajustar lab proporcionalmente
            lab_std = base_stds[lab_name]
            adjustment = corr * vital_z * lab_std * 0.3  # 30% do efeito
            
            values[lab_name] += adjustment
    
    return values


def apply_label_noise(true_label, patient_severity=0.5):
    """
    Apply noise to mortality label (label noise)
    
    CLINICAL REALITY: Labels are not 100% deterministic
    - Documentation errors
    - Deaths after transfer not recorded
    - Patient confusion
    - Recording timing
    
    Args:
        true_label: True label (0=survived, 1=died)
        patient_severity: Patient severity (0-1)
    
    Returns:
        Tuple (noisy_label, was_flipped, reason)
    """
    noise_params = LABEL_NOISE_PARAMS['mortality_label_noise']
    
    # Probability of flip depends on the true label
    if true_label == 0:
        # Survived → marked as dead (false positive)
        flip_prob = noise_params['flip_0_to_1']
        reasons = noise_params['reasons_0_to_1']
    else:
        # Died → marked as survived (false negative)
        flip_prob = noise_params['flip_1_to_0']
        reasons = noise_params['reasons_1_to_0']
    
    # Flip?
    if np.random.random() < flip_prob:
        noisy_label = 1 - true_label
        reason = np.random.choice(reasons)
        return noisy_label, True, reason
    else:
        return true_label, False, None


def apply_censoring(patient_data):
    """
    Apply censoring (uncertain outcome)
    
    Censoring scenarios:
    - Transfer to another hospital
    - Discharge at patient request (AMA)
    - Loss of follow-up
    
    Args:
        patient_data: Dict with patient data
    
    Returns:
        Dict updated with censoring applied
    """
    censoring_params = LABEL_NOISE_PARAMS['censoring']
    
    # Determine if patient is censored
    total_censoring_rate = (
        censoring_params['transfer_rate'] +
        censoring_params['discharge_ama_rate'] +
        censoring_params['lost_followup_rate']
    )
    
    if np.random.random() > total_censoring_rate:
        # Not censored
        patient_data['censored'] = False
        patient_data['censoring_reason'] = None
        return patient_data
    
    # Determine type of censoring
    probs = [
        censoring_params['transfer_rate'],
        censoring_params['discharge_ama_rate'],
        censoring_params['lost_followup_rate']
    ]
    probs = [p / sum(probs) for p in probs]
    
    censoring_type = np.random.choice(
        ['transfer', 'discharge_ama', 'lost_followup'],
        p=probs
    )
    
    # Determine true outcome (unknown to the model)
    if censoring_type == 'transfer':
        true_died = np.random.random() < censoring_params['transfer_mortality_prob']
    elif censoring_type == 'discharge_ama':
        true_died = np.random.random() < censoring_params['discharge_ama_mortality_prob']
    else:  # lost_followup
        true_died = np.random.random() < censoring_params['lost_followup_mortality_prob']
    
    # Update data
    patient_data['censored'] = True
    patient_data['censoring_reason'] = censoring_type
    patient_data['true_outcome_unknown'] = true_died  # True (not observable)
    patient_data['observed_outcome'] = 0  # Always marked as survived (censored)
    
    return patient_data


def apply_clinical_ambiguity(patient_data):
    """
    Apply clinical ambiguity (borderline cases)
    
    Ambiguous scenarios:
    - Transition to palliative care
    - Withdrawal of life support
    - Brain death
    - Transfer to hospice
    
    Args:
        patient_data: Dict with patient data
    
    Returns:
        Dict updates with clinical ambiguity applied
    """
    ambiguity_params = LABEL_NOISE_PARAMS['clinical_ambiguity']
    
    # Check if ambiguous case
    if np.random.random() > ambiguity_params['borderline_cases_rate']:
        patient_data['ambiguous'] = False
        patient_data['ambiguity_scenario'] = None
        return patient_data
    
    # Choose ambiguous scenario
    scenarios = ambiguity_params['borderline_scenarios']
    scenario_names = list(scenarios.keys())
    scenario_probs = [scenarios[s]['probability'] for s in scenario_names]
    
    scenario = np.random.choice(scenario_names, p=scenario_probs)
    scenario_data = scenarios[scenario]
    
    # Apply label uncertainty
    uncertainty = scenario_data['label_uncertainty']
    
    # If patient died, may be marked as survived (and vice-versa)
    if np.random.random() < uncertainty:
        # Flip label
        original_label = patient_data.get('HOSPITAL_EXPIRE_FLAG', 0)
        patient_data['HOSPITAL_EXPIRE_FLAG'] = 1 - original_label
        patient_data['label_flipped_by_ambiguity'] = True
    else:
        patient_data['label_flipped_by_ambiguity'] = False
    
    patient_data['ambiguous'] = True
    patient_data['ambiguity_scenario'] = scenario
    patient_data['ambiguity_description'] = scenario_data['description']
    
    return patient_data


def apply_timing_uncertainty(timestamp, event_type='death'):
    """
    Apply temporal uncertainty (jitter in recording time)
    
    Args:
        timestamp: Timestamp original
        event_type: 'death' ou 'discharge'
    
    Returns:
        Timestamp with jitter applied
    """
    timing_params = LABEL_NOISE_PARAMS['timing_uncertainty']
    
    # Jitter based on event type
    if event_type == 'death':
        jitter_hours = timing_params['death_time_jitter_hours']
    else:
        jitter_hours = timing_params['discharge_time_jitter_hours']
    
    # Apply jitter
    jitter = np.random.uniform(-jitter_hours, jitter_hours)
    noisy_timestamp = timestamp + timedelta(hours=jitter)
    
    # Documentation delay?
    if np.random.random() < timing_params['documentation_delay_prob']:
        delay_min, delay_max = timing_params['documentation_delay_hours']
        delay = np.random.uniform(delay_min, delay_max)
        noisy_timestamp += timedelta(hours=delay)
    
    return noisy_timestamp


def calculate_mortality_risk_stochastic(patient_features, icu_day=1, individual_effect=None):
    """
    Calculates mortality risk using STOCHASTIC and MULTIVARIATE model
    
    AVOIDS simple patterns:
    - Not deterministic (e.g., "if A > X then death")
    - Combines multiple features with non-linear interactions
    - Adds stochastic noise
    - Temporal effects
    
    Args:
        patient_features: Dict with patient features
        icu_day: ICU day (1-based)
        individual_effect: Individual effect (None = generate new)
    
    Returns:
        Mortality risk (0-1)
    """
    model = MORTALITY_RISK_MODEL
    
    # Initialize score (logit)
    score = model['baseline_risk']
    
    # 1. LINEAR EFFECTS (demographics)
    age = patient_features.get('age', 60)
    if age > 50:
        score += model['feature_weights']['age'] * (age - 50)
    
    if patient_features.get('sex') == 'M':
        score += model['feature_weights']['male']
    
    # 2. NON-LINEAR EFFECTS (vitals and labs)
    # Heart Rate
    hr = patient_features.get('heart_rate', 80)
    hr_weights = model['feature_weights']['heart_rate']
    if hr < 50:
        score += hr_weights['low']
    elif hr > 120:
        score += hr_weights['high']
    elif 60 <= hr <= 80:
        score += hr_weights['optimal']
    
    # Systolic BP
    sbp = patient_features.get('systolic_bp', 120)
    sbp_weights = model['feature_weights']['systolic_bp']
    if sbp < 90:
        score += sbp_weights['low']
    elif sbp > 160:
        score += sbp_weights['high']
    elif 110 <= sbp <= 130:
        score += sbp_weights['optimal']
    
    # SpO2
    spo2 = patient_features.get('spo2', 97)
    spo2_weights = model['feature_weights']['spo2']
    if spo2 < 85:
        score += spo2_weights['critical']
    elif spo2 < 92:
        score += spo2_weights['low']
    elif spo2 > 95:
        score += spo2_weights['optimal']
    
    # Temperature
    temp = patient_features.get('temperature', 37.0)
    temp_weights = model['feature_weights']['temperature']
    if temp < 36:
        score += temp_weights['hypothermia']
    elif temp > 40:
        score += temp_weights['hyperthermia']
    elif temp > 38.5:
        score += temp_weights['fever']
    
    # Lactate
    lac = patient_features.get('lactate', 1.5)
    lac_weights = model['feature_weights']['lactate']
    if lac < 2:
        score += lac_weights['normal']
    elif lac < 4:
        score += lac_weights['elevated']
    else:
        score += lac_weights['high']
    
    # Creatinine
    cr = patient_features.get('creatinine', 1.0)
    cr_weights = model['feature_weights']['creatinine']
    if cr < 1.5:
        score += cr_weights['normal']
    elif cr < 3:
        score += cr_weights['elevated']
    else:
        score += cr_weights['high']
    
    # WBC
    wbc = patient_features.get('wbc', 8.0)
    wbc_weights = model['feature_weights']['wbc']
    if wbc < 4:
        score += wbc_weights['low']
    elif wbc > 25:
        score += wbc_weights['very_high']
    elif wbc > 15:
        score += wbc_weights['high']
    
    # 3. NON-LINEAR INTERACTIONS BETWEEN FEATURES
    interactions = model['feature_interactions']
    
    # Shock (PA low + FC high + Lactate high)
    shock_conditions = [sbp < 90, hr > 120, lac > 4]
    n_shock = sum(shock_conditions)
    if n_shock == 3:
        score += interactions['shock']['weight']
    elif n_shock == 2:
        score += interactions['shock']['partial_weight']
    
    # Sepsis-like (Temp anormal + WBC alto + Lactato alto)
    sepsis_conditions = [temp > 38.5 or temp < 36, wbc > 15, lac > 4]
    n_sepsis = sum(sepsis_conditions)
    if n_sepsis == 3:
        score += interactions['sepsis_like']['weight']
    elif n_sepsis == 2:
        score += interactions['sepsis_like']['partial_weight']
    
    # Respiratory failure (SpO2 low + RR high)
    rr = patient_features.get('respiratory_rate', 16)
    resp_conditions = [spo2 < 92, rr > 24]
    if all(resp_conditions):
        score += interactions['respiratory_failure']['weight']
    elif any(resp_conditions):
        score += interactions['respiratory_failure']['partial_weight']
    
    # Cardiorenal (Cr high + PA low)
    cardiorenal_conditions = [cr > 3, sbp < 90]
    if all(cardiorenal_conditions):
        score += interactions['cardiorenal']['weight']
    elif any(cardiorenal_conditions):
        score += interactions['cardiorenal']['partial_weight']
    
    # Frailty (Age + Cr + FC)
    frailty_conditions = [age > 75, cr > 1.5, hr > 100]
    n_frailty = sum(frailty_conditions)
    if n_frailty == 3:
        score += interactions['frailty']['weight']
    elif n_frailty == 2:
        score += interactions['frailty']['partial_weight']
    
    # 4. NON-LINEAR TERMS (quadratic, exponential)
    nonlinear = model['nonlinear_terms']
    
    # Age squared
    score += nonlinear['age_squared']['weight'] * (age ** 2)
    
    # Lactate exponential
    lac_exp = nonlinear['lactate_exponential']
    score += lac_exp['weight'] * (np.exp(lac_exp['scale'] * lac) - 1)
    
    # SpO2 inverse
    if spo2 > 0:
        score += nonlinear['spo2_inverse']['weight'] / spo2
    
    # 5. TEMPORAL EFFECTS
    temporal = model['temporal_effects']
    
    # Deterioration over time
    score += temporal['deterioration_rate'] * icu_day
    
    # Early mortality boost (first 24h)
    if icu_day <= 1:
        score += temporal['early_mortality_boost']
    
    # Late recovery bonus (after 7 days)
    if icu_day > 7:
        score += temporal['late_recovery_bonus']
    
    # 6. STOCHASTICITY (noise)
    stoch = model['stochasticity']
    
    # Base noise (Gaussian)
    base_noise = np.random.normal(0, stoch['base_noise'])
    score += base_noise
    
    # Individual variation (random effect)
    if individual_effect is None:
        individual_effect = np.random.normal(0, stoch['individual_variation'])
    score += individual_effect
    
    # Measurement error (already applied, but adds uncertainty)
    measurement_noise = np.random.normal(0, stoch['measurement_error'])
    score += measurement_noise
    
    # 7. CONVERT TO PROBABILITY (sigmoid)
    probability = 1.0 / (1.0 + np.exp(-score))
    
    # Limit between min and max
    probability = np.clip(
        probability,
        model['min_probability'],
        model['max_probability']
    )
    
    return probability, individual_effect


def augment_patient_data(original_data, feature_name):
    """
    Applies data augmentation with small perturbations
    
    PRESERVA REALISMO: Perturbations small (does not create unrealistic data)
    
    Args:
        original_data: Original value
        feature_name: Feature name
    
    Returns:
        Augmented value
    """
    if not DIVERSITY_AND_BALANCING['augmentation']['enabled']:
        return original_data
    
    perturbations = DIVERSITY_AND_BALANCING['augmentation']['perturbations']
    
    # Map feature names
    feature_map = {
        'Heart Rate': 'heart_rate',
        'Systolic blood pressure': 'systolic_bp',
        'Diastolic blood pressure': 'diastolic_bp',
        'Respiratory Rate': 'respiratory_rate',
        'Temperature': 'temperature',
        'SpO2': 'spo2',
        'Lactate': 'lactate',
        'Creatinine': 'creatinine',
        'White blood cell count': 'wbc',
        'Hemoglobin': 'hemoglobin',
        'Glucose': 'glucose',
        'Potassium': 'potassium',
        'Sodium': 'sodium',
        'Bicarbonate': 'bicarbonate',
        'Blood Urea Nitrogen': 'bun'
    }
    
    mapped_name = feature_map.get(feature_name)
    if mapped_name not in perturbations:
        return original_data
    
    params = perturbations[mapped_name]
    std = params['std']
    max_delta = params['max_delta']
    
    # Generate perturbation
    perturbation = np.random.normal(0, std)
    perturbation = np.clip(perturbation, -max_delta, max_delta)
    
    augmented_value = original_data + perturbation
    
    # Ensure physiological limits
    if feature_name in PATHOLOGICAL_RANGES:
        min_val, max_val = PATHOLOGICAL_RANGES[feature_name]
        augmented_value = np.clip(augmented_value, min_val, max_val)
    
    return augmented_value


def should_augment_case(patient_died):
    """
    Determines if case should be augmented
    
    Args:
        patient_died: If patient died
    
    Returns:
        True if case should be augmented
    """
    aug_params = DIVERSITY_AND_BALANCING['augmentation']
    
    if not aug_params['enabled']:
        return False
    
    # Augment only positive cases?
    if aug_params['augment_positive_only'] and not patient_died:
        return False
    
    # Augmentation probability
    aug_prob = aug_params['augmentation_factor'] - 1.0  # 0.5 for factor 1.5
    
    return np.random.random() < aug_prob


def generate_diverse_age_distribution():
    """
    Generates age distribution with diversity guaranteed
    
    Returns:
        Array of ages
    """
    strata = DIVERSITY_AND_BALANCING['demographic_strata']
    age_groups = strata['age_groups']
    distribution = strata['age_distribution']
    
    ages = []
    for (group_name, (min_age, max_age)), prob in zip(age_groups.items(), distribution):
        n_patients = int(N_PATIENTS * prob)
        group_ages = np.random.randint(min_age, max_age + 1, n_patients)
        ages.extend(group_ages)
    
    # Complete until N_PATIENTS if necessary
    while len(ages) < N_PATIENTS:
        # Add random age
        group_name = np.random.choice(list(age_groups.keys()), p=distribution)
        min_age, max_age = age_groups[group_name]
        ages.append(np.random.randint(min_age, max_age + 1))
    
    # Truncate if necessary
    ages = ages[:N_PATIENTS]
    
    # Shuffle
    np.random.shuffle(ages)
    
    return np.array(ages)


def determine_domain(patient_id):
    """
    Determines if patient belongs to holdout domain
    
    Args:
        patient_id: Patient ID
    
    Returns:
        'source' or 'holdout'
    """
    if not DATASET_SHIFT['enabled']:
        return 'source'
    
    # Use hash of ID for determinism
    np.random.seed(patient_id)
    is_holdout = np.random.random() < DATASET_SHIFT['holdout_fraction']
    np.random.seed(None)  # Reset seed
    
    return 'holdout' if is_holdout else 'source'


def apply_measurement_bias(value, feature_name, domain, intensity_multiplier=1.0):
    """
    Applies systematic measurement bias (dataset shift)
    
    Simula: Mensurement devices with calibration difference in holdout domain
    
    Args:
        value: Original value
        feature_name: Feature name
        domain: 'source' or 'holdout'
        intensity_multiplier: Intensity multiplier
    
    Returns:
        Value with bias applied
    """
    if domain == 'source' or not DATASET_SHIFT['shift_types']['measurement_bias']['enabled']:
        return value
    
    bias_params = DATASET_SHIFT['shift_types']['measurement_bias']
    
    # Check if feature has bias defined
    if feature_name in bias_params['vitals']:
        params = bias_params['vitals'][feature_name]
    elif feature_name in bias_params['labs']:
        params = bias_params['labs'][feature_name]
    else:
        return value
    
    # Apply systematic bias
    bias = params['bias'] * intensity_multiplier
    value_with_bias = value + bias
    
    # Apply increase in variance (less precise measurements)
    std_increase = params['std_increase']
    if std_increase > 1.0:
        # Add extra noise
        extra_noise = np.random.normal(0, abs(value) * (std_increase - 1.0) * 0.05)
        value_with_bias += extra_noise
    
    return value_with_bias


def apply_population_shift(age, sex, domain):
    """
    Applies population shift (different demographics)
    
    Args:
        age: Original age
        sex: Original sex
        domain: 'source' or 'holdout'
    
    Returns:
        Tuple (age_shifted, sex_shifted)
    """
    if domain == 'source' or not DATASET_SHIFT['shift_types']['population_shift']['enabled']:
        return age, sex
    
    pop_params = DATASET_SHIFT['shift_types']['population_shift']
    
    # Age shift
    age_shifted = age + pop_params['age_shift']
    age_shifted = int(np.clip(age_shifted, 18, 95))
    
    # Sex shift (probability of flip)
    if sex == 'F' and np.random.random() < pop_params['male_ratio_shift']:
        sex_shifted = 'M'
    else:
        sex_shifted = sex
    
    return age_shifted, sex_shifted


def apply_temporal_shift_to_missing(base_missing_prob, domain):
    """
    Applies temporal shift (different clinical practices)
    
    Args:
        base_missing_prob: Base missing probability
        domain: 'source' or 'holdout'
    
    Returns:
        Adjusted probability
    """
    if domain == 'source' or not DATASET_SHIFT['shift_types']['temporal_shift']['enabled']:
        return base_missing_prob
    
    temp_params = DATASET_SHIFT['shift_types']['temporal_shift']
    
    # Increase missing rate
    adjusted_prob = base_missing_prob * temp_params['missing_rate_change']
    
    return min(0.95, adjusted_prob)


def get_shift_intensity():
    """
    Determines shift intensity
    
    Returns:
        Intensity multiplier
    """
    intensities = DATASET_SHIFT['shift_intensity']
    
    # Choose intensity
    probs = [intensities['mild']['probability'],
             intensities['moderate']['probability'],
             intensities['severe']['probability']]
    
    intensity_type = np.random.choice(['mild', 'moderate', 'severe'], p=probs)
    
    return intensities[intensity_type]['multiplier']


def get_documentation_quality(timestamp):
    """
    Determines documentation quality based on shift and day of week
    
    REALITY: Documentation quality varies by shift and weekend
    - Day: Best documentation (more staff)
    - Night: Worst documentation (less staff)
    - Weekend: Worst documentation
    
    Args:
        timestamp: Timestamp of measurement
    
    Returns:
        Dict with completeness and accuracy
    """
    hour = timestamp.hour
    is_weekend = timestamp.weekday() >= 5  # Saturday=5, Sunday=6
    
    # Determine shift
    if 7 <= hour < 15:
        shift = 'day'
    elif 15 <= hour < 23:
        shift = 'evening'
    else:
        shift = 'night'
    
    # Get base quality by shift
    quality = DOCUMENTATION_PATTERNS['by_shift'][shift].copy()
    
    # Adjust for weekend
    if is_weekend:
        weekend_quality = DOCUMENTATION_PATTERNS['by_weekday']['weekend']
        quality['completeness'] = min(quality['completeness'], weekend_quality['completeness'])
        quality['accuracy'] = min(quality['accuracy'], weekend_quality['accuracy'])
    
    return quality


def apply_documentation_quality(value, timestamp):
    """
    Applies documentation quality effects
    
    - Completeness: Probability of measurement being recorded
    - Accuracy: Precision of recorded value
    
    Args:
        value: Original value
        timestamp: Timestamp of measurement
    
    Returns:
        Tuple (value, should_record)
    """
    quality = get_documentation_quality(timestamp)
    
    # Completeness: Decide if records
    should_record = np.random.random() < quality['completeness']
    
    if not should_record:
        return None, False
    
    # Accuracy: Add documentation error
    # The lower the accuracy, the greater the error
    accuracy_error = 1.0 - quality['accuracy']
    documentation_noise = np.random.normal(0, abs(value) * accuracy_error * 0.1)
    value_documented = value + documentation_noise
    
    return value_documented, True


def generate_patient_characteristics(patient_id, age, sex, condition):
    """
    Generates UNIQUE patient characteristics (Anti-Overfitting)
    
    AVOID: All patients with sepsis having EXACTLY +25 bpm
    CREATE: Each patient has their own baseline and deterioration rate
    
    Args:
        patient_id: Patient ID
        age: Age
        sex: Sex
        condition: Clinical condition
    
    Returns:
        Dict with individual characteristics
    """
    if not INDIVIDUAL_VARIABILITY['enabled']:
        return None
    
    # Seed per patient (reproducible but unique)
    np.random.seed(patient_id)
    
    characteristics = {
        # Unique baseline offsets
        'baseline_offsets': {},
        
        # Individual deterioration rate
        'deterioration_rate': np.random.normal(
            INDIVIDUAL_VARIABILITY['deterioration_rate']['mean'],
            INDIVIDUAL_VARIABILITY['deterioration_rate']['std']
        ),
        
        # Condition sensitivity
        'condition_sensitivity': np.random.normal(
            INDIVIDUAL_VARIABILITY['condition_sensitivity']['mean'],
            INDIVIDUAL_VARIABILITY['condition_sensitivity']['std']
        ),
        
        # Individual progression pattern
        'progression_pattern': None,
        'progression_params': {}
    }
    
    # Generate baseline offsets
    for feature, params in INDIVIDUAL_VARIABILITY['baseline_offsets'].items():
        offset = np.random.normal(0, params['std'])
        characteristics['baseline_offsets'][feature] = offset
    
    # Choose progression pattern
    patterns = INDIVIDUAL_VARIABILITY['progression_patterns']
    pattern_names = list(patterns.keys())
    pattern_probs = [patterns[p]['probability'] for p in pattern_names]
    
    chosen_pattern = np.random.choice(pattern_names, p=pattern_probs)
    characteristics['progression_pattern'] = chosen_pattern
    
    # Generate pattern parameters
    pattern_config = patterns[chosen_pattern]
    
    if chosen_pattern == 'exponential':
        characteristics['progression_params']['rate'] = np.random.uniform(*pattern_config['rate_range'])
        
    elif chosen_pattern == 'sigmoid':
        characteristics['progression_params']['inflection'] = np.random.uniform(*pattern_config['inflection_range'])
        characteristics['progression_params']['steepness'] = np.random.uniform(*pattern_config['steepness_range'])
        
    elif chosen_pattern == 'linear':
        characteristics['progression_params']['rate'] = np.random.uniform(*pattern_config['rate_range'])
        
    elif chosen_pattern == 'step':
        characteristics['progression_params']['step_time'] = np.random.uniform(*pattern_config['step_time_range'])
        characteristics['progression_params']['magnitude'] = np.random.uniform(*pattern_config['step_magnitude_range'])
        
    elif chosen_pattern == 'oscillating':
        characteristics['progression_params']['base_rate'] = np.random.uniform(*pattern_config['base_rate_range'])
        characteristics['progression_params']['amplitude'] = pattern_config['amplitude']
        characteristics['progression_params']['period'] = pattern_config['period']
    
    # Reset seed
    np.random.seed(None)
    
    return characteristics


def apply_temporal_progression_nonlinear(value, hours_elapsed, patient_chars):
    """
    Applies nonlinear temporal progression (Anti-Overfitting)
    
    AVOID: Progression always linear (value * (1 + 0.3 * t))
    CREATE: Multiple deterioration patterns (exponential, sigmoid, step, etc.)
    
    Args:
        value: Base value
        hours_elapsed: Hours elapsed since admission
        patient_chars: Patient characteristics
    
    Returns:
        Value with progression applied
    """
    if patient_chars is None:
        # Fallback to linear progression
        return value * (1 + 0.3 * hours_elapsed / 48)
    
    pattern = patient_chars['progression_pattern']
    params = patient_chars['progression_params']
    t = hours_elapsed
    
    if pattern == 'exponential':
        rate = params['rate']
        progression = np.exp(rate * t / 48) - 1
        
    elif pattern == 'sigmoid':
        inflection = params['inflection']
        steepness = params['steepness']
        progression = 1 / (1 + np.exp(-steepness * (t - inflection))) - 0.5
        
    elif pattern == 'linear':
        rate = params['rate']
        progression = rate * t / 48
        
    elif pattern == 'step':
        step_time = params['step_time']
        magnitude = params['magnitude']
        progression = magnitude if t > step_time else 0
        
    elif pattern == 'oscillating':
        base_rate = params['base_rate']
        amplitude = params['amplitude']
        period = params['period']
        progression = base_rate * t / 48 + amplitude * np.sin(2 * np.pi * t / period)
    
    else:
        # Fallback
        progression = 0.3 * t / 48
    
    return value * (1 + progression)


def sample_stochastic_correlation(feature1, feature2):
    """
    Samples stochastic correlation (Anti-Overfitting)
    
    AVOID: Correlation always 0.45
    CREATE: Correlation varies (e.g., 0.35-0.55)
    
    Args:
        feature1: First feature name
        feature2: Second feature name
    
    Returns:
        Sampled correlation
    """
    if not STOCHASTIC_CORRELATIONS['enabled']:
        # Fallback for fixed correlations
        if feature1 in CORRELATIONS.get('vitals', {}) and feature2 in CORRELATIONS['vitals'][feature1]:
            return CORRELATIONS['vitals'][feature1][feature2]
        return 0.0
    
    # Look for correlation in the configuration
    for category in ['vitals', 'labs', 'vitals_labs']:
        if category not in STOCHASTIC_CORRELATIONS:
            continue
            
        corr_dict = STOCHASTIC_CORRELATIONS[category]
        
        # Try both directions
        key1 = (feature1, feature2)
        key2 = (feature2, feature1)
        
        if key1 in corr_dict:
            config = corr_dict[key1]
        elif key2 in corr_dict:
            config = corr_dict[key2]
        else:
            continue
        
        # Sample correlation
        base = config['base']
        std = config['std']
        corr = np.random.normal(base, std)
        
        # Limit between -1 and 1
        corr = np.clip(corr, -0.99, 0.99)
        
        return corr
    
    return 0.0


def apply_condition_impact(value, feature_name, condition, apply_chaos=False):
    """
    Applies condition impact
    
    Args:
        value: Base value
        feature_name: Feature name
        condition: Patient condition
        apply_chaos: If True, breaks correlations (30% of cases)
    
    Returns:
        Value with condition impact applied
    """
    # CONTROLLED CHAOS: 30% of cases ignore condition impact
    if apply_chaos and np.random.random() < ENTROPY_PARAMS['chaos_probability']:
        # Return value without impact (breaks correlation)
        return value
    
    if condition not in CONDITION_IMPACTS:
        return value
    
    impacts = CONDITION_IMPACTS[condition]
    
    if feature_name in impacts:
        multiplier = impacts[feature_name]
        return value * multiplier
    
    return value


def apply_physiological_limits(value, feature_name):
    """
    Applies physiological limits
    
    Args:
        value: Value to limit
        feature_name: Feature name
    
    Returns:
        Value within physiological limits
    """
    if feature_name in PATHOLOGICAL_RANGES:
        min_val, max_val = PATHOLOGICAL_RANGES[feature_name]
        return np.clip(value, min_val, max_val)
    
    return value


# ============================================================================
# 1. PATIENTS.csv
# ============================================================================
print("[1/7] Generating PATIENTS.csv...")

# Generate age distribution with guaranteed DIVERSITY
ages = generate_diverse_age_distribution()

patients = pd.DataFrame({
    'SUBJECT_ID': range(10000, 10000 + N_PATIENTS),
    'GENDER': np.random.choice(['M', 'F'], N_PATIENTS, p=[0.52, 0.48]),
    'AGE': ages,
    'DOB': [datetime(2100, 1, 1) - timedelta(days=int(age*365.25)) for age in ages],
    'DOD': [None] * N_PATIENTS
})

# Determine domain (source vs holdout) for each patient
patients['DOMAIN'] = [determine_domain(sid) for sid in patients['SUBJECT_ID']]

# Apply population shift to holdout domain
for idx, row in patients.iterrows():
    if row['DOMAIN'] == 'holdout':
        age_shifted, sex_shifted = apply_population_shift(row['AGE'], row['GENDER'], 'holdout')
        patients.at[idx, 'AGE'] = age_shifted
        patients.at[idx, 'GENDER'] = sex_shifted

print(f"   Age distribution:")
print(f"      18-45 years: {((ages >= 18) & (ages < 45)).sum()} ({((ages >= 18) & (ages < 45)).sum()/len(ages):.1%})")
print(f"      45-65 years: {((ages >= 45) & (ages < 65)).sum()} ({((ages >= 45) & (ages < 65)).sum()/len(ages):.1%})")
print(f"      65-80 years: {((ages >= 65) & (ages < 80)).sum()} ({((ages >= 65) & (ages < 80)).sum()/len(ages):.1%})")
print(f"      80-95 years: {((ages >= 80) & (ages <= 95)).sum()} ({((ages >= 80) & (ages <= 95)).sum()/len(ages):.1%})")
print()
print(f"   Dataset Shift:")
print(f"      Source domain: {(patients['DOMAIN'] == 'source').sum()} ({(patients['DOMAIN'] == 'source').sum()/len(patients):.1%})")
print(f"      Holdout domain: {(patients['DOMAIN'] == 'holdout').sum()} ({(patients['DOMAIN'] == 'holdout').sum()/len(patients):.1%})")
print()

# Generate individual characteristics per patient (Anti-Overfitting)
print("   Generating individual characteristics per patient...")
patient_characteristics = {}
for idx, row in patients.iterrows():
    subject_id = row['SUBJECT_ID']
    # Generate unique characteristics (without condition yet, will be defined later)
    chars = generate_patient_characteristics(subject_id, row['AGE'], row['GENDER'], None)
    patient_characteristics[subject_id] = chars

print(f"   ✓ {len(patient_characteristics)} patients with unique characteristics")
print(f"      Progression patterns:")
if patient_characteristics:
    patterns = [chars['progression_pattern'] for chars in patient_characteristics.values() if chars]
    from collections import Counter
    pattern_counts = Counter(patterns)
    for pattern, count in pattern_counts.items():
        print(f"         {pattern}: {count} ({count/len(patterns):.1%})")

# Stochastic mortality (base on multivariate model)
# NOTE: Real mortality will be determined during admission generation
# based on clinical features (vitals + labs)
# Here we only generate initial probability based on demographics

mortality_risk_initial = []
for idx, row in patients.iterrows():
    # Demographic features only
    demo_features = {
        'age': row['AGE'],
        'sex': row['GENDER'],
        # Clinical features will be average values (placeholder)
        'heart_rate': 80,
        'systolic_bp': 120,
        'spo2': 97,
        'temperature': 37.0,
        'lactate': 1.5,
        'creatinine': 1.0,
        'wbc': 8.0,
        'respiratory_rate': 16
    }
    
    # Calculate initial risk (will be refined by admission)
    risk, _ = calculate_mortality_risk_stochastic(demo_features, icu_day=3)
    mortality_risk_initial.append(risk)

# Use initial risk to determine if patient has DOD
# (death at some point, not necessarily at admission)
mortality_mask = np.random.random(N_PATIENTS) < np.array(mortality_risk_initial)
patients.loc[mortality_mask, 'DOD'] = [
    datetime(2100, 1, 1) + timedelta(days=int(x)) 
    for x in np.random.randint(0, 365*2, mortality_mask.sum())
]

# Save initial risk for later use
patients['initial_mortality_risk'] = mortality_risk_initial

patients.to_csv(f"{OUTPUT_DIR}/PATIENTS.csv", index=False)
print(f"   ✓ {len(patients)} patients generated")
print(f"   ✓ Mortality rate: {mortality_mask.mean():.1%}")


# ============================================================================
# 2. ADMISSIONS.csv
# ============================================================================
print("[2/7] Generating ADMISSIONS.csv...")

admissions_list = []
hadm_id = 20000

for idx, row in patients.iterrows():
    subject_id = row['SUBJECT_ID']
    patient_died = pd.notna(row['DOD'])
    
    if patient_died:
        n_admissions = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
    else:
        n_admissions = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
    
    for i in range(n_admissions):
        admit_time = datetime(2100, 1, 1) + timedelta(days=np.random.randint(0, 365*3))
        
        # Last admission can be fatal
        is_fatal_admission = (i == n_admissions - 1) and patient_died
        
        if is_fatal_admission:
            # Death during admission (1-30 days)
            los_days = np.random.randint(1, 30)
            discharge_time = admit_time + timedelta(days=los_days)
            true_hospital_expire_flag = 1
        else:
            los_days = np.random.randint(1, 14)
            discharge_time = admit_time + timedelta(days=los_days)
            true_hospital_expire_flag = 0
        
        # Create dict with admission data
        admission_data = {
            'HADM_ID': hadm_id,
            'SUBJECT_ID': subject_id,
            'ADMITTIME': admit_time,
            'DISCHTIME': discharge_time,
            'HOSPITAL_EXPIRE_FLAG': true_hospital_expire_flag,
            'TRUE_LABEL': true_hospital_expire_flag  # Save true label
        }
        
        # APPLY LABEL NOISE AND AMBIGUITY
        # 1. Censoring (transfer, AMA discharge, etc.)
        admission_data = apply_censoring(admission_data)
        
        if admission_data['censored']:
            # Patient censored: use observed outcome
            admission_data['HOSPITAL_EXPIRE_FLAG'] = admission_data['observed_outcome']
        else:
            # 2. Clinical ambiguity (borderline cases)
            admission_data = apply_clinical_ambiguity(admission_data)
            
            # 3. Label noise (random flip)
            noisy_label, was_flipped, flip_reason = apply_label_noise(
                admission_data['HOSPITAL_EXPIRE_FLAG']
            )
            admission_data['HOSPITAL_EXPIRE_FLAG'] = noisy_label
            admission_data['label_flipped'] = was_flipped
            admission_data['flip_reason'] = flip_reason
        
        # 4. Timing uncertainty (jitter nos timestamps)
        if admission_data['HOSPITAL_EXPIRE_FLAG'] == 1:
            # Apply jitter to death time
            admission_data['DISCHTIME'] = apply_timing_uncertainty(
                discharge_time, event_type='death'
            )
        else:
            # Apply jitter to discharge time
            admission_data['DISCHTIME'] = apply_timing_uncertainty(
                discharge_time, event_type='discharge'
            )
        
        admissions_list.append(admission_data)
        
        hadm_id += 1

admissions = pd.DataFrame(admissions_list)
admissions.to_csv(f"{OUTPUT_DIR}/ADMISSIONS.csv", index=False)
print(f"   ✓ {len(admissions)} admissions generated")
print(f"   ✓ In-hospital mortality: {admissions['HOSPITAL_EXPIRE_FLAG'].mean():.1%}")


# ============================================================================
# 3. ICUSTAYS.csv
# ============================================================================
print("[3/7] Generating ICUSTAYS.csv...")

icustays_list = []
icustay_id = 30000

# Not all admissions have ICU (60-70%)
icu_admissions = admissions.sample(frac=0.65, random_state=42)

for idx, row in icu_admissions.iterrows():
    hadm_id = row['HADM_ID']
    subject_id = row['SUBJECT_ID']
    admit_time = row['ADMITTIME']
    discharge_time = row['DISCHTIME']
    died = row['HOSPITAL_EXPIRE_FLAG']
    
    # ICU usually starts right after admission
    icu_intime = admit_time + timedelta(hours=np.random.randint(0, 24))
    
    # Calculate available time
    max_available_hours = int((discharge_time - icu_intime).total_seconds() / 3600)
    
    # ICU duration (ensure low < high)
    if died:
        max_icu_hours = min(200, max_available_hours)
    else:
        max_icu_hours = min(168, max_available_hours)
    
    # Ensure we have at least 24h available
    if max_icu_hours < 24:
        max_icu_hours = 24
        icu_outtime = icu_intime + timedelta(hours=24)
    else:
        icu_los_hours = np.random.randint(24, max_icu_hours + 1)
        icu_outtime = icu_intime + timedelta(hours=icu_los_hours)
    
    # Ensure it doesn't exceed discharge time
    if icu_outtime > discharge_time:
        icu_outtime = discharge_time
    
    icu_los_hours = (icu_outtime - icu_intime).total_seconds() / 3600
    
    icustays_list.append({
        'ICUSTAY_ID': icustay_id,
        'HADM_ID': hadm_id,
        'SUBJECT_ID': subject_id,
        'INTIME': icu_intime,
        'OUTTIME': icu_outtime,
        'LOS': icu_los_hours / 24.0
    })
    
    icustay_id += 1

icustays = pd.DataFrame(icustays_list)
icustays.to_csv(f"{OUTPUT_DIR}/ICUSTAYS.csv", index=False)

# Calculate ICU mortality
icu_deaths = icustays.merge(admissions[['HADM_ID', 'HOSPITAL_EXPIRE_FLAG']], on='HADM_ID')
print(f"   ✓ {len(icustays)} ICU stays generated")
print(f"   ✓ ICU mortality: {icu_deaths['HOSPITAL_EXPIRE_FLAG'].mean():.1%}")


# ============================================================================
# 4. CHARTEVENTS.csv - WITH ADVANCED PATTERNS, SLEEP AND MEALS
# ============================================================================
print("[4/7] Generating CHARTEVENTS.csv (with advanced patterns)...")

chartevents_list = []
itemid_counter = 40000

# Mapping of ITEMIDs to vital signs
vital_signs = {
    'Heart Rate': 211,
    'Systolic blood pressure': 51,
    'Diastolic blood pressure': 8368,
    'Respiratory Rate': 618,
    'Temperature': 223761,
    'SpO2': 646
}

for idx, row in icustays.iterrows():
    icustay_id = row['ICUSTAY_ID']
    subject_id = row['SUBJECT_ID']
    hadm_id = row['HADM_ID']
    intime = row['INTIME']
    los_hours = row['LOS'] * 24
    
    # Determine if patient died
    died = admissions[admissions['HADM_ID'] == hadm_id]['HOSPITAL_EXPIRE_FLAG'].values[0]
    
    # Get patient demographics
    patient_age = patients[patients['SUBJECT_ID'] == subject_id]['AGE'].values[0]
    patient_sex = patients[patients['SUBJECT_ID'] == subject_id]['GENDER'].values[0]
    
    # HIERARCHICAL GENERATION: P(Condition | Demographics)
    condition = sample_condition_given_demographics(patient_age, patient_sex, died)
    
    # Get patient characteristics (Anti-Overfitting)
    patient_chars = patient_characteristics.get(subject_id, None)
    
    # Generate measurements every hour (with temporal jitter)
    for hour in range(0, min(int(los_hours), 48)):  # Maximum 48h
        # Apply temporal jitter (variation in spacing)
        actual_hour = apply_time_jitter(hour, max_jitter_hours=1.0)
        charttime = intime + timedelta(hours=actual_hour)
        hour_of_day = get_hour_from_timestamp(actual_hour)
        
        # Calculate temporal progress
        progress = hour / min(los_hours, 48) if died else 0.0
        
        # CORRELATED GENERATION: P(Vitals | Condition)
        # Generate all vitals at once with correlations
        vital_values = generate_vitals_given_condition(condition)
        
        # Process each vital
        for vital_name, itemid in vital_signs.items():
            value = vital_values.get(vital_name, 75.0)
            
            # 1. Apply BASELINE OFFSET individual (Anti-Overfitting)
            if patient_chars and vital_name in patient_chars['baseline_offsets']:
                value += patient_chars['baseline_offsets'][vital_name]
            
            # 2. Apply temporal progress NON-LINEAR (Anti-Overfitting)
            if died:
                # CHAOS: 30% of cases have random progression
                if np.random.random() < ENTROPY_PARAMS['chaos_probability']:
                    # Non-linear or reverse progression
                    progress = np.random.uniform(0, 1)
                
                # Use non-linear temporal progression
                if patient_chars:
                    value = apply_temporal_progression_nonlinear(value, hour, patient_chars)
                else:
                    # Fallback: Gradual deterioration LINEAR (old method)
                    if vital_name in ['Heart Rate', 'Respiratory Rate']:
                        value *= (1 + 0.3 * progress)
                    elif vital_name in ['Systolic blood pressure', 'Diastolic blood pressure']:
                        value *= (1 - 0.2 * progress)
                    elif vital_name == 'SpO2':
                        value *= (1 - 0.1 * progress)
            
            # 4. Apply circadian pattern
            value = apply_circadian_pattern(value, vital_name, hour_of_day)
            
            # 5. Apply sleep effect
            value = apply_sleep_effect(value, vital_name, hour_of_day)
            
            # 6. Apply meal effect
            value = apply_meal_effect(value, vital_name, hour_of_day)
            
            # 7. Add entropy (general noise)
            value = add_entropy(value, vital_name)
            
            # 8. Add signal-specific noise (SNR)
            value = add_signal_specific_noise(value, vital_name)
            
            # 9. Add temporal artifacts
            value = add_temporal_artifact(value, vital_name)
            
            # 10. Apply physiological limits
            value = apply_physiological_limits(value, vital_name)
            
            # 11. Calculate severity score (using values before noise)
            severity_score = calculate_severity_score(
                vital_values, {}, condition, died, progress
            )
            
            # 12. Verify realistic missingness (MAR/MNAR/MCAR)
            if should_be_missing(severity_score, 'vital_signs', hour_of_day):
                # Measurement missing
                continue
            
            # 13. Stochastic events (sudden complications)
            if died and np.random.random() < ENTROPY_PARAMS['event_probability'] * progress:
                # Sudden deterioration
                if vital_name in ['Heart Rate', 'Respiratory Rate']:
                    value *= np.random.uniform(1.2, 1.5)
                elif vital_name in ['Systolic blood pressure', 'Diastolic blood pressure']:
                    value *= np.random.uniform(0.7, 0.9)
                
                value = apply_physiological_limits(value, vital_name)
            
            chartevents_list.append({
                'ICUSTAY_ID': icustay_id,
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'ITEMID': itemid,
                'CHARTTIME': charttime,
                'VALUE': round(value, 2),
                'VALUENUM': round(value, 2)
            })

chartevents = pd.DataFrame(chartevents_list)
chartevents.to_csv(f"{OUTPUT_DIR}/CHARTEVENTS.csv", index=False)
print(f"   ✓ {len(chartevents)} vital sign events generated")


# ============================================================================
# 5. LABEVENTS.csv - WITH ADVANCED PATTERNS
# ============================================================================
print("[5/7] Generating LABEVENTS.csv (with advanced patterns)...")

labevents_list = []

lab_tests = {
    'Lactate': 50813,
    'Creatinine': 50912,
    'White blood cell count': 51301,
    'Hemoglobin': 51222,
    'Glucose': 50931,
    'Potassium': 50971,
    'Sodium': 50983,
    'Bicarbonate': 50882,
    'Blood Urea Nitrogen': 51006
}

for idx, row in icustays.iterrows():
    icustay_id = row['ICUSTAY_ID']
    subject_id = row['SUBJECT_ID']
    hadm_id = row['HADM_ID']
    intime = row['INTIME']
    los_hours = row['LOS'] * 24
    
    died = admissions[admissions['HADM_ID'] == hadm_id]['HOSPITAL_EXPIRE_FLAG'].values[0]
    
    # Get patient demographics
    patient_age = patients[patients['SUBJECT_ID'] == subject_id]['AGE'].values[0]
    patient_sex = patients[patients['SUBJECT_ID'] == subject_id]['GENDER'].values[0]
    
    # HIERARCHICAL GENERATION: P(Condition | Demographics)
    condition = sample_condition_given_demographics(patient_age, patient_sex, died)
    
    # Get patient characteristics (Anti-Overfitting)
    patient_chars = patient_characteristics.get(subject_id, None)
    
    # Labs every 4-6 hours (with temporal jitter)
    for hour in range(0, min(int(los_hours), 48), np.random.randint(4, 7)):
        # Apply temporal jitter
        actual_hour = apply_time_jitter(hour, max_jitter_hours=0.5)
        charttime = intime + timedelta(hours=actual_hour)
        hour_of_day = get_hour_from_timestamp(actual_hour)
        
        # Calculate temporal progress
        progress = hour / min(los_hours, 48) if died else 0.0
        
        # Generate vitals for this timestep (for correlation with labs)
        vital_values_for_labs = generate_vitals_given_condition(condition)
        
        # CORRELATED GENERATION: P(Labs | Condition, Vitals)
        lab_values = generate_labs_given_condition_and_vitals(condition, vital_values_for_labs)
        
        # Process each lab
        for lab_name, itemid in lab_tests.items():
            value = lab_values.get(lab_name, 100.0)
            
            # 1. Apply BASELINE OFFSET individual (Anti-Overfitting)
            if patient_chars and lab_name in patient_chars['baseline_offsets']:
                value += patient_chars['baseline_offsets'][lab_name]
            
            # 2. Apply temporal progress NON-LINEAR (Anti-Overfitting)
            if died:
                # CHAOS: 30% of cases have random progression
                if np.random.random() < ENTROPY_PARAMS['chaos_probability']:
                    progress = np.random.uniform(0, 1)
                
                # Use non-linear temporal progression
                if patient_chars:
                    value = apply_temporal_progression_nonlinear(value, hour, patient_chars)
                else:
                    # Fallback: Gradual deterioration LINEAR (old method)
                    if lab_name in ['Lactate', 'Creatinine', 'Blood Urea Nitrogen']:
                        value *= (1 + 1.5 * progress)
                    elif lab_name == 'White blood cell count':
                        value *= (1 + 0.8 * progress)
            
            value = apply_meal_effect(value, lab_name, hour_of_day)
            value = add_entropy(value, lab_name)
            
            # Add signal-specific noise
            value = add_signal_specific_noise(value, lab_name)
            
            # Add temporal artifacts
            value = add_temporal_artifact(value, lab_name)
            
            value = apply_physiological_limits(value, lab_name)
            
            # Calculate severity score
            severity_score = calculate_severity_score(
                {}, lab_values, condition, died, progress
            )
            
            # Verify realistic missingness (MAR/MNAR/MCAR)
            if should_be_missing(severity_score, 'labs', hour_of_day):
                # Measurement missing
                continue
            
            if died and np.random.random() < ENTROPY_PARAMS['event_probability'] * progress:
                if lab_name in ['Lactate', 'Creatinine']:
                    value *= np.random.uniform(1.3, 1.8)
                value = apply_physiological_limits(value, lab_name)
            
            labevents_list.append({
                'ICUSTAY_ID': icustay_id,
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'ITEMID': itemid,
                'CHARTTIME': charttime,
                'VALUE': str(round(value, 2)),
                'VALUENUM': round(value, 2)
            })

labevents = pd.DataFrame(labevents_list)
labevents.to_csv(f"{OUTPUT_DIR}/LABEVENTS.csv", index=False)
print(f"   ✓ {len(labevents)} events of laboratory generated")


# ============================================================================
# 6. PRESCRIPTIONS.csv
# ============================================================================
print("[6/7] Generating PRESCRIPTIONS.csv...")

prescriptions_list = []

medications = [
    'Norepinephrine', 'Vancomycin', 'Furosemide', 'Insulin',
    'Heparin', 'Propofol', 'Fentanyl', 'Midazolam'
]

for idx, row in icustays.iterrows():
    icustay_id = row['ICUSTAY_ID']
    subject_id = row['SUBJECT_ID']
    hadm_id = row['HADM_ID']
    intime = row['INTIME']
    
    died = admissions[admissions['HADM_ID'] == hadm_id]['HOSPITAL_EXPIRE_FLAG'].values[0]
    
    if died:
        n_meds = np.random.randint(3, 8)
    else:
        n_meds = np.random.randint(1, 5)
    
    selected_meds = np.random.choice(medications, n_meds, replace=False)
    
    for med in selected_meds:
        start_time = intime + timedelta(hours=np.random.randint(0, 12))
        
        prescriptions_list.append({
            'ICUSTAY_ID': icustay_id,
            'SUBJECT_ID': subject_id,
            'HADM_ID': hadm_id,
            'STARTDATE': start_time,
            'DRUG': med
        })

prescriptions = pd.DataFrame(prescriptions_list)
prescriptions.to_csv(f"{OUTPUT_DIR}/PRESCRIPTIONS.csv", index=False)
print(f"   ✓ {len(prescriptions)} prescriptions generated")


# ============================================================================
# 7. DIAGNOSES_ICD.csv
# ============================================================================
print("[7/7] Generating DIAGNOSES_ICD.csv...")

diagnoses_list = []

icd9_codes = {
    'sepsis': ['99591', '99592', '78552'],
    'heart_failure': ['42823', '42833', '42843'],
    'aki': ['5849', '5845', '5846'],
    'copd': ['49121', '49122', '4928'],
    'diabetes': ['25000', '25001', '25002']
}

for idx, row in admissions.iterrows():
    hadm_id = row['HADM_ID']
    subject_id = row['SUBJECT_ID']
    died = row['HOSPITAL_EXPIRE_FLAG']
    
    if died:
        condition = np.random.choice(['sepsis', 'heart_failure', 'aki'], p=[0.5, 0.3, 0.2])
        codes = icd9_codes[condition]
        n_codes = np.random.randint(2, len(codes) + 1)
    else:
        all_codes = [code for codes in icd9_codes.values() for code in codes]
        n_codes = np.random.randint(1, 3)
        codes = np.random.choice(all_codes, n_codes, replace=False)
    
    for seq_num, code in enumerate(codes[:n_codes], 1):
        diagnoses_list.append({
            'HADM_ID': hadm_id,
            'SUBJECT_ID': subject_id,
            'ICD9_CODE': code,
            'SEQ_NUM': seq_num
        })

diagnoses = pd.DataFrame(diagnoses_list)
diagnoses.to_csv(f"{OUTPUT_DIR}/DIAGNOSES_ICD.csv", index=False)
print(f"   ✓ {len(diagnoses)} diagnoses generated")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print()
print("=" * 70)
print("✅ SYNTHETIC DATA GENERATION COMPLETED!")
print("=" * 70)
print()
print("Statistics:")
print(f"  • Patients: {len(patients):,}")
print(f"  • Admissions: {len(admissions):,}")
print(f"  • ICU Stays: {len(icustays):,}")
print(f"  • Vital Events: {len(chartevents):,}")
print(f"  • Lab Events: {len(labevents):,}")
print(f"  • Prescriptions: {len(prescriptions):,}")
print(f"  • Diagnoses: {len(diagnoses):,}")
print()
print("Label Noise and Ambiguity:")
n_censored = admissions['censored'].sum() if 'censored' in admissions.columns else 0
n_ambiguous = admissions['ambiguous'].sum() if 'ambiguous' in admissions.columns else 0
n_flipped = admissions['label_flipped'].sum() if 'label_flipped' in admissions.columns else 0
true_mortality = admissions['TRUE_LABEL'].mean() if 'TRUE_LABEL' in admissions.columns else admissions['HOSPITAL_EXPIRE_FLAG'].mean()
observed_mortality = admissions['HOSPITAL_EXPIRE_FLAG'].mean()
print(f"  • Censored cases: {n_censored} ({n_censored/len(admissions):.1%})")
print(f"  • Ambiguous cases: {n_ambiguous} ({n_ambiguous/len(admissions):.1%})")
print(f"  • Label flips: {n_flipped} ({n_flipped/len(admissions):.1%})")
print(f"  • True mortality: {true_mortality:.1%}")
print(f"  • Observed mortality: {observed_mortality:.1%}")
print(f"  • Discrepancy: {abs(true_mortality - observed_mortality):.1%}")
print()
print("Enhancements implemented:")
print("  ✓ Circadian patterns (24h)")
print("  ✓ Sleep simulation (23h-6h)")
print("  ✓ Meal effects (7h/12h/18h)")
print("  ✓ Extreme noise and entropy (±50%)")
print("  ✓ Stochastic events (40%)")
print("  ✓ Controlled chaos (30% break correlations)")
print("  ✓ Random temporal progression (30%)")
print("  ✓ Impact quantified by condition")
print("  ✓ Physiological limits validated")
print()
print("Advanced temporal variability:")
print("  ✓ Signal-specific noise (SNR realistic ±0.5-8%)")
print("  ✓ Time jitter (±10% spacing variation)")
print("  ✓ Dropout temporal (10% missing)")
print("  ✓ Isolated artifacts (3% probability)")
print()
print("Realistic missingness (MCAR/MAR/MNAR):")
print("  ✓ MCAR: 30% completely random")
print("  ✓ MAR: 50% dependent on severity score")
print("  ✓ MNAR: 20% dependent on values non-observed")
print("  ✓ Realidade: Patients with higher severity have more measurements")
print("  ✓ Temporal patterns: +30% missing at night, +15% at turns")
print()
print("Realistic correlations:")
print("  ✓ Hierarchical generation: Demographics → Condition → Vitals → Labs")
print("  ✓ Multivariate covariance matrix (vitals and labs)")
print("  ✓ Vital correlations: FC↔RR (0.45), PA↔FC (-0.30), PA↔SpO2 (0.25)")
print("  ✓ Lab correlations: Cr↔BUN (0.80), Lac↔HCO3 (-0.65)")
print("  ✓ Vital-lab correlations: FC↔Lac (0.50), PA↔Lac (-0.55)")
print("  ✓ P(Condition|Age,Sex): Older +80-120% risk")
print()
print("Label noise and clinical ambiguity:")
print("  ✓ Label flip: 1% (0→1), 2% (1→0)")
print("  ✓ Censoring: 8% (transfer, AMA discharge, dropout)")
print("  ✓ Ambiguity: 8% (palliative care, withdrawal of support)")
print("  ✓ Timing uncertainty: ±6h death, ±4h discharge")
print("  ✓ Documentation delay: 15% cases (2-24h)")
print()
print("Stochastic mortality model:")
print("  ✓ Avoid patterns (non-deterministic)")
print("  ✓ Non-linear interactions: shock, sepsis, resp failure")
print("  ✓ Non-linear terms: age^2, lactate^exp, SpO2^-1")
print("  ✓ Temporal effects: deterioration, early mortality, recovery")
print("  ✓ Stochasticity: ±15% noise, individual variation")
print()
print("Diversity and balance:")
print("  ✓ Age distribution (18-45, 45-65, 65-80, 80-95)")
print("  ✓ Target mortality: 18% (vs ~10-12% natural)")
print("  ✓ Data augmentation: +50% positive cases with perturbations")
print("  ✓ Small perturbations: ±2-5 vitals, ±0.3-2 labs")
print("  ✓ Condition diversity: sepsis, HF, AKI, resp, multi-organ")
print()
print("Dataset shift (holdout domain):")
print("  ✓ 15% holdout (different distribution)")
print("  ✓ Measurement bias: FC +3, PA -5, SpO2 -1, Lac +0.3")
print("  ✓ Population shift: +5 years, +5% men, +10% severe")
print("  ✓ Temporal shift: -15% measurements, +20% missing")
print("  ✓ Evaluate generalization and detect memorization")
print()
print("Documentation quality:")
print("  ✓ Turn-based: Day (95%), Afternoon (90%), Night (85%)")
print("  ✓ Weekends: 87% completeness, 92% accuracy")
print("  ✓ Documentation error proportional to quality")
print("  ✓ Simulate real variation with staff availability")
print()
print("Individual variability (anti-overfitting):")
print("  ✓ Unique baseline offsets: ±5 bpm FC, ±8 mmHg PA")
print("  ✓ Individual deterioration rate: 0.3 ± 0.1")
print("  ✓ 5 progression patterns: exponential, sigmoid, linear, step, oscillating")
print("  ✓ Condition sensitivity: 1.0 ± 0.2")
print()
print("Stochastic correlations (anti-overfitting):")
print("  ✓ Correlations vary per patient: 0.45 ± 0.10")
print("  ✓ Covariance matrix noise: ±10%")
print("  ✓ Avoid fixed patterns memorizable")
print()
print(f"Files saved to: {OUTPUT_DIR}/")
print()
print("Next steps:")
print("  1. Process data: python scripts/process_synthetic_data.py")
print("  2. Train models + Generate visualizations:")
print("     ./scripts/run_plots_and_report.sh")
print()
print("Or run complete pipeline (data + training + validation):")
print("  ./scripts/regenerate_data.sh")
print()
print("For K-Fold validation:")
print("  ./scripts/run_validation_and_report.sh")
print("=" * 70)
