#!/usr/bin/env python3
"""
Synthetic Data Processing Script for mimic3-benchmarks format
Adapts synthetic data to the format expected by the benchmark pipeline
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

# Configurations
INPUT_DIR = "data/mimic3_synthetic_v3/raw"
OUTPUT_DIR = "data/root"
BENCHMARK_DIR = "data/in-hospital-mortality"

print("=" * 70)
print("SYNTHETIC DATA PROCESSING SCRIPT - MIMIC3-BENCHMARKS FORMAT")
print("=" * 70)
print()

# Create directory structure
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/test", exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(f"{BENCHMARK_DIR}/train", exist_ok=True)
os.makedirs(f"{BENCHMARK_DIR}/test", exist_ok=True)

# ============================================================================
# Load synthetic data
# ============================================================================
print("[1/5] Loading synthetic data...")

patients = pd.read_csv(f"{INPUT_DIR}/PATIENTS.csv")
admissions = pd.read_csv(f"{INPUT_DIR}/ADMISSIONS.csv", parse_dates=['ADMITTIME', 'DISCHTIME'])
icustays = pd.read_csv(f"{INPUT_DIR}/ICUSTAYS.csv", parse_dates=['INTIME', 'OUTTIME'])
chartevents = pd.read_csv(f"{INPUT_DIR}/CHARTEVENTS.csv", parse_dates=['CHARTTIME'])
labevents = pd.read_csv(f"{INPUT_DIR}/LABEVENTS.csv", parse_dates=['CHARTTIME'])
prescriptions = pd.read_csv(f"{INPUT_DIR}/PRESCRIPTIONS.csv", parse_dates=['STARTDATE'])
diagnoses = pd.read_csv(f"{INPUT_DIR}/DIAGNOSES_ICD.csv")

print(f"   ✓ {len(patients)} patients")
print(f"   ✓ {len(icustays)} ICU stays")
print()

# ============================================================================
# Create structure per patient (SUBJECT_ID)
# ============================================================================
print("[2/5] Creating patient structure...")

for subject_id in patients['SUBJECT_ID'].unique():
    subject_dir = f"{OUTPUT_DIR}/{subject_id}"
    os.makedirs(subject_dir, exist_ok=True)
    
    # Filter patient data
    patient_admissions = admissions[admissions['SUBJECT_ID'] == subject_id]
    patient_icustays = icustays[icustays['SUBJECT_ID'] == subject_id]
    patient_diagnoses = diagnoses[diagnoses['SUBJECT_ID'] == subject_id]
    
    # Save stays.csv
    stays_data = patient_icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'LOS']].copy()
    stays_data.to_csv(f"{subject_dir}/stays.csv", index=False)
    
    # Save diagnoses.csv
    patient_diagnoses.to_csv(f"{subject_dir}/diagnoses.csv", index=False)
    
    # Create events.csv (combining chartevents and labevents)
    patient_chartevents = chartevents[chartevents['SUBJECT_ID'] == subject_id]
    patient_labevents = labevents[labevents['SUBJECT_ID'] == subject_id]
    
    # Combine events
    events_list = []
    
    # Add chartevents
    for _, event in patient_chartevents.iterrows():
        events_list.append({
            'SUBJECT_ID': event['SUBJECT_ID'],
            'HADM_ID': event['HADM_ID'],
            'ICUSTAY_ID': event['ICUSTAY_ID'],
            'CHARTTIME': event['CHARTTIME'],
            'ITEMID': event['ITEMID'],
            'VALUE': event['VALUE']
        })
    
    # Add labevents
    for _, event in patient_labevents.iterrows():
        events_list.append({
            'SUBJECT_ID': event['SUBJECT_ID'],
            'HADM_ID': event['HADM_ID'],
            'ICUSTAY_ID': event.get('ICUSTAY_ID', None),
            'CHARTTIME': event['CHARTTIME'],
            'ITEMID': event['ITEMID'],
            'VALUE': event['VALUE']
        })
    
    if events_list:
        events_df = pd.DataFrame(events_list)
        events_df = events_df.sort_values('CHARTTIME')
        events_df.to_csv(f"{subject_dir}/events.csv", index=False)

print(f"   ✓ Structure created for {len(patients)} patients")
print()

# ============================================================================
# Create episodes (per ICUSTAY)
# ============================================================================
print("[3/5] Creating episodes per ICU stay...")

episode_count = 0

for _, icu in icustays.iterrows():
    subject_id = icu['SUBJECT_ID']
    icustay_id = icu['ICUSTAY_ID']
    hadm_id = icu['HADM_ID']
    
    subject_dir = f"{OUTPUT_DIR}/{subject_id}"
    
    # Get patient information
    patient = patients[patients['SUBJECT_ID'] == subject_id].iloc[0]
    admission = admissions[admissions['HADM_ID'] == hadm_id].iloc[0]
    
    # Calculate age at admission
    dob = pd.to_datetime(patient['DOB'])
    admit_time = icu['INTIME']
    age = (admit_time - dob).days / 365.25
    
    # Determine outcome (mortality)
    mortality = admission['HOSPITAL_EXPIRE_FLAG']
    
    # Create episode file
    episode_data = {
        'Icustay': icustay_id,
        'Age': round(age, 2),
        'Gender': 1 if patient['GENDER'] == 'M' else 0,
        'Height': round(np.random.normal(170, 10), 2),  # Simulated height
        'Weight': round(np.random.normal(75, 15), 2),   # Simulated weight
        'Ethnicity': admission.get('ETHNICITY', 'UNKNOWN'),
        'Mortality': mortality,
        'Length of Stay': round(icu['LOS'], 4)
    }
    
    episode_df = pd.DataFrame([episode_data])
    episode_df.to_csv(f"{subject_dir}/episode{episode_count + 1}.csv", index=False)
    
    # Create timeseries for episode
    # Filter events within 48h window (for mortality prediction)
    window_end = icu['INTIME'] + timedelta(hours=48)
    
    icu_chartevents = chartevents[
        (chartevents['ICUSTAY_ID'] == icustay_id) &
        (chartevents['CHARTTIME'] >= icu['INTIME']) &
        (chartevents['CHARTTIME'] <= window_end)
    ].copy()
    
    icu_labevents = labevents[
        (labevents['HADM_ID'] == hadm_id) &
        (labevents['CHARTTIME'] >= icu['INTIME']) &
        (labevents['CHARTTIME'] <= window_end)
    ].copy()
    
    # Create timeseries
    timeseries_list = []
    
    # Process chartevents
    for _, event in icu_chartevents.iterrows():
        hours = (event['CHARTTIME'] - icu['INTIME']).total_seconds() / 3600
        timeseries_list.append({
            'Hours': round(hours, 4),
            'Itemid': event['ITEMID'],
            'Value': event['VALUE']
        })
    
    # Process labevents
    for _, event in icu_labevents.iterrows():
        hours = (event['CHARTTIME'] - icu['INTIME']).total_seconds() / 3600
        timeseries_list.append({
            'Hours': round(hours, 4),
            'Itemid': event['ITEMID'],
            'Value': event['VALUE']
        })
    
    if timeseries_list:
        timeseries_df = pd.DataFrame(timeseries_list)
        timeseries_df = timeseries_df.sort_values('Hours')
        timeseries_df.to_csv(f"{subject_dir}/episode{episode_count + 1}_timeseries.csv", index=False)
    
    episode_count += 1

print(f"   ✓ {episode_count} episodes created")
print()

# ============================================================================
# Split into train/val/test using HASH per patient (NO LEAKAGE)
# ============================================================================
print("[4/5] Splitting into train/val/test by patient hash...")

def hash_patient_id(patient_id):
    """Deterministic hash of patient_id for consistent split"""
    return hash(str(patient_id)) % 100

all_subjects = list(patients['SUBJECT_ID'].unique())

# Split by hash: 70% train, 15% val, 15% test
train_subjects = [s for s in all_subjects if hash_patient_id(s) < 70]
val_subjects = [s for s in all_subjects if 70 <= hash_patient_id(s) < 85]
test_subjects = [s for s in all_subjects if hash_patient_id(s) >= 85]

print(f"   ✓ Train: {len(train_subjects)} patients ({len(train_subjects)/len(all_subjects)*100:.1f}%)")
print(f"   ✓ Val:   {len(val_subjects)} patients ({len(val_subjects)/len(all_subjects)*100:.1f}%)")
print(f"   ✓ Test:  {len(test_subjects)} patients ({len(test_subjects)/len(all_subjects)*100:.1f}%)")

# Check intersection (should be 0)
train_set = set(train_subjects)
val_set = set(val_subjects)
test_set = set(test_subjects)

assert len(train_set & val_set) == 0, "LEAKAGE: Train and Val share patients!"
assert len(train_set & test_set) == 0, "LEAKAGE: Train and Test share patients!"
assert len(val_set & test_set) == 0, "LEAKAGE: Val and Test share patients!"

print(f"   ✓ Verification: ZERO leakage between splits")
print()

# ============================================================================
# Create listfiles for in-hospital mortality
# ============================================================================
print("[5/5] Creating listfiles for mortality...")

train_list = []
val_list = []
test_list = []

for subject_id in train_subjects:
    subject_dir = f"{OUTPUT_DIR}/{subject_id}"
    
    # Find all patient episodes
    episode_files = [f for f in os.listdir(subject_dir) if f.startswith('episode') and not f.endswith('_timeseries.csv')]
    
    for episode_file in episode_files:
        episode_df = pd.read_csv(f"{subject_dir}/{episode_file}")
        
        # Check if has corresponding timeseries
        episode_num = episode_file.replace('episode', '').replace('.csv', '')
        timeseries_file = f"episode{episode_num}_timeseries.csv"
        
        if os.path.exists(f"{subject_dir}/{timeseries_file}"):
            train_list.append({
                'stay': f"{subject_id}/episode{episode_num}_timeseries.csv",
                'y_true': int(episode_df['Mortality'].iloc[0])
            })

for subject_id in val_subjects:
    subject_dir = f"{OUTPUT_DIR}/{subject_id}"
    
    episode_files = [f for f in os.listdir(subject_dir) if f.startswith('episode') and not f.endswith('_timeseries.csv')]
    
    for episode_file in episode_files:
        episode_df = pd.read_csv(f"{subject_dir}/{episode_file}")
        
        episode_num = episode_file.replace('episode', '').replace('.csv', '')
        timeseries_file = f"episode{episode_num}_timeseries.csv"
        
        if os.path.exists(f"{subject_dir}/{timeseries_file}"):
            val_list.append({
                'stay': f"{subject_id}/episode{episode_num}_timeseries.csv",
                'y_true': int(episode_df['Mortality'].iloc[0])
            })

for subject_id in test_subjects:
    subject_dir = f"{OUTPUT_DIR}/{subject_id}"
    
    episode_files = [f for f in os.listdir(subject_dir) if f.startswith('episode') and not f.endswith('_timeseries.csv')]
    
    for episode_file in episode_files:
        episode_df = pd.read_csv(f"{subject_dir}/{episode_file}")
        
        episode_num = episode_file.replace('episode', '').replace('.csv', '')
        timeseries_file = f"episode{episode_num}_timeseries.csv"
        
        if os.path.exists(f"{subject_dir}/{timeseries_file}"):
            test_list.append({
                'stay': f"{subject_id}/episode{episode_num}_timeseries.csv",
                'y_true': int(episode_df['Mortality'].iloc[0])
            })

# Create validation directory
os.makedirs(f"{BENCHMARK_DIR}/val", exist_ok=True)

# Save listfiles
train_listfile = pd.DataFrame(train_list)
val_listfile = pd.DataFrame(val_list)
test_listfile = pd.DataFrame(test_list)

train_listfile.to_csv(f"{BENCHMARK_DIR}/train/listfile.csv", index=False)
val_listfile.to_csv(f"{BENCHMARK_DIR}/val/listfile.csv", index=False)
test_listfile.to_csv(f"{BENCHMARK_DIR}/test/listfile.csv", index=False)

print(f"   ✓ Train listfile: {len(train_list)} episodes")
print(f"   ✓ Val listfile:   {len(val_list)} episodes")
print(f"   ✓ Test listfile:  {len(test_list)} episodes")
print()

# Mortality statistics
train_mortality_rate = train_listfile['y_true'].mean()
val_mortality_rate = val_listfile['y_true'].mean()
test_mortality_rate = test_listfile['y_true'].mean()

print("Mortality Statistics:")
print(f"   • Train: {train_mortality_rate:.1%} ({train_listfile['y_true'].sum()}/{len(train_list)})")
print(f"   • Val:   {val_mortality_rate:.1%} ({val_listfile['y_true'].sum()}/{len(val_list)})")
print(f"   • Test:  {test_mortality_rate:.1%} ({test_listfile['y_true'].sum()}/{len(test_list)})")
print()

# ============================================================================
# Create feature vocabulary
# ============================================================================
print("Creating feature vocabulary...")

all_itemids = pd.concat([
    chartevents['ITEMID'],
    labevents['ITEMID']
]).unique()

vocab = {int(itemid): f"ITEM_{itemid}" for itemid in all_itemids}

with open(f"{BENCHMARK_DIR}/vocab.json", 'w') as f:
    json.dump(vocab, f, indent=2)

print(f"   ✓ Vocabulary with {len(vocab)} features")
print()

# ============================================================================
# Final summary
# ============================================================================
print("=" * 70)
print("✅ PROCESSING COMPLETED!")
print("=" * 70)
print()
print("Structure created:")
print(f"  • {OUTPUT_DIR}/ - Patient data")
print(f"  • {BENCHMARK_DIR}/train/ - Training dataset")
print(f"  • {BENCHMARK_DIR}/test/ - Test dataset")
print()
print("Important files:")
print(f"  • {BENCHMARK_DIR}/train/listfile.csv")
print(f"  • {BENCHMARK_DIR}/test/listfile.csv")
print(f"  • {BENCHMARK_DIR}/vocab.json")
print()
print("Next step: Train models with ./scripts/run_plots_and_report.sh")
print("=" * 70)
