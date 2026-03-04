import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000  # 1000 patients

data = {
    # Patient basic info
    'patient_id': range(1, n+1),
    'age': np.random.randint(18, 75, n),
    'gender': np.random.choice(['Male', 'Female'], n),
    'bmi': np.round(np.random.uniform(17, 40, n), 1),

    # Trial-related info
    'trial_phase': np.random.choice(['Phase I', 'Phase II', 'Phase III'], n),
    'disease_type': np.random.choice(['Oncology', 'Cardiology', 'Neurology', 'Diabetes'], n),
    'treatment_arm': np.random.choice(['Drug', 'Placebo'], n),
    'site_distance_km': np.random.randint(1, 150, n),

    # Engagement & behavior
    'visits_completed': np.random.randint(1, 12, n),
    'visits_missed': np.random.randint(0, 6, n),
    'adverse_events': np.random.randint(0, 5, n),
    'protocol_deviations': np.random.randint(0, 3, n),
    'days_in_trial': np.random.randint(10, 180, n),

    # Socioeconomic
    'employment_status': np.random.choice(['Employed', 'Unemployed', 'Retired'], n),
    'has_caregiver': np.random.choice([0, 1], n),
    'insurance_coverage': np.random.choice([0, 1], n),
}

df = pd.DataFrame(data)

# Create dropout label based on logical rules
# (higher risk if: many missed visits, far from site, many adverse events)
dropout_score = (
    (df['visits_missed'] / (df['visits_completed'] + 1)) * 40 +
    (df['site_distance_km'] / 150) * 20 +
    (df['adverse_events'] / 5) * 25 +
    (df['protocol_deviations'] / 3) * 15 +
    np.random.uniform(0, 20, n)  # some randomness
)

df['dropout'] = (dropout_score > 35).astype(int)

# Save it
df.to_csv('data/clinical_trial_data.csv', index=False)
print(f"✅ Dataset created with {n} patients")
print(f"Dropout rate: {df['dropout'].mean()*100:.1f}%")
print(df.head())