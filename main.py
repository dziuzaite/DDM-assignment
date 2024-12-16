import pandas as pd
import pyddm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

# Load the dataset
df = pd.read_csv("dataset-14 - Copy.tsv", sep='\t')

# Calculate accuracy (1 for correct response, 0 for incorrect)
df['accuracy'] = np.where(df['R'] == df['S'], 1, 0)

# 1) Histogram of Frequency of Reaction Times in Raw Data
plt.figure(figsize=(12, 6))
plt.hist(df['rt'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Reaction Times (RT) in Raw Data')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Frequency')
plt.show()

# 2) Histogram of Frequency of Reaction Times in Data Without Outliers
# Remove outliers using IQR (Interquartile Range)
Q1 = df['rt'].quantile(0.25)
Q3 = df['rt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers = df[(df['rt'] >= lower_bound) & (df['rt'] <= upper_bound)]

plt.figure(figsize=(12, 6))
plt.hist(data_no_outliers['rt'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
plt.title('Histogram of Reaction Times (RT) Without Outliers')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Frequency')
plt.show()

# 3) Histogram and Density Plot of Reaction Times for Correct (1) and Incorrect (0) Responses
plt.figure(figsize=(12, 6))
for accuracy_val in [0, 1]:
    subset = df[df['accuracy'] == accuracy_val]
    plt.hist(subset['rt'], bins=30, alpha=0.5, label=f'Accuracy {accuracy_val}', density=True)

plt.title('Histogram of Reaction Times for Correct (1) and Incorrect (0) Responses')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Density')
plt.legend(title='Accuracy')
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(df[df['accuracy'] == 0]['rt'], label='Incorrect Responses', color='red', shade=True)
sns.kdeplot(df[df['accuracy'] == 1]['rt'], label='Correct Responses', color='green', shade=True)
plt.title('Density Plot of Reaction Times for Correct and Incorrect Responses')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Density')
plt.legend(title='Accuracy')
plt.show()

# 4) Histogram and Density Plot of Reaction Times for Conditions
plt.figure(figsize=(12, 6))
for condition in ['speed', 'accuracy']:
    subset = df[df['instruction'] == condition]
    plt.hist(subset['rt'], bins=30, alpha=0.5, label=f'Condition: {condition}', density=True)

plt.title('Histogram of Reaction Times for Speed vs Accuracy Conditions')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Density')
plt.legend(title='Condition')
plt.show()

plt.figure(figsize=(12, 6))
sns.kdeplot(df[df['instruction'] == 'speed']['rt'], label='Speed Condition', color='blue', shade=True)
sns.kdeplot(df[df['instruction'] == 'accuracy']['rt'], label='Accuracy Condition', color='purple', shade=True)
plt.title('Density Plot of Reaction Times for Speed vs Accuracy Conditions')
plt.xlabel('Reaction Time (s)')
plt.ylabel('Density')
plt.legend(title='Condition')
plt.show()

# 5) Visualize aggregated (median) RTs per subject and condition
# Calculate median reaction times
median_rt = df.groupby(['subjects', 'instruction']).agg(median_rt=('rt', 'median')).reset_index()

plt.figure(figsize=(12, 6))
for condition in median_rt['instruction'].unique():
    subset = median_rt[median_rt['instruction'] == condition]
    plt.bar(
        subset['subjects'] + (0.2 if condition == 'speed' else -0.2),
        subset['median_rt'],
        width=0.4,
        label=condition
    )

plt.title('Aggregated (Median) Reaction Times per Subject and Condition')
plt.xlabel('Subject')
plt.ylabel('Median RT (s)')
plt.xticks(ticks=median_rt['subjects'].unique(), labels=median_rt['subjects'].unique())
plt.legend(title='Condition')
plt.show()

# 6) Paired t-tests for Reaction Times and Accuracy

# Pair reaction times per subject for each condition
rt_speed = df[df['instruction'] == 'speed'].groupby('subjects')['rt'].mean()
rt_accuracy = df[df['instruction'] == 'accuracy'].groupby('subjects')['rt'].mean()

# Drop subjects without data for both conditions (just in case)
paired_data = pd.DataFrame({'speed': rt_speed, 'accuracy': rt_accuracy}).dropna()

# Perform the paired t-test for reaction times
t_stat_rt, p_value_rt = ttest_rel(paired_data['speed'], paired_data['accuracy'])
print(f"Paired t-test for reaction times: t = {t_stat_rt:.2f}, p = {p_value_rt:.3e}")

# Visualize mean RTs per condition
plt.figure(figsize=(8, 6))
mean_rts = paired_data.mean()
plt.bar(mean_rts.index, mean_rts.values, color=['blue', 'purple'], alpha=0.7)
plt.title('Mean Reaction Times per Condition', pad=20)
plt.ylabel('Mean Reaction Time (s)')
plt.xlabel('Condition')
plt.xticks(ticks=[0, 1], labels=['Speed', 'Accuracy'])
plt.show()

# Pair accuracy per subject for each condition
accuracy_speed = df[df['instruction'] == 'speed'].groupby('subjects')['accuracy'].mean()
accuracy_accuracy = df[df['instruction'] == 'accuracy'].groupby('subjects')['accuracy'].mean()

# Drop subjects without data for both conditions (just in case)
paired_accuracy = pd.DataFrame({'speed': accuracy_speed, 'accuracy': accuracy_accuracy}).dropna()

# Perform the paired t-test for accuracy
t_stat_accuracy, p_value_accuracy = ttest_rel(paired_accuracy['speed'], paired_accuracy['accuracy'])
print(f"Paired t-test for accuracy: t = {t_stat_accuracy:.2f}, p = {p_value_accuracy:.3e}")

# Visualize mean accuracy per condition
plt.figure(figsize=(8, 6))
mean_accuracy = paired_accuracy.mean()
plt.bar(mean_accuracy.index, mean_accuracy.values, color=['blue', 'purple'], alpha=0.7)
plt.title('Mean Accuracy per Condition', pad=20)
plt.ylabel('Mean Accuracy')
plt.xlabel('Condition')
plt.show()

#DDM

T_dur = 4.0

# FAST Condition
m_fast = pyddm.Model(
    drift=pyddm.DriftConstant(drift=1.0),  # Hypothetical drift rate
    noise=pyddm.NoiseConstant(noise=0.8),
    bound=pyddm.BoundConstant(B=0.8),  # Smaller bounds for speed
    overlay=pyddm.OverlayNonDecision(nondectime=0.2),  # Hypothetical non-decision time
    T_dur=T_dur
)
sol_fast = m_fast.solve()
sample_fast = sol_fast.resample(10000)

# Plot correct and error RT distributions for FAST
correct_rts_fast = sample_fast.corr
error_rts_fast = sample_fast.err

plt.figure(figsize=(10, 6))
ax1 = plt.subplot(2, 1, 1)
plt.hist(correct_rts_fast, bins=np.arange(0, T_dur, 20 * .005))
plt.title("FAST - Correct RT distribution")
plt.subplot(2, 1, 2, sharey=ax1)
plt.hist(error_rts_fast, bins=np.arange(0, T_dur, 20 * .005))
plt.title("FAST - Error RT distribution")
plt.tight_layout()
plt.show()

# b) Infinite Simulations and PDFs
# Infinite trials for FAST
correct_pdf_fast = sol_fast.pdf("correct")
error_pdf_fast = sol_fast.pdf("error")

# Plot
plt.figure(figsize=(10, 6))
ax1 = plt.subplot(2, 1, 1)
plt.plot(m_fast.t_domain(), correct_pdf_fast)
plt.title("FAST - Correct RT density")
plt.subplot(2, 1, 2, sharey=ax1)
plt.plot(m_fast.t_domain(), error_pdf_fast)
plt.title("FAST - Error RT density")
plt.tight_layout()
plt.show()

# c) Fitting DDM to Your Dataset
# Mapping both 'S' and 'R' columns to 1 and 0 for compatibility with PyDDM
df['S'] = df['S'].map({'left': 1, 'right': 0})  # Stimulus column mapping
df['R'] = df['R'].map({'left': 1, 'right': 0})  # Response column mapping

# Calculate the maximum response time in the dataset
max_rt = df['rt'].max()

# Set T_dur to be at least as long as the max response time, with an optional buffer (e.g., 10%)
T_dur = max_rt * 1.1  # Add a small buffer to max RT to ensure sufficient simulation time

# Split by condition
df_fast = df[df['instruction'] == "speed"]
df_acc = df[df['instruction'] == "accuracy"]

# Convert to PyDDM Sample
sample_fast = pyddm.Sample.from_pandas_dataframe(df_fast, rt_column_name="rt", choice_column_name="R")
sample_acc = pyddm.Sample.from_pandas_dataframe(df_acc, rt_column_name="rt", choice_column_name="R")

# Create model for the 'FAST' condition
m_fit_fast = pyddm.Model(
    drift=pyddm.DriftConstant(drift=pyddm.Fittable(minval=-5, maxval=5)),
    noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=0.1, maxval=2)),
    bound=pyddm.BoundConstant(B=pyddm.Fittable(minval=0.3, maxval=1.5)),
    overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=0.5)),
    T_dur=T_dur
)

# Fit the 'FAST' model
pyddm.fit_adjust_model(model=m_fit_fast, sample=sample_fast, lossfunction=pyddm.LossRobustLikelihood, verbose=True)

# Create model for the 'ACCURATE' condition
m_fit_acc = pyddm.Model(
    drift=pyddm.DriftConstant(drift=pyddm.Fittable(minval=-5, maxval=5)),
    noise=pyddm.NoiseConstant(noise=pyddm.Fittable(minval=0.1, maxval=2)),
    bound=pyddm.BoundConstant(B=pyddm.Fittable(minval=0.3, maxval=1.5)),
    overlay=pyddm.OverlayNonDecision(nondectime=pyddm.Fittable(minval=0, maxval=0.5)),
    T_dur=T_dur  # Set the T_dur to the calculated value
)

# Fit the 'ACCURATE' model
pyddm.fit_adjust_model(model=m_fit_acc, sample=sample_acc, lossfunction=pyddm.LossRobustLikelihood, verbose=True)

# Display the 'FAST' condition model
print("model for the 'FAST' condition:")
pyddm.display_model(m_fit_fast)

# Display the 'ACCURATE' condition model
print("model for the 'ACCURATE' condition:")
pyddm.display_model(m_fit_acc)

# Example values from your model output (from the printed results)
params_fast = {
    'drift': 0.014136,
    'noise': 1.520159,
    'B': 1.259357,
    'nondectime': 0.121770
}

params_accurate = {
    'drift': 0.002556,
    'noise': 1.559992,
    'B': 1.455933,
    'nondectime': 0.200614
}

# Prepare data for plotting
parameters = ['drift', 'noise', 'B', 'nondectime']
fast_values = [params_fast[param] for param in parameters]
accurate_values = [params_accurate[param] for param in parameters]

# Create a figure and axis for the plots
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Plot the parameters for the FAST and ACCURATE conditions
ax.scatter(parameters, fast_values, label='FAST', color='blue', marker='o', s=100)
ax.scatter(parameters, accurate_values, label='ACCURATE', color='red', marker='x', s=100)

# Add labels, title, and legend
ax.set_xlabel('Parameters')
ax.set_ylabel('Fitted Value')
ax.set_title('DDM Parameters: FAST vs ACCURATE Condition')
ax.legend()

# Show the plot
plt.show()
