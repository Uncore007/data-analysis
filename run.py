import pandas as pd

# Import the dataset
df = pd.read_csv("student_habits_performance.csv")

# initial inspection, to ensure the data is loaded correctly
print(df.head())
print(df.info())

# Check for missing values
df_clean = df.dropna(subset=[
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation",
    "exam_score"
]).reset_index(drop=True)

# This map is used to encode Yes/No columns
yes_no_map = {"Yes": 1, "No": 0}

# Encode `part_time_job` and `extracurricular_participation`
for col in ["part_time_job", "extracurricular_participation"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].map(yes_no_map)

# Encode gender (assuming “male”/“female”)
if "gender" in df_clean.columns:
    gender_map = {"male": 1, "female": 0}
    df_clean["gender"] = df_clean["gender"].map(gender_map)

# Encode diet quality
if "diet_quality" in df_clean.columns:
    diet_map = {"Poor": 0, "Fair": 1, "Good": 2}
    df_clean["diet_quality"] = df_clean["diet_quality"].map(diet_map)

# Encode parental education level
if "parental_education_level" in df_clean.columns:
    parent_ed_map = {
        "None": 0,
        "High School": 1,
        "Bachelor": 2,
        "Master": 3,
    }
    df_clean["parental_education_level"] = df_clean["parental_education_level"].map(parent_ed_map)

# Encode internet quality
if "internet_quality" in df_clean.columns:
    internet_map = {
        "Poor": 0,
        "Average": 1,
        "Good": 2,
    }
    df_clean["internet_quality"] = df_clean["internet_quality"].map(internet_map)

# Print the cleaned DataFrame info to check data types and ensure encoding worked
print(df_clean.dtypes)

habit_columns = [
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation"
]

# Calculate Pearson correlations with exam_score
corrs = {}
for col in habit_columns:
    if col in df_clean.columns:
        corrs[col] = df_clean[col].corr(df_clean["exam_score"])

# Sort correlations by absolute value
corr_series = pd.Series(corrs)
corr_series_sorted = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)

# Display the sorted correlations
print("Pearson correlations with exam_score (sorted by absolute value):")
print(corr_series_sorted)

import matplotlib.pyplot as plt

# Distribution of Exam Scores
plt.figure(figsize=(8, 5))
df_clean["exam_score"].plot(kind='hist', bins=20, title='Distribution of Exam Scores')
plt.xlabel("Exam Score")
plt.ylabel("Frequency")

# Show the plot
plt.show()

# Distribution of Study Hours
plt.figure(figsize=(8, 5))
df_clean["study_hours_per_day"].plot(kind='hist', bins=15, title='Distribution of Study Hours per Day')
plt.xlabel("Study Hours per Day")
plt.ylabel("Frequency")

# Show the plot
plt.show()

# Bar plot of Pearson correlations with exam_score
plt.figure(figsize=(10, 6))
bars = plt.bar(
    corr_series_sorted.index,
    corr_series_sorted.values,
    width=0.6
)
plt.axhline(0, color="black", linewidth=0.8)  # zero line
plt.xticks(rotation=45, ha="right")
plt.ylabel("Pearson Correlation with Exam Score")
plt.title("Impact of Lifestyle Factors on Exam Score")

# Annotate each bar with its correlation coefficient (rounded to 2 decimals)
for bar in bars:
    height = bar.get_height()
    label_y = height + (0.01 if height >= 0 else -0.05)
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        label_y,
        f"{height:.2f}",
        ha="center",
        va="bottom" if height >= 0 else "top",
        fontsize=9
    )

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()