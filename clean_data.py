import pandas as pd

# Input and output file names
input_file = "DataScience_salaries_2024.csv"
output_file = "cleaned_DataScience_salaries_2024.csv"

# Read the CSV file
df = pd.read_csv(input_file)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing spaces from text data
text_columns = df.select_dtypes(include=["object"]).columns

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()

# Replace empty strings with missing values
df.replace("", pd.NA, inplace=True)

# Remove rows containing missing values
df = df.dropna()

# Convert numeric columns to numeric values
numeric_columns = [
    "work_year",
    "salary",
    "salary_in_usd",
    "remote_ratio"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows made invalid during numeric conversion
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("Cleaning completed successfully!")
print("Cleaned dataset shape:", df.shape)
print("Cleaned file created:", output_file)