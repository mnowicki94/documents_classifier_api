import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Set display options to show all rows and columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Load the dataset
file_path = "/Users/nom3wz/Documents/repos/documents_classifier/data/newsCorpora.csv"
df = pd.read_csv(
    file_path,
    delimiter="\t",
    header=None,
    names=[
        "ID",
        "Title",
        "URL",
        "Publisher",
        "Category",
        "Story",
        "Hostname",
        "Timestamp",
    ],
)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

print("\nFirst 5 rows of the dataset:")
print(data.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include="all"))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Distribution of categories
plt.figure(figsize=(10, 6))
sns.countplot(x="Category", data=df, order=df["Category"].value_counts().index)
plt.title("Distribution of News Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Distribution of publishers
top_publishers = df["Publisher"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_publishers.index, y=top_publishers.values)
plt.title("Top 10 Publishers")
plt.xlabel("Publisher")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Distribution of articles over time
df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
df["Year"] = df["Timestamp"].dt.year
plt.figure(figsize=(10, 6))
sns.countplot(x="Year", data=df, order=df["Year"].value_counts().index)
plt.title("Distribution of Articles Over Time")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()
