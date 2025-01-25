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
data = pd.read_csv(
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
print(data.describe(include="all"))

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Distribution of categories
plt.figure(figsize=(10, 6))
sns.countplot(x="Category", data=data, order=data["Category"].value_counts().index)
plt.title("Distribution of News Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# Distribution of publishers
top_publishers = data["Publisher"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_publishers.index, y=top_publishers.values)
plt.title("Top 10 Publishers")
plt.xlabel("Publisher")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Distribution of articles over time
data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit="s")
data["Year"] = data["Timestamp"].dt.year
plt.figure(figsize=(10, 6))
sns.countplot(x="Year", data=data, order=data["Year"].value_counts().index)
plt.title("Distribution of Articles Over Time")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()
