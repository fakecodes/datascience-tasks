import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the provided URL
url = "https://raw.githubusercontent.com/rebekz/datascience_course/data_2024_1/data/2024_1/isp_churned_2024_1.json"
data = pd.read_json(url)

# Display the column names to verify
print(data.columns)

# Churn rate visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='churn', data=data, palette='viridis')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Visualization of churn rate based on customer type
plt.figure(figsize=(12, 6))
sns.countplot(x='customer_type', hue='churn', data=data, palette='viridis')
plt.title('Churn Rate by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Monthly data volume distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['avg_data_volume'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Average Data Volume')
plt.xlabel('Average Data Volume')
plt.ylabel('Frequency')
plt.show()

# Churn rate based on island
plt.figure(figsize=(12, 6))
sns.countplot(x='island', hue='churn', data=data, palette='viridis')
plt.title('Churn Rate by Island')
plt.xlabel('Island')
plt.ylabel('Count')
plt.show()

# Scatter plot of Average Data Volume vs Average Number of Domains
plt.figure(figsize=(10, 6))
sns.scatterplot(x='avg_data_volume', y='avg_num_domains', hue='churn', data=data, palette='viridis')
plt.title('Average Data Volume vs Average Number of Domains')
plt.xlabel('Average Data Volume')
plt.ylabel('Average Number of Domains')
plt.show()

# Encoding categorical variables to numeric for correlation heatmap
# This will convert categorical variables to numeric representations
data_encoded = data.copy()
data_encoded['customer_type'] = data_encoded['customer_type'].astype('category').cat.codes
data_encoded['island'] = data_encoded['island'].astype('category').cat.codes
data_encoded['churn'] = data_encoded['churn'].astype('category').cat.codes

# Generating the correlation heatmap with only numeric columns
plt.figure(figsize=(12, 8))
numeric_columns = data_encoded.select_dtypes(include=['float64', 'int64']).columns
corr = data_encoded[numeric_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
