# --- Data manipulation imports ---
import pandas as pd                          # For data manipulation and analysis
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For plotting and visualizations
import seaborn as sns                        # For advanced statistical visualizations

# --- Scikit-learn Imports ---
from sklearn.model_selection import train_test_split    # To split data into training and testing sets
from sklearn.ensemble import RandomForestRegressor      # Random Forest model for regression tasks
from sklearn.tree import export_text, plot_tree         # For visualizing and exporting decision trees

from sklearn.metrics import (
    accuracy_score,                # Metric for classification accuracy
    classification_report,         # Generates a report with precision, recall, F1-score
    confusion_matrix,              # Computes confusion matrix for classification tasks
    mean_absolute_error,           # MAE for regression evaluation
    mean_squared_error,             # MSE/RMSE for regression evaluation
    r2_score                       # R^2 for regression evaluation
)

# Define the path to the CSV file containing the dataset
data_dir = "C://Users//Shabnam//Desktop//Social Media Addiction//social_media_addiction.csv"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(data_dir)

# Display the first few rows of the DataFrame to get an overview of the data
print("head of Data\n",df.head(4))
# Size of the Data
print("shape of data\n",df.shape)
print("tail of data\n", df.tail(4))
print("info of data\n",df.info)
print("description of data\n",df.describe)

#First Moment Business Descision
print("Mean")
print("Mean for Age", df.Age.mean())
print("Mean for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.mean())
print("Mean for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.mean())
print("Mean for Mental_Health_Score", df.Mental_Health_Score.mean())
print("Mean for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.mean())
print("Mean for Addicted_Score", df.Addicted_Score.mean())

print("Median")
print("Median for Age", df.Age.median())
print("Median for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.median())
print("Median for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.median())
print("Median for Mental_Health_Score", df.Mental_Health_Score.median())
print("Median for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.median())
print("Median for Addicted_Score", df.Addicted_Score.median())

print("Mode")
print("Mode for Age", df.Age.mode())
print("Mode for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.mode())
print("Mode for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.mode())
print("Mode for Mental_Health_Score", df.Mental_Health_Score.mode())
print("Mode for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.mode())
print("Mode for Addicted_Score", df.Addicted_Score.mode())

print("Variance")
print("Variance for Age", df.Age.var())
print("Variance for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.var())
print("Variance for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.var())
print("Variance for Mental_Health_Score", df.Mental_Health_Score.var())
print("Variance for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.var())
print("Variance for Addicted_Score", df.Addicted_Score.var())

print("Std Deviation")
print("Std Deviation for Age", df.Age.std())
print("Std Deviation for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.std())
print("Std Deviation for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.std())
print("Std Deviation for Mental_Health_Score", df.Mental_Health_Score.std())
print("Std Deviation for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.std())
print("Std Deviation for Addicted_Score", df.Addicted_Score.std())

print("Range")
print("Range for Age", max(df.Age)-min(df.Age))
print("Range for Avg_Daily_Usage_Hours",max(df.Avg_Daily_Usage_Hours)-min(df.Avg_Daily_Usage_Hours))
print("Range for Sleep_Hours_Per_Night", max(df.Sleep_Hours_Per_Night)-min(df.Sleep_Hours_Per_Night))
print("Range for Mental_Health_Score", max(df.Mental_Health_Score)-min(df.Mental_Health_Score))
print("Range for Conflicts_Over_Social_Media", max(df.Conflicts_Over_Social_Media)-min(df.Conflicts_Over_Social_Media))
print("Range for Addicted_Score",max(df.Addicted_Score)-min(df.Addicted_Score))

print("Skew")
print("Skew for Age", df.Age.std())
print("Skew for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.std())
print("Skew for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.std())
print("Skew for Mental_Health_Score", df.Mental_Health_Score.std())
print("Skew for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.std())
print("Skew for Addicted_Score", df.Addicted_Score.std())

print("Kurtosis")
print("Kurtosis for Age", df.Age.std())
print("Kurtosis for Avg_Daily_Usage_Hours", df.Avg_Daily_Usage_Hours.std())
print("Kurtosis for Sleep_Hours_Per_Night", df.Sleep_Hours_Per_Night.std())
print("Kurtosis for Mental_Health_Score", df.Mental_Health_Score.std())
print("Kurtosis for Conflicts_Over_Social_Media", df.Conflicts_Over_Social_Media.std())
print("Kurtosis for Addicted_Score", df.Addicted_Score.std())

#dropping coloumns 
columns_drop = ['Student_ID', 'Country']
df.drop (columns_drop, inplace=True, axis=1)
df.head()

df['Gender'].unique
mapping = {'Male':0,'Female':1}
unmapped_values = df.loc[~df['Gender'].isin(mapping.keys()), 'Gender'].unique()

if len(unmapped_values) > 0:
    raise ValueError(f"Error: The following values in 'Gender' are not in the mapping: {unmapped_values}")
else:
    # Safe to map since all values are accounted for
    df['Gender'] = df['Gender'].map(mapping)

df['Academic_Level'].unique
mapping = {'Undergraduate':0,'Graduate':1,'High School': 2}
unmapped_values = df.loc[~df['Academic_Level'].isin(mapping.keys()), 'Academic_Level'].unique()

if len(unmapped_values) > 0:
    raise ValueError(f"Error: The following values in 'Academic_Level' are not in the mapping: {unmapped_values}")
else:
    # Safe to map since all values are accounted for
    df['Academic_Level'] = df['Academic_Level'].map(mapping)


df['Affects_Academic_Performance'].unique
mapping = {'Yes':0,'No':1}
unmapped_values = df.loc[~df['Affects_Academic_Performance'].isin(mapping.keys()), 'Affects_Academic_Performance'].unique()

if len(unmapped_values) > 0:
    raise ValueError(f"Error: The following values in 'Affects_Academic_Performance' are not in the mapping: {unmapped_values}")
else:
    # Safe to map since all values are accounted for
    df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map(mapping)

# Display the first few rows of the updated DataFrame to verify changes
print(df.head())

print(df['Most_Used_Platform'].unique)
# TODO: Specify columns to one-hot encode
columns_to_encode = ['Most_Used_Platform']
df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
print(df.head())

print(df['Relationship_Status'].unique)
# TODO: Specify columns to one-hot encode
columns_to_encode2 = ['Relationship_Status']
df = pd.get_dummies(df, columns=columns_to_encode2, drop_first=True)
print(df.head())

X = df.drop('Mental_Health_Score', axis=1)

y = df['Mental_Health_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing feature sets and labels
print("X_train shape:", X_train.shape)  # Shape of training features
print("X_test shape:", X_test.shape)    # Shape of testing features
print("y_train shape:", y_train.shape)  # Shape of training labels
print("y_test shape:", y_test.shape)    # Shape of testing labels

# Initialize a Random Forest Regressor with 100 decision trees
# random_state ensures reproducibility of results
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train (fit) the model using the training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred

# Calculate the Mean Absolute Error (MAE) between the true and predicted values
mae = mean_absolute_error(y_test, y_pred)

# Print the MAE to evaluate the average magnitude of errors in predictions
print(f"MAE: {mae}")

mse = mean_squared_error(y_test, y_pred)

# Print the MSE value
print(f"MSE: {mse}")

r2 = r2_score(y_test, y_pred)

# Plot the Predicted vs. Actual values
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # y = x line
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title(f"Predicted vs Actual (R² = {r2:.2f})")
plt.grid(True)
plt.show()

importances = model.feature_importances_
feature_names = X_train.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

top_n = 20
feature_importances.head(top_n).plot(kind='barh', figsize=(8,6))
plt.gca().invert_yaxis()
plt.title("Top Feature Importances")
plt.xlabel("Importance Score")
plt.show()

feature = 'Avg_Daily_Usage_Hours'
sns.set(style="whitegrid", context="talk")

# Create the plot
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df,
    x=feature,
    y='Mental_Health_Score',
    scatter_kws={"s": 60, "alpha": 0.6},
    line_kws={"color": "red", "lw": 2}
)

# Customize titles and labels
plt.title(f"{feature.replace('_', ' ')} vs. Mental Health Score", fontsize=16)
plt.xlabel(feature.replace('_', ' '), fontsize=14)
plt.ylabel("Mental Health Score", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
"""
plt.figure(figsize=(20, 10))
sns.barplot(data=avg_scores_df, x='Category', y='Average_Mental_Health_Score', hue='Category', palette='flare', legend=False)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Mental Health Score')
plt.title('Average Mental Health Score by ' + feature)
plt.tight_layout()
plt.show()
"""
# TODO: Replace with your feature name
feature = 'Academic_Level'

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x=feature, y='Mental_Health_Score')
plt.title(f'Average Mental Health Score by {feature}')
plt.xlabel(feature)
plt.ylabel('Average Mental Health Score')
plt.show()

# TODO: Replace with feature with multiple categories
feature = "Most_Used_Platform"

binary_columns = [col for col in df.columns if col.startswith(feature + '_')]

avg_scores = {}
for col in binary_columns:
    label = col.replace(feature + '_', '')
    avg_scores[label] = df[df[col] == 1]['Mental_Health_Score'].mean()

avg_scores_df = pd.DataFrame(list(avg_scores.items()), columns=['Category', 'Average_Mental_Health_Score'])
avg_scores_df = avg_scores_df.sort_values(by='Average_Mental_Health_Score', ascending=False)

plt.figure(figsize=(20, 10))
sns.barplot(data=avg_scores_df, x='Category', y='Average_Mental_Health_Score', hue='Category', palette='flare', legend=False)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Mental Health Score')
plt.title('Average Mental Health Score by ' + feature)
plt.tight_layout()
plt.show()

# Compute correlation matrix
correlations = df.corr()

# Select top 10 features most positively correlated with 'Mental_Health_Score' (excluding itself)
top_corr_features = correlations['Mental_Health_Score'].drop('Mental_Health_Score').sort_values(ascending=False).head(10).index

# Subset the DataFrame to these features + target
top_corr_matrix = df[top_corr_features.union(['Mental_Health_Score'])].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))

# Plot the heatmap with the mask applied to show only the lower triangle
plt.figure(figsize=(10, 8))
sns.heatmap(top_corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, annot_kws={"size":10}, fmt=".2f")
plt.title("Lower Triangle Heatmap: Top 10 Features Positively Correlated with Mental_Health_Score")
plt.show()

# Compute correlation matrix
correlations = df.corr()

# Select top 10 features most negatively correlated with 'Mental_Health_Score' (lowest correlation values)
top_neg_corr_features = correlations['Mental_Health_Score'].drop('Mental_Health_Score').sort_values(ascending=True).head(10).index

# Subset the DataFrame to these features + target
top_neg_corr_matrix = df[top_neg_corr_features.union(['Mental_Health_Score'])].corr()

# Create a mask for the upper triangle to show only the lower triangle
mask = np.triu(np.ones_like(top_neg_corr_matrix, dtype=bool))

# Plot the heatmap with the mask applied
plt.figure(figsize=(10, 8))
sns.heatmap(top_neg_corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, annot_kws={"size":10}, fmt=".2f")
plt.title("Lower Triangle Heatmap: Top 10 Features Negatively Correlated with Mental_Health_Score")
plt.show()
