Social Media Addiction & Mental Health Analysis
================================================
Uncovering how social media habits affect student mental health through data-driven EDA and predictive modeling.


INTRODUCTION
------------
This project explores the relationship between social media usage patterns and mental health outcomes among students. Using a dataset of 705 student records, it investigates how daily usage hours, platform preferences, relationship status, and academic level correlate with mental health scores. The pipeline covers the full data science workflow from cleaning and encoding to model training and visualization, culminating in a Random Forest Regressor that predicts mental health scores and a rich set of EDA plots that surface the most impactful behavioral signals.


TECHNOLOGIES
------------
- Python          — Core language
- Pandas          — Data manipulation and cleaning
- NumPy           — Numerical operations
- Matplotlib      — Base plotting
- Seaborn         — Statistical visualizations
- Scikit-learn    — Model training, evaluation, and data splitting


FEATURES
--------
- Descriptive Statistics: Mean, median, mode, variance, standard deviation, range, skew, and kurtosis for all numerical columns
- Label Encoding: Ordinal mapping for Gender, Academic_Level, and Affects_Academic_Performance
- One-Hot Encoding: Encoding for high-cardinality categorical columns (Most_Used_Platform, Relationship_Status)
- Random Forest Regressor: Predicts Mental_Health_Score with MAE, MSE, and R-squared evaluation
- Feature Importance Plot: Top 20 most predictive features ranked by importance score
- Regression Plot: Linear trend between Avg_Daily_Usage_Hours and mental health score
- Category Bar Plots: Average mental health score broken down by platform and academic level
- Correlation Heatmaps: Lower-triangle heatmaps for top 10 positively and negatively correlated features


THE PROCESS
-----------
1. Data Loading
   Read the CSV file into a Pandas DataFrame and inspected shape, head, tail, and column types to understand the structure of the data.

2. Exploratory Statistics
   Computed first through fourth moment statistics including mean, variance, skew, and kurtosis for all continuous features to understand distributions before any modeling.

3. Preprocessing
   - Dropped irrelevant identifier columns (Student_ID, Country)
   - Ordinal-encoded binary and ordered categorical columns (Gender, Academic_Level, Affects_Academic_Performance)
   - One-hot encoded nominal categorical columns (Most_Used_Platform, Relationship_Status) using pandas get_dummies

4. Model Training
   Split the data 80/20 into training and test sets. Trained a Random Forest Regressor with 100 estimators on the training set and generated predictions on the test set.

5. Model Evaluation
   Evaluated predictions using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. Plotted predicted vs actual values with a reference diagonal to visually assess model fit.

6. Feature Importance
   Extracted and plotted the top 20 feature importances from the trained Random Forest to identify which behavioral and demographic signals most strongly predict mental health scores.

7. EDA Visualizations
   - Regression plot of daily usage hours vs mental health score
   - Correlation heatmaps highlighting the strongest positive and negative relationships with the target variable
  
  ### **GRAPHS** - 
  **Predicted vs Actual** — Scatter plot comparing model predictions against true Mental Health Scores, with a red y=x reference line and R² in the title    
  **Top Feature Importances** — Horizontal bar chart of the top 20 features ranked by Random Forest importance score    
  **Avg Daily Usage Hours vs Mental Health Score** — Regression plot with a red trend line showing the linear relationship between daily usage and mental health    
  **Average Mental Health Score by Academic Level** — Bar plot grouped by Academic_Level (Undergraduate, Graduate, High School)    
  **Average Mental Health Score by Most Used Platform** — Bar plot grouped by one-hot encoded platform categories (Instagram, TikTok, etc.)     
  **Positive Correlation Heatmap** — Lower-triangle heatmap of the top 10 features most positively correlated with Mental Health Score    
  **Negative Correlation Heatmap** — Lower-triangle heatmap of the top 10 features most negatively correlated with Mental Health Score    

  <img width="800" height="600" alt="bargraph" src="https://github.com/user-attachments/assets/62609848-6681-46ce-8673-f2ccf5e4297a" />
  <img width="1000" height="600" alt="daily_vs_score" src="https://github.com/user-attachments/assets/ca0fa7b2-8083-4673-b7b5-9c99107c0d02" />
  <img width="1000" height="800" alt="heatmap" src="https://github.com/user-attachments/assets/44595147-b203-4775-823c-f941039e54bc" />
  <img width="1000" height="800" alt="heatmap2" src="https://github.com/user-attachments/assets/88e27c3f-ddc5-47f0-8e84-f6a7974d3939" />
  <img width="640" height="480" alt="model" src="https://github.com/user-attachments/assets/477e222e-e4ef-46bd-a695-4234bb3e4b0a" />
  <img width="1366" height="655" alt="platform_vs_score" src="https://github.com/user-attachments/assets/779caa73-2c8d-43c8-a78a-16efac4878c8" />
  <img width="800" height="600" alt="score_vs_acad" src="https://github.com/user-attachments/assets/1c533926-f2dd-4e56-99b2-5f8a135f7bb6" />




  
