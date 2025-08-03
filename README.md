#  Mental Health Trends in the Tech Industry

## Project Overview

This capstone project investigates mental health trends across various regions, age groups, and mental health conditions, specifically within the technology industry. Using a real-world dataset, the project integrates Python analytics, machine learning models, and Power BI dashboards to derive meaningful insights.

---

## Dataset

- **Name:** `dmhasnsduhmh2018.csv`
- **Fields:** Age Range, Region, Mental Health Condition, Year, Value
- **Source:** Provided by course instructors
- **Format:** Tabular CSV

---

## Methodology

### 1. Data Cleaning & Preprocessing
- Removed missing and inconsistent values
- Extracted `Start Year` from range (e.g., "2009–2010")
- Created a new `Risk Level` column based on value thresholds
- Encoded categorical variables for clustering
- Scaled numerical features

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions and trends over time
- Used boxplots, bar charts, and line plots
- Discovered patterns in region, age group, and condition reporting

### 3. Machine Learning Model
- Model: KMeans Clustering
- Optimized number of clusters using Elbow & Silhouette methods
- Applied PCA for dimensionality reduction
- Grouped observations into low, medium, and high-risk clusters

### 4. Power BI Dashboard
- Built multi-page interactive report:
  - Summary View (KPIs)
  - Risk Analysis
  - Region Drill-through
  - DAX Insights Page
- Includes slicers, Smart Narratives, tooltips, and bookmarks

---

## DAX Formulas

Some of the custom DAX measures used in Power BI:
```DAX
Average Value = AVERAGE(MentalHealthData[Value])
Risk Level = SWITCH(TRUE(), [Value] < 5, "Low", [Value] < 15, "Medium", "High")
YoY Change = VAR Prev = CALCULATE(AVERAGE(MentalHealthData[Value]), PREVIOUSYEAR(MentalHealthData[Start Year])) RETURN AVERAGE(MentalHealthData[Value]) - Prev
# mental_health_EDA
