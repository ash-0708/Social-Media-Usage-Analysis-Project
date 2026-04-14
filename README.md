# 📊 Social Media Usage Analysis Dashboard

## 📌 Project Overview
This project analyzes social media usage patterns across multiple platforms to uncover insights about user engagement, activity levels, and growth trends.
The analysis focuses on understanding how user behavior varies across platforms like Instagram, Twitter, Facebook, TikTok, LinkedIn, Pinterest, and Snapchat.

---

## 🎯 Objectives
- Analyze user engagement metrics
- Compare platform performance
- Identify patterns in user activity
- Build an interactive dashboard for visualization

---

## 📂 Dataset Information

The dataset contains 1000 users with the following features:

- User_ID
- App (Platform name)
- Daily_Minutes_Spent
- Posts_Per_Day
- Likes_Per_Day
- Follows_Per_Day

### ✅ Data Quality
- No missing values
- Clean and structured dataset
- Outliers handled using IQR method

---

## 🧹 Data Cleaning
- Verified data types
- Checked for null values
- Handled outliers using Tukey IQR method

---

## 📊 Exploratory Data Analysis

### 🔹 Key Findings:
- Instagram has the highest average daily usage (~264 minutes)
- User engagement varies significantly across platforms

---

## 🔗 Correlation Insights

| Metric | Correlation |
|------|--------|
| Time Spent vs Likes | 0.014 |
| Posts vs Follows | 0.018 |

👉 Indicates weak relationships between activity and engagement.

---

## 📈 Dashboard Features

The interactive dashboard includes:

- Average time spent by platform
- Likes and follows comparison
- Distribution of posts per day
- Scatter plot (Time vs Likes)

---

## 🛠 Tools & Technologies

- Python (Pandas, NumPy)
- Plotly (Interactive Visualizations)
- Jupyter Notebook

---

## 📁 Project Structure
├── social_media_usage_cleaned.csv
├── analysis_summary.txt
├── Social_Media_Usage.pbix
├── README.md

---

## 🚀 How to Run

1. Clone the repository.
2. Open the dashboard:

---
## 💡 Key Insights

- High activity does not guarantee higher engagement
- Instagram leads in user attention
- User growth is not strongly dependent on posting frequency

---

## 📌 Future Improvements

- Add time-series analysis
- Include user segmentation
- Deploy dashboard using Power BI

---

## 👩‍💻 Author
Ashwati Jain
