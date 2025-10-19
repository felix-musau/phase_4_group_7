### 🚗 Chicago Traffic Crash Cause Prediction
Introduction

This project analyzes the factors influencing vehicle crashes in Chicago using machine learning. The goal is to build a predictive model that identifies the primary contributory cause of a car accident based on vehicle, environmental, and roadway conditions. The insights from this project aim to help the City of Chicago and road safety agencies make data-driven decisions to reduce crash frequency and severity.

### Business Problem

Traffic accidents remain a major public safety concern in Chicago, resulting in injuries, fatalities, and significant economic loss. Understanding the underlying causes of these accidents is essential for improving road safety, urban planning, and public awareness.

This project focuses on predicting the primary contributory cause of traffic crashes — such as driver error, weather, or road design — using historical data. By identifying patterns and high-risk conditions, city authorities can design targeted interventions, allocate enforcement resources effectively, and prevent future crashes.

### Methodology
### Data Source

The dataset was obtained from the City of Chicago Data Portal, containing nearly 1 million crash records with detailed information about:

Road and weather conditions

Lighting and traffic control devices

Crash types, severity, and timing

Vehicle and injury details

### Data Cleaning

Removed irrelevant columns such as LOCATION, LATITUDE, LONGITUDE, and DATE_POLICE_NOTIFIED.

Standardized column names and string formats (trimmed whitespace, converted to uppercase).

Handled missing values: categorical features filled with "Unknown", numerical with the median.

Encoded categorical variables using OneHotEncoder and scaled numeric features using StandardScaler.

Removed duplicates and grouped rare causes under “OTHER” to reduce cardinality.

### Data Splitting

The data was split into:

Training set (70%)

Validation set (15%)

Test set (15%)
using stratified sampling to preserve class balance.

### Modeling

Multiple models were trained and compared, including:

Decision Tree Classifier (baseline model)

Random Forest Classifier

Gradient Boosting & XGBoost

Neural Network (Keras Sequential Model) for improved performance

Each model was evaluated using Accuracy, Precision, Recall, and F1-Score. The Neural Network used dropout and regularization to reduce overfitting.

### Key Findings & Recommendations
🔍 Insights

Most crashes occurred during clear weather and daylight, indicating that driver behavior plays a larger role than environmental factors.

The most common causes included:

Failing to yield the right-of-way

Following too closely

Failing to reduce speed

Fridays and evening hours (3–6 PM) showed the highest crash frequencies.

Major intersections and high-speed zones had disproportionately high crash counts.

### 🚦 Recommendations

Enforcement & Awareness: Strengthen campaigns against speeding and right-of-way violations.

Infrastructure: Improve traffic signage, lane markings, and intersection lighting.

Predictive Safety Monitoring: Deploy models citywide to flag high-risk intersections in real time.

Targeted Patrols: Use model outputs to allocate traffic police and cameras to high-risk zones.

### Technologies Used

Python – Core programming language

Pandas / NumPy – Data cleaning and transformation

Scikit-learn – Model training, preprocessing, and evaluation

Imbalanced-learn (SMOTE) – Oversampling for imbalanced classes

TensorFlow / Keras – Deep learning (Neural Network model)

Matplotlib / Seaborn / Plotly – Data visualization and analytics

Jupyter Notebook – Development and experimentation environment

### Future Improvements

Implement hyperparameter tuning using KerasTuner for automated optimization.

Integrate geospatial analysis to visualize accident hotspots on city maps.

Experiment with deep learning architectures (LSTM, CNN) to model time and spatial dependencies.

Deploy as an interactive web dashboard for real-time crash risk monitoring and visualization.

### Conclusion

By leveraging machine learning and deep learning, this project demonstrates how predictive analytics can uncover the major factors contributing to traffic accidents in Chicago. These insights can empower city planners, law enforcement, and transportation agencies to design data-driven policies, improve public safety, and ultimately reduce crash-related injuries and fatalities.

Through continued model tuning and integration with live data, this approach can form the foundation for an AI-powered traffic safety monitoring system for urban environments.
