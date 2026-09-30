# Smart City Traffic Optimization & Advanced Analytics Capstone

## Project Overview
This project implements an end-to-end machine learning, deep learning, and MLOps pipeline designed to analyze, predict, and optimize urban mobility across a major traffic corridor. The project spans six core tasks: supervised modeling, unsupervised clustering and association rules, deep learning with SHAP explainability, advanced MLflow tracking, a smart city traffic recommendation system, and a simulated deployment with drift monitoring.

---

## Directory Structure
```text
smart-city-traffic-capstone/
│
├── part3_machine_learning/
│   ├── data/
│   │   └── features_traffic_proxy.csv                # Processed feature dataset
│   ├── Task_1_Supervised_ML.ipynb                    # Classification & Regression models
│   ├── Task_2_Unsupervised_Machine_Learning.ipynb    # K-Means clustering & Apriori rules
│   ├── Task_3_Deep_Learning.ipynb                    # Neural network & SHAP explainability
│   └── Task_4 and 5_Advanced AI and Traffic...ipynb  # MLflow tracking & Routing system
│
├── mlruns/                                           # MLflow local tracking directory
└── README.md                                         # Project documentation

# Dataset Note: Accident Proxy Source
Real Accident Data Status: A direct, real-time historical accident dataset was not publicly available for this specific corridor.

# Proxy Methodology: To overcome this limitation, a documented proxy label (high_risk) was engineered directly from the available traffic volume, weather severity, and speed/density interactions within the traffic dataset to simulate accident likelihood and high-risk bottleneck states reliably.

# How to Run the Project
Prerequisites
Ensure you have Python installed along with the required scientific computing and machine learning libraries:

!pip install pandas numpy scikit-learn tensorflow keras shap mlxtend mlflow

# Execution Steps
Open JupyterLab or Jupyter Notebook within your environment and run the notebooks in sequential order:

# Task 1 (Supervised ML): Open Task_1_Supervised_ML.ipynb to execute data splitting, feature engineering (cyclical time encodings, weather flags), and train/evaluate the Logistic Regression, Random Forest Classifier, Linear Regression, and Random Forest Regressor models.

# Task 2 (Unsupervised ML): Open Task_2_Unsupervised_Machine_Learning.ipynb to execute K-Means clustering (segmenting traffic operational profiles) and Apriori association rule mining.

# Task 3 (Deep Learning & Explainability): Open Task_3_Deep_Learning.ipynb to train the sequential Keras neural network and generate SHAP explainability plots via the surrogate Random Forest model.

# Task 4, 5 & 6 (Advanced AI, Recommendations & MLOps): Open Task_4 and 5_Advanced AI and Traffic Recommendation System.ipynb to initialize the MLflow tracking server, log runs, execute the smart city travel timing recommendation function, and simulate deployment monitoring.
