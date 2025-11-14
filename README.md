#  Electric-Vehicles-Project  

## Problem Statement

With the rapid growth of the Electric Vehicle (EV) industry, analyzing and forecasting car performance, specifications, and market trends has become crucial for manufacturers, analysts, and consumers.
However, raw datasets like Electric_Vehicle_Population_Data.csv and detailed_ev_charging_stations.csv often contain inconsistent, unstructured, or incomplete information — making it difficult to extract insights or predict future trends effectively.

This project combines AI-powered chat interactions with data analytics and machine learning to make EV insights accessible. It helps users analyze and predict EV performance, price, battery health, and market growth through an intuitive web-based dashboard.

📋 Project Overview

This repository contains the following components:

Streamlit Web App (Frontend)

Upload raw EV and charging-station datasets, run cleaning, view EDA, train and run models, and chat with an AI assistant.

Data Processing & EDA

Robust cleaning and preprocessing to handle missing values, inconsistent units, and duplicate records.

Interactive visualizations to explore range, efficiency, charger types, costs, and manufacturer trends.

Machine Learning Models

Trained models and heuristics for: Battery Health (SoH), EcoScore, Maintenance Cost, Price/Range prediction, and Market Forecasting.

Chatbot

Rule-based Streamlit chatbot (extendable with OpenAI/Hugging Face) for interactive Q&A: range, cost calculations, charger recommendations, battery health explanations, etc.

⚙️ Features
🔹 Chatbot

Built using Streamlit with a conversational UI.

Answer EV-related queries (e.g., best range, charging cost estimates).

Optional integration with OpenAI or Hugging Face for richer generative responses.

🔹 EV Data Analysis

Cleaning & Preprocessing: Normalizes column names, converts units, handles missing values and outliers, removes duplicates.

EDA: Visualizations for top manufacturers, range distribution, charger types, price distributions, and eco-scores.

Visualizations: Plotly-based interactive charts for dashboards and reports.

Model Training: Random Forest regressors (and heuristic proxies) for:

EV price/range prediction

Battery State of Health (SoH) estimation

EcoScore computation (sustainability)

Maintenance cost estimation

Market growth forecasting (Prophet recommended)

Deployment: Models can be used live in the Streamlit UI or via a lightweight API.

📊 Example Insights

From EV Population Data

Top EV manufacturers (example): Tesla, Nissan, Hyundai, Tata

Average BEV range (example): ~370 km

EV registrations: Increasing trend from 2018–2025

From Charging Station Data

Leading operators (example): EVgo, ChargePoint, Tata Power

Average cost per kWh (example): $0.25

Most common charger type: AC Level 2

Chatbot Example Questions

“Which EV has the best range?”

“Show me cars under ₹25,00,000.”

“Compare Tata Nexon EV and Hyundai Kona.”

“Estimate battery health for a 3-year-old EV.”

🧠 Machine Learning Models
Model	Purpose	Typical Output (example)
EV Price Predictor	Predict price from specs	₹1,850,000
Battery Health Estimator	Estimate SoH (%) from age/usage	89.4%
EcoScore Model	Sustainability score (0–100)	82
Maintenance Cost Predictor	Annual maintenance estimate	₹6,200
Market Forecast Model	Next-year EV growth prediction	+4,700 vehicles

Notes:

Some models use heuristic proxies if labeled data is limited.
Upload Datasets — Upload Electric_Vehicle_Population_Data.csv and detailed_ev_charging_stations.csv or use the demo sample data.

Run Cleaning — The app cleans and normalizes the data (units, numeric parsing, duplicates).

Explore EDA — Interactive charts for manufacturers, range, efficiency, and station metrics.

Train Models — Optionally create models from your dataset (Random Forest regressors and proxies).

Predict — Input a vehicle's specs and get:

Battery SoH (est.)

EcoScore

Estimated maintenance

Predicted price (if model present)

Chat — Use the assistant to ask about range, charging cost, battery health, and more. Optionally enable OpenAI key for generative replies.

Download — Cleaned CSVs and model artifacts can be downloaded.

🧩 Example Inputs & Outputs (short)

Input (Predict form):

Make: Tata
Model: Nexon EV
Model Year: 2022
Battery: 40 kWh
Range: 312 km
Efficiency: 128 Wh/km
Fast charges/week: 1


Output (example):

Battery SoH (est): ~88.2%

EcoScore: 72 /100

Estimated Annual Maintenance: ₹4,700

Predicted Price: ₹1,400,000 (if price model trained)

🔮 Future Improvements

Integrate real-time EV market APIs for live pricing and availability.

Deploy the app to Streamlit Cloud, Hugging Face Spaces, or a cloud VM.

Add geo-based charging-station recommender and optimal charger placement analytics.

Improve chatbot using retrieval-augmented generation (RAG) and LLMs for context-aware answers.

Incorporate telemetry and time-series data to maintain model performance and enable online learning.

🧰 Technologies Used

Programming: Python 3.x

Data Analysis: pandas, numpy, matplotlib, seaborn, plotly

Machine Learning: scikit-learn, prophet (optional), joblib

Web App / Chatbot: Streamlit

API (optional): Flask

Environment: Google Colab / VS Code / Local Python venv

✅ Notes & Tips

Use the demo mode if you do not have datasets ready — it allows exploring the UI and logic.

If dataset columns have unconventional names or embedded units (e.g., "Range* (km)"), the cleaning functions try to extract numeric values and unify names.

For best model results, provide datasets with numeric columns for battery health, range, efficiency, price, and recorded maintenance.

📸 Screenshots

Add polished screenshots under assets/ (e.g., assets/dashboard.png, assets/predict.png, assets/chat.png) for README visualization.

🧑‍💻 Author

Rushmitha Arelli
B.Tech Student | Data Science & AI Enthusiast
📧 rushmithaarelli05@gmail.com

💼 GitHub: https://github.com/Rushmitha05
