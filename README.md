#  Electric-Vehicles-Project  

##  Problem Statement  

With the rapid growth of the **Electric Vehicle (EV)** industry, analyzing and forecasting car performance, specifications, and market trends has become crucial for manufacturers, analysts, and consumers.  
However, raw datasets like `Electric_Vehicle_Population_Data.csv` and `detailed_ev_charging_stations.csv` often contain inconsistent, unstructured, or incomplete information — making it difficult to extract insights or predict future trends effectively.  

This project focuses on combining **AI-powered chat interactions** with **data analytics and machine learning** to better understand and communicate EV market patterns.  
It helps users analyze and predict EV performance, price, battery health, and market growth using an intuitive web-based dashboard.  

---

## 📋 Project Overview  

This repository contains two main components:  

### 🔹 Chatbot Development  
A conversational **Streamlit chatbot** capable of answering EV-related queries using basic NLP concepts.  
- Responds to questions about EV models, range, cost, and charging.  
- Can be enhanced using **OpenAI** or **Hugging Face** APIs for smart contextual replies.  

### 🔹 EV Data Analysis & Machine Learning  
Performs **comprehensive data preprocessing, cleaning, and exploratory data analysis (EDA)** on real-world EV datasets to identify:  
- Vehicle performance and manufacturer trends.  
- Charging efficiency and operating costs.  
- EV price, range, and sustainability patterns.  

The project also includes multiple trained ML models for intelligent prediction and analysis.  

---

## ⚙️ Features  

### 🔹 Chatbot  
- Built using **Streamlit** and NLP logic.  
- Provides interactive responses to EV-related queries.  
- Extendable with advanced LLMs (OpenAI, Hugging Face).  

### 🔹 EV Data Analysis  
**Data Cleaning & Preprocessing:**  
Handles missing values, outliers, and inconsistent data formats.  

**Exploratory Data Analysis (EDA):**  
Generates visual insights into range, capacity, and charging trends.  

**Visualization:**  
Displays EV market insights with charts on range, cost, and battery efficiency.  

**Model Training:**  
Uses **Random Forest Regressors** and **Prophet** to train predictive models for:  
- EV price and range  
- Battery health (SoH)  
- EcoScore (sustainability)  
- Maintenance cost  
- EV market forecasting  

**Model Deployment:**  
All models integrated for real-time predictions inside the Streamlit UI.  

---

## 📊 Example Insights  

### From EV Population Data:  
- **Top EV Manufacturers:** Tesla, Nissan, Hyundai, Tata  
- **Average Range (BEV):** ~370 km  
- **EV Registrations:** Rising rapidly from 2018–2025  

### From Charging Station Data:  
- **Leading Operators:** EVgo, ChargePoint, Tata Power  
- **Average cost per kWh:** $0.25  
- **Most popular charger type:** AC Level 2  

### Chatbot Demo Queries:  
- “Which EV has the best range?”  
- “Show me cars under ₹25,00,000.”  
- “Compare Tata Nexon EV and Hyundai Kona.”  
- “Predict the battery health of a 3-year-old EV.”  

---

## 🧠 Machine Learning Models  

| Model | Description | Example Output |
|--------|--------------|----------------|
| **EV Price Predictor** | Predicts EV price based on specifications. | ₹18.5L |
| **Battery Health Estimator** | Predicts battery condition (SoH%) using age, efficiency, and usage. | 89.4% |
| **EcoScore Model** | Calculates a sustainability score (0–100). | 82 |
| **Maintenance Cost Predictor** | Estimates upcoming maintenance/service cost. | ₹6,200 |
| **Market Forecast Model** | Predicts next-year EV growth using Prophet. | +4,700 next year |

---

## 🧰 Technologies Used  

| Area | Tools / Libraries |
|------|--------------------|
| **Programming** | Python 3.x |
| **Data Analysis** | pandas, numpy, matplotlib, seaborn |
| **Machine Learning** | scikit-learn, prophet, joblib |
| **Chatbot / Web App** | Streamlit |
| **Version Control** | Git + GitHub |
| **Environment** | Google Colab / VS Code |


Chatbot Demo Questions:

“Which EV has the best range?”

“Show me cars under $25,000.”

“Compare Tesla and Nissan electric models.”

🔮 Future Improvements

Integrate chatbot with live EV market APIs for real-time insights.

Deploy the dashboard to Streamlit Cloud / Hugging Face Spaces.

Build geo-based recommender systems for charger placement.

Add LLM-powered conversational layer for intelligent Q&A.

Include real-time telemetry data for dynamic model updates.

🧑‍💻 Author

Rushmitha Arelli
🎓 B.Tech Student | Data Science & AI Enthusiast

📧 Email: rushmithaarelli05@gmail.com

💼 GitHub: github.com/Rushmitha05
