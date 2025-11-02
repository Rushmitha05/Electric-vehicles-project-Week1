# Electric-vehicles-project-Week1
Problem Statement

With the rapid growth of the electric vehicle (EV) industry, analyzing and forecasting car performance, specifications, and market trends has become crucial for manufacturers, analysts, and consumers. However, raw datasets like Electric_Vehicle_Population_Data.csv and detailed_ev_charging_stations.csv often contain inconsistent, unstructured, or incomplete information — making it difficult to extract insights or predict future trends effectively.

This project focuses on combining AI-powered chat interactions with data analytics and visualization to better understand and communicate EV market patterns.

📋 Project Overview

This repository contains two main components:

🔹 Chatbot Development

A simple, conversational chatbot prototype built with Streamlit, capable of handling EV-related user queries and responses using NLP.

🔹 EV Data Analysis

Comprehensive data preprocessing, cleaning, and exploratory data analysis (EDA) on real-world EV datasets to identify performance patterns, manufacturer trends, and charging efficiency.

Goal: To merge interactive AI conversation with analytics — enabling intelligent, data-driven insights into the EV industry.
⚙️ Features
🔹 Chatbot

Built using Streamlit and basic NLP logic.

Responds to EV-related queries (model, range, cost, etc.).

Can be extended with OpenAI or Hugging Face APIs for smarter conversations.

🔹 EV Data Analysis

Data Cleaning & Preprocessing:
Handles missing values, inconsistent formats, and outliers.

Exploratory Data Analysis (EDA):
Insightful visualizations and metrics across EV models and stations.

Visualization:
Charts showing relationships between charging capacity, range, and cost efficiency.

Model Training:
Trains a Random Forest Regressor to predict numeric targets (e.g., EV price or range).

Model Deployment:
Integrated within Streamlit UI — real-time prediction using trained model.
📊 Example Insights

From EV Population Data:

Top EV Manufacturers: Tesla, Nissan, Chevrolet, Ford

Average Electric Range (BEV): ~230 miles

Increasing EV registrations from 2018–2024

From Charging Station Data:

Leading Operators: EVgo, ChargePoint, Greenlots

Average cost per kWh: $0.25

70% of stations powered by renewable energy

Most popular charger type: AC Level 2

Chatbot Demo Questions:

“Which EV has the best range?”

“Show me cars under $25,000.”

“Compare Tesla and Nissan electric models.”

🔮 Future Improvements

Integrate chatbot with real-time EV market APIs.

Add ML-based price prediction using updated datasets.

Deploy Streamlit dashboard to Streamlit Cloud / Hugging Face Spaces.

Add REST API endpoints for remote data access.

Enhance chatbot with retrieval-based context using LLMs.

🧑‍💻 Author

Rushmitha Arelli
🎓 B.Tech Student | Data Science & AI Enthusiast

📧 Email: rushmithaarelli05@gmail.com

💼 GitHub: github.com/Rushmitha05
