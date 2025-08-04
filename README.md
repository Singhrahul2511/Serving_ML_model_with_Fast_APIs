# 🧠 Insurance Premium Prediction Service

> **Predict insurance premium categories (Low, Medium, High) using a machine learning-powered web service.**

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-brightgreen?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-ff4b4b?logo=streamlit)
![License](https://img.shields.io/github/license/yourusername/insurance_premium_predictor)

---

## 📌 Overview

This project is a **modular ML web application** designed to predict insurance premium categories (`Low`, `Medium`, `High`) based on user inputs. It is built with a **decoupled architecture**:

- 🛠️ **Model Training Pipeline** — Trains and serializes a scikit-learn pipeline.
- ⚙️ **Backend API (FastAPI)** — Exposes a RESTful prediction endpoint.
- 🎛️ **Frontend App (Streamlit)** — Collects user inputs and displays prediction results interactively.

> **Design Focus:** Scalability, modularity, and rapid development

---

## 🧱 System Architecture

```mermaid
graph TD
    A[User Input via Streamlit UI] --> B[HTTP POST /predict]
    B --> C[FastAPI Backend]
    C --> D[Pydantic Validation]
    D --> E[Feature Engineering]
    E --> F[Trained ML Model (Random Forest)]
    F --> G[Prediction Output]
    G --> H[Response Sent as JSON]
    H --> I[Streamlit UI Displays Result]
🗂️ Project Structure
insurance_premium_predictor/
├── app/                   # FastAPI backend logic
│   └── main.py
├── data/
│   └── insurance.csv      # Training dataset
├── model/
│   └── model.pkl          # Serialized ML model
├── scripts/
│   └── train_model.py     # Model training pipeline
├── ui/
│   └── streamlit_app.py   # Streamlit frontend
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
🚀 Getting Started
✅ Prerequisites
Python 3.9+

pip and venv installed

🛠️ Installation & Setup
# 1. Clone the repository
git clone https://github.com/<your-username>/insurance_premium_predictor.git
cd insurance_premium_predictor

# 2. Create & activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
🧠 Train the Model
Before starting the backend, train the ML model using:

python scripts/train_model.py
A model.pkl file will be generated in the model/ directory.

💻 Running the Application
🔌 Terminal 1: Start the FastAPI Backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
Access API documentation at: http://127.0.0.1:8000/docs

🖥️ Terminal 2: Start the Streamlit Frontend
# Make sure virtual environment is active
streamlit run ui/streamlit_app.py
The app will open automatically in your default browser.

🧪 Example Input
Feature	Value
Age	35
Gender	Male
City	Tier 2
Weight (kg)	72
Height (cm)	174
Smoking Status	No
Prediction Output: Medium

🧰 Tech Stack
Component	Tech
ML Model	Scikit-learn, RandomForestClassifier
Backend	FastAPI, Pydantic
Frontend	Streamlit
Deployment	Uvicorn (dev server)
Dev Tools	Python 3.9, Git, Virtualenv
📎 License
Distributed under the MIT License. See LICENSE for more information.

🙋‍♂️ Author
Rahul Kumar
📫 Email
🔗 LinkedIn
🌐 Portfolio
🎥 YouTube

