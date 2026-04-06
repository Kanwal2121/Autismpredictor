# 🧠 Autism Prediction Machine Learning Application

A complete, production-ready Machine Learning pipeline and interactive web application designed to predict the likelihood of Autism Spectrum Traits in adults using the standard **AQ-10** screening tool.

## 🚀 Features
- **Exploratory Data Analysis**: A detailed Jupyter Notebook (`notebook.ipynb`) covering rigorous data cleaning, handling class imbalances via SMOTE, and comprehensive feature engineering.
- **Robust Machine Learning**: Benchmarked multiple algorithms (Logistic Regression, Random Forest, Decision Tree). The pipeline incorporates `StandardScaler`, `OneHotEncoder`, and trains exclusively on a SMOTE-balanced subset.
- **Interactive Web UI**: A sleek, user-friendly **Streamlit** dashboard mapping human-readable inputs safely to strict model geometries.

## 🛠 Tech Stack
- **Python** (Pandas, Numpy)
- **Scikit-Learn** (Pipelines, Logistic Regression, Transformations)
- **Imbalanced-Learn** (SMOTE for class balancing)
- **Streamlit** (Frontend framework)

## 📂 Project Structure
- `app.py`: The Main Streamlit application file.
- `notebook.ipynb`: The heavily documented Jupyter Notebook where models were trained.
- `Dataset.csv`: The core Adult Autism Screening dataset.
- `preprocessor.joblib` / `best_model.joblib`: Serialized Machine Learning artifacts.
- `requirements.txt`: Deployment dependencies.

## 💻 Local Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Autism-Predictor.git
   cd Autism-Predictor
   ```
2. **Create a virtual environment:**
   *(Optional but highly recommended)*
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the web application:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment
This project is perfectly optimized for immediate deployment on **Streamlit Community Cloud**. Simply connect your GitHub repository and point the Main file path to `app.py`.
