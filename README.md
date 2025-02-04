# 📚 Student Performance Prediction

## 🎯 Project Overview
This project predicts student exam performance based on various factors such as gender, race/ethnicity, parental education, lunch type, and test preparation course. The workflow includes data collection, exploratory data analysis (EDA), preprocessing, model training, and deployment using Flask.

## 🏗️ Key Components

### 🚀 Flask Application (`app.py` and `application.py`)
- **Routes:**
  - `/` - Renders the home page (`index.html`).
  - `/prediction` - Handles form submissions and makes predictions.
- **Form Data Handling:**
  - Collects input features (gender, race/ethnicity, parental education, lunch, test preparation, reading & writing scores).
  - Converts form data into a DataFrame.
- **Prediction Pipeline:**
  - Uses `PredictPipeline` to generate predictions.
  - Displays results on `home.html`.

### 📂 Data Ingestion (`src/components/data_ingestion.py`)
- **Configuration (`DataIngestionConfig`)** manages data paths.
- **Process (`DataIngestion`)**:
  - Reads and loads the dataset.
  - Splits data into training and testing sets.
  - Saves processed data to designated paths.

### 🔄 Data Transformation (`src/components/data_tranformation.py`)
- **Preprocessing Pipelines:**
  - 🧮 **Numerical Pipeline**: Imputes missing values & scales features.
  - 🔤 **Categorical Pipeline**: Handles missing values & encodes categorical variables.
- **Preprocessor** integrates both pipelines for efficient data processing.

### 🔍 Prediction Pipeline (`src/pipeline/predict_pipeline.py`)
- **`CustomData`**: Converts form inputs into a DataFrame.
- **`PredictPipeline`**:
  - Loads the trained model.
  - Makes predictions.

### 🛠️ Utilities (`src/utils.py`)
- `save_object`: Saves models to files.
- `load_model`: Loads saved models.
- `evaluate_models`: Compares the performance of different models.

### 📊 Exploratory Data Analysis (`notebook/1. EDA STUDENT PERFORMANCE.ipynb`)
- **Data Visualization** using various plots.
- **Key Insights** on how different factors influence student performance.

### 🤖 Model Training (`notebook/2. MODEL TRAINING.ipynb`)
- **Model Comparison** of different algorithms.
- **Evaluation Metrics**: R2 Score, MAE, RMSE.

## 📂 Key Files & Directories
- 📌 `app.py`, `application.py` - Flask backend files.
- 📌 `src/components/` - Data ingestion & transformation scripts.
- 📌 `src/pipeline/` - Prediction pipeline.
- 📌 `src/utils.py` - Utility functions.
- 📌 `notebook/EDA.ipynb` - Exploratory data analysis.
- 📌 `notebook/MODEL_TRAINING.ipynb` - Model training.
- 📌 `templates/` - HTML templates.
- 📌 `static/` - Static assets (CSS, images).

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/HardikNegi9/Student-Performace-Prediction.git
cd Student-Performace-Prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```sh
python app.py
```

### 4️⃣ Access the Web Application
Open your browser and go to:
```
http://localhost:5000
```

## 🤝 Contributing
Feel free to fork this repository, make improvements, and submit pull requests!

## 💡 Future Enhancements
- Improve model accuracy with feature engineering.
- Deploy using cloud services (AWS, GCP, or Azure).
- Implement a user authentication system.

---

🚀 **Happy Coding!** 😊

