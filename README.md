# 🧠 NeuroGuard AI — Real-Time EEG Disease Detection

NeuroGuard AI is a real-time EEG-based detection system that analyzes brainwave patterns to identify neurological conditions such as **seizures, Alzheimer’s, Parkinson’s**, and more. The system uses deep learning models trained on EEG datasets and processes live or uploaded EEG signals to generate predictions instantly. It uses **Explainable ai (xAI)** to determine the anomalies from the eeg graphs and explains the reasoning behind it.

---

## ✨ Key Features

* ⚡ **Real-time EEG signal processing**
* 🧠 **CNN/1D-CNN/ConvNet models** for EEG classification
* 📊 **Multi-disease detection** (Seizure, Alzheimer, Parkinson, etc.)
* 🔍 **Feature extraction** using NumPy, Pandas
* 🖥️ **Interactive dashboard** built with Flask + Plotly.js
* 🔔 **Alert system** for abnormal EEG activity
* 📂 **Support for CSV format**

---

## ⚙️ Tech Stack

| Component              | Description                          |
| ---------------------- | ------------------------------------ |
| **Python**             | Core language                        |
| **TensorFlow / Keras** | Deep learning models                 |
| **NumPy, Pandas**      | Data processing                      |
| **Scikit-Learn**       | Train/test splitting & preprocessing |
| **Flask**              | Backend API & UI server              |
| **Plotly.js**          | Real-time EEG graphs                 |
| **HTML/CSS/JS**        | Frontend interface                   |

---

## 📁 Folder Structure

```
NeuroGuard/
│── app.py               # Main Flask backend
│── data/                # EEG dataset (CSV)
│     └── emotions.csv
│── templates/           # Frontend HTML files
│       └── index.html
└── reports/             # Reports generated
```

---

## 🔧 Setup & Installation

### 1️⃣ Create Environment

```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install Requirements

```
pip install -r requirements.txt
```

Or manually:

```
pip install pandas numpy scikit-learn tensorflow flask matplotlib twilio
```

---

## ▶️ Running the Application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🧪 API Endpoints

### Predict from EEG CSV

**POST** `/predict`

```json
{
  "file_path": "data/sample_eeg.csv"
}
```

**Response**

```json
{
  "disease": "Seizure",
  "confidence": 0.94
}
```

### Real-Time Stream

**GET** `/stream`

* Returns live EEG activity and prediction updates.

---

## 🗂️ Supported Datasets

* Seizure datasets (CHB-MIT, Bonn EEG)
* Alzheimer EEG datasets
* Parkinson EEG datasets
  *(Any CSV dataset with channels/time series will work.)*

---

## ⚠️ Notes

* Works best with **cleaned EEG signals** (artifact removal recommended).
* Model accuracy depends on dataset size & preprocessing quality.
