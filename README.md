## 🌾 CropSense – AI & IoT Based Smart Crop Recommendation System

## 📌 Project Description

CropSense is an **AI and IoT-based agriculture decision support system** that helps farmers choose the most suitable crop based on soil and environmental conditions.

The system uses **IoT sensors to collect real-time field data**, which is then processed by a **machine learning model** to recommend the most suitable crop. The system also provides information such as sowing time, growth duration, fertilizer recommendations, and estimated fertilizer cost.

The project integrates **hardware sensors, machine learning models, and a web interface** to provide a complete smart farming solution.

## 🚜 Features

* 🌱 AI-based crop recommendation
* 📡 Real-time data collection using IoT sensors
* 🌍 Soil and environmental parameter analysis
* 💊 Fertilizer recommendation
* 💰 Estimated fertilizer cost calculation
* ⏳ Crop growth duration prediction
* 📅 Sowing and harvest window estimation
* 🌐 Web-based user interface

## 🧠 Technologies Used

Software

* Python
* Machine Learning (Scikit-learn)
* Pandas & NumPy
* Flask (Backend API)
* HTML / CSS / JavaScript (Frontend)

Hardware (IoT)

* Microcontroller
* Soil sensors
* Environmental sensors
* Data communication with backend system

## 📂 Project Structure

```id="o7v3gc"
CropSense/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── index.html
│
├── models/
│   └── crop_model.pkl
│
├── datasets/
│   ├── crop_requirements.csv
│   ├── crop_stages.csv
│   └── fertilizer_data.csv
│
├── hardware/
│   └── sensor_code.ino
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository

```id="6ru85y"
git clone https://github.com/brindha4999/CropSense.git
```

2. Navigate to the project directory

```id="jnszzv"
cd cropsense
```

3. Install required dependencies

```id="f48h0c"
pip install -r requirements.txt
```

## ▶️ Running the Project

### Step 1 – Run Backend Server

```id="qfypxz"
python backend.py
```

This starts the Flask server.


### Step 2 – Open Frontend

Open:

```id="3v6m1q"
frontend.html
```

in your web browser.


### Step 3 – Sensor Data Collection (Hardware)

IoT sensors collect field parameters such as:

* Soil nutrients
* Soil moisture
* Temperature
* Humidity

These values can be transmitted to the system and used as input for the crop recommendation model.

## 🌍 Impact

CropSense helps farmers make **data-driven farming decisions** by combining **AI and IoT technologies**. This improves crop productivity, reduces resource wastage, and supports sustainable agriculture.

## 📜 License

This project is developed for educational and research purposes.
