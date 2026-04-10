# 🌾 CropSense – Smart Crop Recommendation System

## 📌 Project Description

CropSense is a machine learning–based agriculture decision support system that helps farmers choose the most suitable crop based on soil nutrients and environmental conditions. The system analyzes parameters such as nitrogen, phosphorus, potassium, pH, temperature, humidity, and rainfall to recommend the best crop.

The model is deployed through a Python backend and connected to a simple web-based frontend so users can easily input values and receive recommendations.

---

## 🚜 Features

* AI-based crop recommendation
* Soil nutrient analysis (N, P, K, pH)
* Climate factor analysis (temperature, humidity, rainfall)
* Fertilizer recommendation
* Estimated fertilizer cost
* Crop growth duration estimation
* Harvest window prediction

---

## 🧠 Technologies Used

* Python
* Machine Learning (Scikit-learn)
* Pandas & NumPy
* Flask (Backend API)
* HTML / CSS / JavaScript (Frontend)

---

## 📂 Project Structure

```id="m4ns7c"
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
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository

```id="9l9sv8"
git clone https://github.com/yourusername/cropsense.git
```

2. Navigate to the project folder

```id="czqgpw"
cd cropsense
```

3. Install the required dependencies

```id="m3d89s"
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 – Start the Backend Server

Run the backend using Python:

```id="dc9b1c"
python app.py
```

This will start the Flask server.

---

### Step 2 – Open the Frontend

Navigate to the **frontend folder** and open:

```id="m9o3sp"
index.html
```

in your web browser.

---

## 🧪 How to Use

1. Enter soil nutrient values:

   * Nitrogen
   * Phosphorus
   * Potassium

2. Enter environmental conditions:

   * Temperature
   * Humidity
   * Rainfall
   * Soil pH

3. Submit the form.

4. CropSense will display:

   * Recommended crop
   * Suitable sowing month
   * Growth duration
   * Estimated harvest time
   * Fertilizer recommendations
   * Estimated fertilizer cost

---

## 📊 Dataset

The system uses agricultural datasets that contain information about soil nutrients, climate conditions, and crop requirements.

---

## 📈 Future Improvements

* Weather forecasting integration
* Mobile application for farmers
* IoT soil sensor integration
* Satellite-based crop monitoring

---

## 📜 License

This project is created for educational and research purposes.
