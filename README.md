# 📈 Google Stock Price Prediction using LSTM

This project is a **Deep Learning-based stock price prediction web application** built using **LSTM (Long Short-Term Memory)** neural networks and **Streamlit**.

The application predicts the **next day's Google stock price** based on the **previous 10 days of Open prices**.

---

# 🚀 Features

- Predict next-day Google stock price
- Two input methods:
  - Manual Entry using sliders
  - CSV Upload for batch prediction
- Interactive Streamlit web interface
- Uses a pre-trained LSTM deep learning model
- Data scaled using MinMaxScaler

---

# 🧠 Model Information

- Model Type: **LSTM (Long Short-Term Memory)**
- Framework: **TensorFlow / Keras**
- Input: **Last 10 days stock open prices**
- Output: **Predicted next-day open price**

LSTM networks are effective for **time-series forecasting problems like stock prediction**.

---

# 📂 Project Structure

```
Google-Stock-Prediction/
│
├── app.py
├── lstm_model.pkl
├── scaler.pkl
├── Google_Stock_Price_Train.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/google-stock-lstm.git
cd google-stock-lstm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📊 How It Works

1. User inputs the **last 10 days stock prices**
2. Data is **scaled using the saved scaler**
3. Data reshaped into **LSTM format (1, 10, 1)**
4. Model predicts the **next day stock price**
5. Prediction is **inverse scaled and displayed**

---

# 🖥️ Input Methods

### 1️⃣ Manual Entry
Users can enter the last 10 days' prices using **interactive sliders**.

### 2️⃣ CSV Upload
Users can upload a **CSV file containing an `Open` column with at least 10 rows**.

---

# 📦 Requirements

Libraries used in this project:

- streamlit
- numpy
- pandas
- scikit-learn
- joblib
- tensorflow

Install them using:

```bash
pip install -r requirements.txt
```

---

# 📈 Future Improvements

- Add real-time stock data using APIs
- Improve prediction accuracy
- Add visualization graphs
- Deploy using **Streamlit Cloud / AWS**

---

# 👨‍💻 Author

Your Name  
Machine Learning Project
