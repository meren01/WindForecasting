# Wind Power Forecasting (ML & Data Science) 

This project focuses on predicting wind energy output using **Machine Learning** and time-series analysis. It demonstrates the application of advanced predictive modeling to solve energy stability challenges, specifically tailored for regions like **Adana**.

##  Overview
Accurate wind power forecasting is essential for modern grid management. This project analyzes historical wind data and environmental factors to provide reliable energy production estimates.

##  Machine Learning Models
To ensure the highest accuracy, I implemented and compared several industry-standard algorithms:
* **Random Forest (RF):** Used for robust ensemble learning and handling non-linear data relationships.
* **Decision Trees (DT):** Utilized for clear data branching and interpretability of wind patterns.
* **Artificial Neural Networks (ANN):** Implemented to capture complex, high-dimensional patterns in time-series data.

##  Tech Stack
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

* **Language**: Python.
* **Libraries**: Scikit-Learn (for RF and DT), TensorFlow/Keras (for ANN), Pandas, and NumPy.
* **Analysis**: Regression, Time-Series Analysis, and Algorithm Comparison.

##  Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/meren01/WindForecasting.git](https://github.com/meren01/WindForecasting.git)
   ```
     ```bash
   pip install -r requirements.txt
 
Developed by Murat Eren Furfuru

 ```memraid

flowchart TD
    A[Raw Wind Speed Signal<br/>S(t)] --> B[Preprocess<br/>- Parse DateTime & sort<br/>- Missing value imputation]
    B --> C[Causal Windowing<br/>Take past window: x(t-window+1 : t)]
    C --> D[DWT (Mallat) with db4<br/>Level = 3]
    D --> E[MRA-based Reconstruction (inside window)<br/>Isolate each coefficient & waverec]
    E --> F[Wavelet Features at time t<br/>f(t) = {A3(t), D3(t), D2(t), D1(t)}<br/>(take LAST value of each component)]
    B --> G[Lag Features<br/>Lag_1, Lag_3, ...]
    F --> H[Feature Merge<br/>X(t) = concat( f(t), lags, other vars )]
    G --> H
    H --> I[Train ML Model<br/>(RF / XGBoost / ANN / LSTM ...)]
    I --> J[Forecast Output<br/>Ŝ(t+1) (or Ŝ(t+h))]


```

