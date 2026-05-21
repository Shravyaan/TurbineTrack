# ✈️ TurbineTrack: Predictive Maintenance AI Dashboard

TurbineTrack is an end-to-end Machine Learning and Data Engineering solution designed to predict the **Remaining Useful Life (RUL)** of aircraft turbofan engines. Powered by historical sensor telemetry data from NASA's C-MAPSS dataset, the system cleans real-time data streams, runs them through an optimized machine learning engine, and delivers live asset health monitoring via an interactive web interface deployed on the cloud.

---

## 📊 Project Architecture & Data Pipeline

The system is decoupled into an ingestion pipeline, a frozen serialized intelligence core, and a dynamic frontend presentation layer:

1. **Ingestion & Data Cleaning:** Drops invariant flatline sensors (`setting3`, `sensor1`, `sensor5`, `sensor10`, `sensor16`) identified during Exploratory Data Analysis (EDA).
2. **MinMax Scaling Calibration:** Telemetry inputs are dynamically transformed using a scalar calibrated strictly against baseline operational thresholds (`processed_train_FD001.csv`).
3. **State Isolation:** The dashboard groups incoming files by `engine_id`, isolates the **last logged flight cycle** for each individual asset, and pipes the live feature vector to the model.
4. **Model Execution:** Computes real-time RUL and projects dynamic maintenance status alerts.

---

## 🔬 The Engineering Journey & Model Benchmarks

During the R&D sprint, multiple architectures were trained, optimized, and evaluated against the blind test data stream to capture complex degradation patterns:

| Model Architecture | Test RMSE | Performance Notes |
| :--- | :---: | :--- |
| **Linear Regression** | Baseline | High error; failed to capture non-linear degradation curves of aging core components. |
| **Deep Neural Network ($64 \times 32$)** | ~58.00 | Severely overfit the training pool due to high parameter complexity vs tabular dataset size. |
| **Optimized Random Forest ($max\_depth=8$)** | **43.95** | **🏆 Champion Model.** Achieved the ideal "Goldilocks Zone" by resisting overfitting and proving optimal for tabular data. |

### Robustness Stress-Testing
The data ingestion engine was deliberately stress-tested against highly volatile multi-altitude logs (`test_FD002.txt`) and multi-failure modes (`test_FD003.txt`). The pipeline proved **production-grade resilient**, successfully parsing multi-asset arrays without script exceptions or interface layout failures.

---

## 📁 Repository Structure
```text
├── app.py                      # Interactive Streamlit Web Application Dashboard
├── train_model.py              # Active Production Training Script (Champion RF Core)
├── train_model0.py             # Experimental Lab Notebook (Linear, Neural Network history)
├── test_model.py               # Blind Test Evaluation Pipeline Script
├── turbine_rf_model.pkl        # Serialized, frozen weights of the Champion Model
├── processed_train_FD001.csv   # Pre-processed Master Scaler Reference Sheet
├── requirements.txt            # System dependencies for cloud virtualization
└── .gitignore                  # Active tracking exclusion matrix for clean staging
