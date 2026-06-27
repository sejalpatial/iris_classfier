# 🌸 Iris Flower Classifier

A machine learning web app that classifies Iris flowers into 3 species using 3 different ML models — with live predictions, confidence scores, model comparison charts, and a feature scatter plot.

**Live demo → [https://irisclassfier-u3v8p8apbsjknedzxnbl7r.streamlit.app](https://irisclassfier-u3v8p8apbsjknedzxnbl7r.streamlit.app)**

---

## What it does

- Accepts 4 flower measurements as input via sliders (sepal length, sepal width, petal length, petal width)
- Predicts the Iris species in real time: **Setosa**, **Versicolor**, or **Virginica**
- Shows confidence probability for each class
- Lets you switch between 3 ML models and compare their predictions

---

## Models used

| Model                      | Test Accuracy |
| -------------------------- | ------------- |
| Logistic Regression        | 96.66%        |
| K-Nearest Neighbours (KNN) | 100%          |
| Random Forest              | 96.00%        |

All models are trained on an 80/20 train-test split

---

## Tech stack

| Layer           | Tool                             |
| --------------- | -------------------------------- |
| Language        | Python 3.10+                     |
| ML models       | Scikit-learn                     |
| UI & deployment | Streamlit                        |
| Model saving    | Pickle                           |
| Data & charts   | Pandas, NumPy, Matplotlib        |
| Hosting         | Streamlit Community Cloud (free) |

---

## Project structure

```
iris_classifier/
├── iris_model.py              ← Streamlit UI (main file)
├── train.py                   ← Model Training
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
└── models/                    ← Auto-created on first run
    ├── iris_model_logistic.sav
    ├── iris_model_knn.sav
    ├── iris_model_random_forest.sav
```

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/sejalpatial_/iris_classifier.git
cd iris_classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Models are trained and saved automatically on the first run (~5 seconds). After that the app loads instantly.

---

## How it works

```
train.py
  └── Loads UCI Iris dataset (150 samples, 4 features, 3 classes)
  └── Splits into 80% train / 20% test
  └── Trains Logistic Regression, KNN, Random Forest
  └── Saves all models + scaler using joblib

iris_model.py
  └── Loads saved models (cached with @st.cache_resource)
  └── Takes 4 slider inputs from user
  └── Runs prediction + probability on selected model
  └── Accuracy comparison bar chart

```

---

## Dataset

**UCI Iris Dataset** — one of the most well-known datasets in ML.

- 150 samples, 3 classes, 4 features
- Classes: Setosa, Versicolor, Virginica
- Source: Built into `sklearn.datasets.load_iris()`
- Originally published by Ronald Fisher (1936)

---

## Key ML concepts demonstrated

- Multi-class classification
- Feature scaling with StandardScaler
- Train-test split with stratification
- Model evaluation using accuracy score
- Probability output with `predict_proba()`
- Model persistence with pickle
- Comparing multiple models on the same dataset

---

## Deploy your own copy

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set main file to `iris_model.py`
5. Click Deploy — live in ~2 minutes

---

## Author

**[Sejal]** — 2nd year B.Tech student, AI/ML specialization

Connect on LinkedIn → [https://www.linkedin.com/in/sejal-patial/]

---

## License

MIT License — feel free to use, modify, and learn from this project.
