# 📈 Market Macro Analyzer — Streamlit App

> Analyzes the inter-relationship of **Crude Oil · Gold (XAU/USD) · Nifty 50 Index**
> using EDA, Correlation, ML Models, Clustering & Time Series analysis.

## 🗂 Project Structure

```
ml_mini_project_be/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── fuel_prices.csv         ← Bundled fuel dataset
├── .streamlit/
│   └── config.toml         ← Dark theme config
└── final_model.ipynb       ← Original notebook (reference)
```

---

## 🚀 Run Locally

```bash
cd /Users/saumyachavan/ml_mini_project_be
pip install -r requirements.txt
streamlit run app.py
```

App opens at → **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
cd /Users/saumyachavan/ml_mini_project_be
git init
git add app.py requirements.txt fuel_prices.csv .streamlit/
git commit -m "Initial Streamlit app"
```
Then create a repo on GitHub and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/ml_mini_project_be.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to → **https://share.streamlit.io**
2. Sign in with your **GitHub** account
3. Click **"New app"**
4. Select your repo: `YOUR_USERNAME/ml_mini_project_be`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **"Deploy!"** ✅

Your app will be live at:
```
https://YOUR_USERNAME-ml-mini-project-be-app-XXXXX.streamlit.app
```

---

## 📊 App Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, price history, economic eras |
| 📊 EDA & Statistics | Descriptive stats, distributions, returns |
| 🔗 Correlation Analysis | Pearson heatmap, scatter, T-test, Chi² |
| 🤖 ML Models | Linear Reg, Logistic Reg, Naive Bayes, Decision Tree |
| 🔵 Clustering | KMeans with elbow curve, 3D scatter |
| 📉 Time Series | ADF test, Granger causality, seasonal decomposition |

---

## ⚠️ Notes

- Gold & Nifty data is fetched from **Yahoo Finance** (internet required)
- Fuel data is loaded from the **bundled `fuel_prices.csv`**
- All data is **cached** with `@st.cache_data` for fast re-renders
