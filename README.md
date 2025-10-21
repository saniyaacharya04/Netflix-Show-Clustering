'''# Netflix Clustering Dashboard

A **modular, interactive dashboard** for exploring Netflix shows using clustering techniques. Users can analyze shows by genre, rating, type, and release year, and compare **K-Means** and **Hierarchical** clustering methods with PCA visualizations, silhouette scores, and cluster-level word clouds.

---

## 🚀 Features

* **Dynamic Filtering Panel**: Filter shows by genres, ratings, type, and release year.
* **Clustering Modes**: Choose between K-Means, Hierarchical, or Comparison mode.
* **Cluster Analysis**:
  * PCA scatter plots of clusters
  * Genre distribution heatmaps per cluster
  * Rating distribution boxplots
  * Cluster cohesion metrics
  * Top shows per cluster table
* **Word Clouds**: Generate cluster-wise word clouds from show descriptions
* **Clustering Comparison**: Side-by-side K-Means vs Hierarchical PCA plots with silhouette scores
* **Download Data**: Export filtered clustered data as CSV

---

## 🛠 Tech Stack

* Python 3.13  
* Streamlit  
* Pandas, NumPy  
* Scikit-learn  
* Seaborn, Matplotlib, Plotly  
* WordCloud  

---

## 👑 Project Structure

```

netflix-show-clustering/
│
├─ app.py                  # Main Streamlit app
├─ requirements.txt        # Dependencies
├─ data/
│   └─ netflix_titles.csv  # Dataset
└─ src/
├─ preprocess.py       # Data loading & filtering
├─ cluster.py          # Clustering functions
├─ visualize.py        # Plots & metrics
└─ utils.py            # Word cloud & helper functions

````

---

## ⚡ Installation & Setup

1. **Clone the repo:**

```bash
git clone https://github.com/<YOUR_USERNAME>/netflix-show-clustering.git      
cd netflix-show-clustering
````

2. **Create a virtual environment:**

```bash
python3 -m venv netflix_env
source netflix_env/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the app:**

```bash
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---

## ✨ Notes

* Requires `netflix_titles.csv` in the `data/` folder.
* Fully modular structure for easy enhancements and deployment.
* `.history`, temporary files, and local screenshots are **excluded from Git**.
* Optional: Deploy on **Streamlit Cloud** or **Heroku** for public access.

---

## 👉 Author

**Saniya Acharya** – [GitHub Profile](https://github.com/saniyaacharya04)

