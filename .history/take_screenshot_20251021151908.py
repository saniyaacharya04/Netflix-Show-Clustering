import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# -----------------------------
# Setup Chrome in headless mode
# -----------------------------
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# -----------------------------
# Open Streamlit app
# -----------------------------
url = "http://localhost:8501"
driver.get(url)
time.sleep(5)  # Wait for Streamlit app to fully load

# Create screenshots folder if not exists
os.makedirs("screenshots", exist_ok=True)

# -----------------------------
# Helper function to capture elements
# -----------------------------
def capture_element(xpath, filename):
    try:
        elem = driver.find_element(By.XPATH, xpath)
        elem.screenshot(filename)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Failed to capture {filename}: {e}")

# -----------------------------
# Capture sections
# -----------------------------
capture_element("//h2[contains(text(),'PCA Cluster Plot')]/following-sibling::div", "screenshots/PCA_plot.png")
capture_element("//h2[contains(text(),'Genre Distribution Heatmap')]/following-sibling::div", "screenshots/genre_heatmap.png")
capture_element("//h2[contains(text(),'Rating Distribution Boxplot')]/following-sibling::div", "screenshots/rating_boxplot.png")
capture_element("//h2[contains(text(),'Cluster Cohesion Metrics')]/following-sibling::div", "screenshots/cluster_cohesion.png")
capture_element("//h2[contains(text(),'Word Clouds per Cluster')]/following-sibling::div", "screenshots/wordclouds.png")
capture_element("//h2[contains(text(),'Top Shows per Cluster')]/following-sibling::div", "screenshots/top_shows.png")
capture_element("//h2[contains(text(),'Clustering Comparison: K-Means vs Hierarchical')]/following-sibling::div", "screenshots/comparison_PCA.png")

# -----------------------------
# Close browser
# -----------------------------
driver.quit()
print("All screenshots captured successfully!")
