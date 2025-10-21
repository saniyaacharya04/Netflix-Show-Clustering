from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Create screenshots folder
os.makedirs("screenshots", exist_ok=True)

# Selenium options
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--start-maximized')
driver = webdriver.Chrome(options=options)

# Open Streamlit app
driver.get("http://localhost:8501")
time.sleep(5)  # wait for app to load

# Helper to scroll element into view
def capture_element_screenshot(element, filename):
    driver.execute_script("arguments[0].scrollIntoView();", element)
    time.sleep(1)
    element.screenshot(filename)
    print(f"Saved {filename}")

# -----------------------------
# Capture PCA Plot
# -----------------------------
pca_element = driver.find_element(By.XPATH, "//div[contains(@class,'stPlotlyChart')]")
capture_element_screenshot(pca_element, "screenshots/PCA_plot.png")

# -----------------------------
# Capture Genre Heatmap
# -----------------------------
heatmap_element = driver.find_element(By.XPATH, "//div[contains(text(),'Genre Distribution Heatmap')]/following-sibling::div//canvas")
capture_element_screenshot(heatmap_element, "screenshots/Genre_Heatmap.png")

# -----------------------------
# Capture Rating Boxplot
# -----------------------------
rating_element = driver.find_element(By.XPATH, "//div[contains(text(),'Rating Distribution Boxplot')]/following-sibling::div//canvas")
capture_element_screenshot(rating_element, "screenshots/Rating_Boxplot.png")

# -----------------------------
# Capture Word Clouds (Loop all clusters)
# -----------------------------
wordcloud_elements = driver.find_elements(By.XPATH, "//div[contains(text(),'Word Clouds')]//following-sibling::div//canvas")
for i, el in enumerate(wordcloud_elements):
    capture_element_screenshot(el, f"screenshots/WordCloud_Cluster{i}.png")

# -----------------------------
# Capture Top Shows Table
# -----------------------------
tables = driver.find_elements(By.XPATH, "//div[contains(text(),'Top Shows per Cluster')]/following-sibling::div//table")
for i, table in enumerate(tables):
    capture_element_screenshot(table, f"screenshots/TopShows_Cluster{i}.png")

# -----------------------------
# Capture Clustering Comparison
# -----------------------------
comparison_elements = driver.find_elements(By.XPATH, "//div[contains(text(),'Clustering Comparison')]//following-sibling::div//div[contains(@class,'stPlotlyChart')]")
for i, el in enumerate(comparison_elements):
    capture_element_screenshot(el, f"screenshots/Comparison_{i}.png")

driver.quit()
print("All screenshots captured!")
