from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# -----------------------------
# Setup Chrome for headless browsing
# -----------------------------
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# -----------------------------
# Launch Streamlit app
# -----------------------------
app_url = "http://localhost:8501"
driver.get(app_url)

# Wait for Streamlit to load completely
time.sleep(8)  # adjust if your app takes longer

# Create screenshots directory
os.makedirs("screenshots", exist_ok=True)

# -----------------------------
# Capture Single vs Comparison Mode
# -----------------------------
modes = ["Single Clustering", "Comparison Mode"]
for mode in modes:
    # Set display mode in sidebar
    display_radio = driver.find_element(By.XPATH, "//label[contains(text(),'Display Mode')]/following-sibling::div//input[@value='" + mode + "']")
    driver.execute_script("arguments[0].click();", display_radio)
    time.sleep(3)  # wait for plots to update

    # Capture PCA plot(s)
    pca_elements = driver.find_elements(By.XPATH, "//div[contains(@class,'stPlotlyChart')]")
    for i, el in enumerate(pca_elements):
        el.screenshot(f"screenshots/{mode.replace(' ','_')}_PCA_{i+1}.png")

# -----------------------------
# Capture other sections
# -----------------------------
sections = {
    "Genre_Heatmap": "//h2[contains(text(),'Genre Distribution Heatmap')]/following-sibling::div//canvas",
    "Rating_Boxplot": "//h2[contains(text(),'Rating Distribution Boxplot')]/following-sibling::div//canvas",
    "WordClouds": "//h2[contains(text(),'Word Clouds per Cluster')]/following-sibling::div//canvas"
}

for name, xpath in sections.items():
    elements = driver.find_elements(By.XPATH, xpath)
    for i, el in enumerate(elements):
        el.screenshot(f"screenshots/{name}_{i+1}.png")

# -----------------------------
# Done
# -----------------------------
driver.quit()
print("All screenshots saved in ./screenshots")
