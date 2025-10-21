from selenium import webdriver
from selenium.webdriver.chrome.service import Service
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

# Modes to capture
modes = ["Single Clustering", "Comparison Mode"]

for mode in modes:
    # Set the Display Mode using Streamlit query parameter
    # e.g., ?display_mode=Single+Clustering
    driver.get(f"{app_url}?display_mode={mode.replace(' ', '+')}")
    time.sleep(5)  # wait for plots to update

    # Scroll down gradually to ensure lazy-loaded elements appear
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)

    # Capture Plotly charts
    plotly_elements = driver.find_elements("css selector", "div.stPlotlyChart")
    for i, el in enumerate(plotly_elements):
        el.screenshot(f"screenshots/{mode.replace(' ','_')}_PCA_{i+1}.png")

    # Capture canvas elements (heatmaps, boxplots, wordclouds)
    canvas_elements = driver.find_elements("tag name", "canvas")
    for i, el in enumerate(canvas_elements):
        el.screenshot(f"screenshots/{mode.replace(' ','_')}_Canvas_{i+1}.png")

driver.quit()
print("All screenshots saved in ./screenshots")
