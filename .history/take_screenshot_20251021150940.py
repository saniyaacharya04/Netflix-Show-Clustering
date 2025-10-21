# take_screenshot.py
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------
# Setup
# ----------------------------
os.makedirs("screenshots", exist_ok=True)

options = webdriver.ChromeOptions()
options.add_argument("--headless")  # Run Chrome in headless mode
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open Streamlit app
driver.get("http://localhost:8501")
time.sleep(5)  # Wait for Streamlit to load

# ----------------------------
# Capture Plotly Charts (st.plotly_chart)
# ----------------------------
charts = driver.find_elements(By.CSS_SELECTOR, "div.stPlotlyChart")
for i, chart in enumerate(charts):
    chart.screenshot(f"screenshots/chart_{i}.png")
    print(f"Saved Plotly chart: chart_{i}.png")

# ----------------------------
# Capture Matplotlib Canvas (Word Clouds, Heatmaps, Boxplots)
# ----------------------------
canvas_elements = driver.find_elements(By.TAG_NAME, "canvas")
for i, canvas in enumerate(canvas_elements):
    canvas.screenshot(f"screenshots/canvas_{i}.png")
    print(f"Saved Matplotlib canvas: canvas_{i}.png")

# ----------------------------
# Capture Tables (Top Shows per Cluster)
# ----------------------------
tables = driver.find_elements(By.TAG_NAME, "table")
for i, table in enumerate(tables):
    table.screenshot(f"screenshots/table_{i}.png")
    print(f"Saved table: table_{i}.png")

# ----------------------------
# Optional: Capture full page screenshot
# ----------------------------
driver.save_screenshot("screenshots/full_page.png")
print("Saved full page screenshot.")

# ----------------------------
# Finish
# ----------------------------
driver.quit()
print("All screenshots captured successfully!")
