# take_screenshot.py
from selenium import webdriver
import time
import os

# Make sure screenshots folder exists
os.makedirs("screenshots", exist_ok=True)

# Start Chrome in headless mode
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')

driver = webdriver.Chrome(options=options)

# Open your Streamlit app
driver.get("http://localhost:8501")
time.sleep(5)  # Wait for app to load

# Take screenshot
driver.save_screenshot("screenshots/dashboard.png")
print("Screenshot saved at screenshots/dashboard.png")

driver.quit()
