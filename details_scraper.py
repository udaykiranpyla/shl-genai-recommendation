# details_scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import json

# Load product links
with open("product_links.json", "r") as f:
    links = json.load(f)

print("Total links loaded:", len(links))

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

data = []

for idx, link in enumerate(links):
    print(f"Scraping {idx+1}/{len(links)}")

    driver.get(link)
    time.sleep(3)

    # Extract name
    try:
        name = driver.find_element(By.TAG_NAME, "h1").text
    except:
        name = ""

    # Extract description
    try:
        description = driver.find_element(
            By.XPATH, "//meta[@property='og:description']"
        ).get_attribute("content")
    except:
        description = ""

    page_text = driver.page_source

    # Extract remote support
    remote_support = "Yes" if "Remote Testing" in page_text else "No"

    # Extract adaptive support
    adaptive_support = "Yes" if "Adaptive" in page_text else "No"

    # Extract duration (basic detection)
    duration = ""
    if "minutes" in page_text.lower():
        duration = "Available"

    data.append({
        "name": name,
        "url": link,
        "description": description,
        "remote_support": remote_support,
        "adaptive_support": adaptive_support,
        "duration": duration
    })

driver.quit()

df = pd.DataFrame(data)
df.to_csv("shl_catalog_full.csv", index=False)

print("Saved shl_catalog_full.csv")