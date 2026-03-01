# scraper.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json

# Setup browser
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 15)

driver.get("https://www.shl.com/solutions/products/product-catalog/")
time.sleep(5)

product_links = set()

TOTAL_PAGES = 32  # We confirmed there are 32 pages

# Go from last page backwards (stable pagination method)
for page in range(TOTAL_PAGES, 0, -1):

    print(f"Clicking page {page}")

    try:
        page_button = wait.until(
            EC.element_to_be_clickable((By.LINK_TEXT, str(page)))
        )
        driver.execute_script("arguments[0].click();", page_button)
        time.sleep(3)

    except:
        # If page number hidden, click Previous to reveal it
        try:
            prev_button = driver.find_element(By.LINK_TEXT, "Previous")
            driver.execute_script("arguments[0].click();", prev_button)
            time.sleep(2)

            page_button = wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, str(page)))
            )
            driver.execute_script("arguments[0].click();", page_button)
            time.sleep(3)
        except:
            print(f"Could not click page {page}")
            continue

    # Collect product links
    rows = driver.find_elements(By.XPATH, "//table//tr//a")

    for row in rows:
        href = row.get_attribute("href")
        if href and "/product-catalog/" in href:
            product_links.add(href)

driver.quit()

print("Total links found:", len(product_links))

# Save links to JSON file
with open("product_links.json", "w") as f:
    json.dump(list(product_links), f)

print("Saved product_links.json")