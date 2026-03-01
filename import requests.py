import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "https://www.shl.com/solutions/products/product-catalog/"

response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

product_links = []

for a in soup.find_all("a", href=True):
    if "/products/" in a["href"]:
        full_url = "https://www.shl.com" + a["href"]
        product_links.append(full_url)

product_links = list(set(product_links))
print("Total links found:", len(product_links))