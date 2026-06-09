# =========================================
# NBC NEWS TESLA SCRAPER
# Uses exact NBC result-card selectors
# Summary is optional
# =========================================
# %%
# If needed:
# pip install selenium webdriver-manager pandas

import time
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

QUERY = "tesla"
START_DATE = "01/01/2023"
END_DATE = "02/01/2024"
MAX_PAGES = 15
OUTPUT_FILE = "nbc_tesla_news_clean.csv"

# =========================================
# HOW TO USE THIS SCRAPER
# =========================================
# 1. Run the script.
# 2. The NBC search page opens automatically.
# 3. The script fills the date range.
# 4. If needed, click Apply manually in the left sidebar.
# 5. Press Enter in the console when the filtered page looks correct.
# 6. After each page is scraped:
#    - Click the real Next button in the browser
#    - Wait for the next results page to load
#    - Type: next
#    - Press Enter
# 7. If there are no more pages, just press Enter without typing next.
# 8. Type yes at the end to save the cleaned CSV.
# =========================================

# =========================================
# 1. Start Browser
# =========================================
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 20)

driver.get(f"https://www.nbcnews.com/search/?q={QUERY}")

# =========================================
# 2. Fill Date Filter
# =========================================
wait.until(EC.presence_of_element_located((By.ID, "content")))

from_box = wait.until(EC.presence_of_element_located((By.ID, "datepicker_from")))
to_box = wait.until(EC.presence_of_element_located((By.ID, "datepicker_to")))

driver.execute_script("""
const fromInput = arguments[0];
const toInput = arguments[1];
const fromValue = arguments[2];
const toValue = arguments[3];

fromInput.focus();
fromInput.value = '';
fromInput.value = fromValue;
fromInput.dispatchEvent(new Event('input', { bubbles: true }));
fromInput.dispatchEvent(new Event('change', { bubbles: true }));
fromInput.blur();

toInput.focus();
toInput.value = '';
toInput.value = toValue;
toInput.dispatchEvent(new Event('input', { bubbles: true }));
toInput.dispatchEvent(new Event('change', { bubbles: true }));
toInput.blur();
""", from_box, to_box, START_DATE, END_DATE)

time.sleep(1)

print("From input value:", from_box.get_attribute("value"))
print("To input value:", to_box.get_attribute("value"))

# =========================================
# 3. Click Apply
# =========================================
clicked = False

apply_selectors = [
    (By.XPATH, "//*[@id='pubDate_filter']//*[contains(text(), 'Apply')]"),
    (By.CSS_SELECTOR, "#pubDate_filter .actions"),
    (By.CSS_SELECTOR, "#pubDate_filter button"),
]

for by, selector in apply_selectors:
    try:
        el = wait.until(EC.presence_of_element_located((by, selector)))
        driver.execute_script("arguments[0].click();", el)
        print(f"Clicked Apply using: {selector}")
        clicked = True
        time.sleep(4)
        break
    except Exception:
        pass

if not clicked:
    print("\nPlease click Apply manually in the browser.")
    input("After filtered results appear, press Enter here to continue...")

print("\nCheck the filtered results in the browser.")
input("When the results look correct, press Enter to start scraping...")

# =========================================
# 4. Parse Rows
# =========================================
def clean_text(text):
    if not text:
        return None
    return " ".join(text.split()).strip()

def parse_row(row):
    try:
        link = row.find_element(By.CSS_SELECTOR, "a")
        url = link.get_attribute("href")
    except:
        url = None

    try:
        title = clean_text(row.find_element(By.CSS_SELECTOR, "div.item_text_content h3").text)
    except:
        title = None

    try:
        summary = clean_text(row.find_element(By.CSS_SELECTOR, "div.item_text_content h4").text)
    except:
        summary = None

    try:
        date_text = clean_text(row.find_element(By.CSS_SELECTOR, "div.item_text_content .date").text)
    except:
        date_text = None

    return {
        "title": title,
        "url": url,
        "date_text": date_text,
        "summary": summary
    }

# =========================================
# 5. Scrape Pages
# =========================================
all_rows = []
seen_urls = set()

for page in range(1, MAX_PAGES + 1):
    time.sleep(4)

    rows = driver.find_elements(By.CSS_SELECTOR, "#resultdata .queryly_item_row")
    print(f"\nPage {page}: found {len(rows)} rows")

    if not rows:
        print("No rows found. Stopping.")
        break

    page_added = 0

    for row in rows:
        item = parse_row(row)

        if item["url"] and item["url"] not in seen_urls:
            all_rows.append(item)
            seen_urls.add(item["url"])
            page_added += 1

    print("New rows added from this page:", page_added)
    print("Scraped rows so far:", len(all_rows))

    if page == MAX_PAGES:
        break

    user_input = input(
        "After clicking the real Next button in the browser and waiting for the new page, type 'next' and press Enter. Press Enter only to stop: "
    ).strip().lower()

    if user_input != "next":
        print("Stopping pagination.")
        break

# =========================================
# 6. Clean Data
# =========================================
news = pd.DataFrame(all_rows)

news["title"] = news["title"].apply(clean_text)
news["summary"] = news["summary"].apply(clean_text)
news["date"] = pd.to_datetime(news["date_text"], errors="coerce")

news = news.dropna(subset=["title", "url", "date"]).copy()
news = news.drop_duplicates(subset=["url"]).copy()
news = news.drop_duplicates(subset=["title", "date"]).copy()

news = news[
    (news["date"] >= pd.Timestamp("2023-01-01")) &
    (news["date"] < pd.Timestamp("2024-02-01"))
].copy()

news = news.sort_values("date").reset_index(drop=True)

print("\n=== CLEANED DATA ===")
print("Rows after cleaning:", len(news))
print(news[["date", "title", "url", "summary"]].head(10))

# =========================================
# 7. Save CSV
# =========================================
save_csv = input(f"\nSave cleaned CSV as {OUTPUT_FILE}? Type 'yes' to save: ").strip().lower()

if save_csv == "yes":
    news.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE}")
else:
    print("CSV not saved.")

driver.quit()

# %%
