# =========================================
# NYT ARTICLE SEARCH API
# Tesla / Cybertruck / Elon Musk news
# =========================================
# %%
# If needed:
# pip install requests pandas

import time
import requests
import pandas as pd

API_KEY = "API_KEY" # create api key from new yorktimes and apply here

QUERIES = [
    "Tesla",
    "Cybertruck",
    '"Elon Musk"'
]

BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
month_starts = pd.date_range("2023-01-01", "2024-01-01", freq="MS")

all_articles = []

for query in QUERIES:
    print(f"\n==============================")
    print(f"Starting query: {query}")
    print(f"==============================")

    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(1)

        begin_date = month_start.strftime("%Y%m%d")
        end_date = month_end.strftime("%Y%m%d")

        print(f"\n=== Fetching {query} / {month_start.strftime('%Y-%m')} ===")

        for page in range(10):
            params = {
                "q": query,
                "begin_date": begin_date,
                "end_date": end_date,
                "page": page,
                "sort": "oldest",
                "fq": 'typeOfMaterials:("News","Article") AND desk:("Business","SundayBusiness","Technology")',
                "api-key": API_KEY,
            }

            success = False

            for attempt in range(5):
                response = requests.get(BASE_URL, params=params, timeout=30)

                if response.status_code == 429:
                    wait_time = 10 * (attempt + 1)
                    print(f"{query} {month_start.strftime('%Y-%m')} page {page}: 429, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                success = True
                break

            if not success:
                print(f"Stopping {query} / {month_start.strftime('%Y-%m')} at page {page} due to repeated rate limits.")
                break

            data = response.json()
            response_node = data.get("response", {})
            docs = response_node.get("docs", None)
            meta = response_node.get("meta", {}) or response_node.get("metadata", {})

            if page == 0:
                print(f"{query} {month_start.strftime('%Y-%m')} total hits: {meta.get('hits')}")

            if docs is None:
                print(f"{query} {month_start.strftime('%Y-%m')} page {page}: no more results for this month.")
                break

            if not isinstance(docs, list):
                print(f"{query} {month_start.strftime('%Y-%m')} page {page}: unexpected docs format -> {type(docs)}")
                print(data)
                break

            print(f"{query} {month_start.strftime('%Y-%m')} page {page}: {len(docs)} articles")

            if len(docs) == 0:
                break

            for doc in docs:
                headline_obj = doc.get("headline", {})
                headline = headline_obj.get("main") if isinstance(headline_obj, dict) else None

                all_articles.append({
                    "query": query,
                    "headline": headline,
                    "snippet": doc.get("snippet"),
                    "web_url": doc.get("web_url"),
                    "pub_date": doc.get("pub_date"),
                    "desk": doc.get("desk"),
                    "section_name": doc.get("section_name"),
                    "type_of_material": doc.get("type_of_material"),
                    "document_type": doc.get("document_type"),
                    "uri": doc.get("uri"),
                    "month_bucket": month_start.strftime("%Y-%m")
                })

            time.sleep(3)

news = pd.DataFrame(all_articles)

print("\n=== RAW NYT DATA ===")
print("Rows before cleaning:", len(news))
print(news.head(10))

news["pub_date"] = pd.to_datetime(news["pub_date"], errors="coerce")

news = news.dropna(subset=["headline", "web_url", "pub_date"]).copy()
news = news.drop_duplicates(subset=["web_url"]).copy()
news = news.sort_values("pub_date").reset_index(drop=True)

print("\n=== CLEANED NYT DATA ===")
print("Rows after cleaning:", len(news))
print(news.head(10))

news.to_csv("nyt_tesla_multiquery_news.csv", index=False)
print("Saved nyt_tesla_multiquery_news.csv")

# %%
