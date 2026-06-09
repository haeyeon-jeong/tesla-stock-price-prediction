# =========================================
# CLEAN + COMBINE NYT AND NBC TESLA NEWS
# For NLP / stock prediction modeling
# =========================================
# %%
import pandas as pd

# -----------------------------------------
# 1. Load files
# -----------------------------------------
nyt = pd.read_csv("nyt_tesla_multiquery_news.csv")
nbc = pd.read_csv("nbc_tesla_news_clean.csv")

# -----------------------------------------
# 2. Standardize NYT structure
# -----------------------------------------
nyt["headline"] = nyt["headline"].fillna("")
nyt["snippet"] = nyt["snippet"].fillna("")
nyt["pub_date"] = pd.to_datetime(nyt["pub_date"], errors="coerce", utc=True).dt.tz_localize(None)

nyt_clean = pd.DataFrame({
    "date": nyt["pub_date"],
    "title": nyt["headline"],
    "summary": nyt["snippet"],
    "url": nyt["web_url"] if "web_url" in nyt.columns else nyt["url"],
    "source": "NYT"
})

# -----------------------------------------
# 3. Standardize NBC structure
# -----------------------------------------
nbc["title"] = nbc["title"].fillna("")
nbc["summary"] = nbc["summary"].fillna("") if "summary" in nbc.columns else ""
nbc["date"] = pd.to_datetime(nbc["date"], errors="coerce", utc=True).dt.tz_localize(None)

nbc_clean = pd.DataFrame({
    "date": nbc["date"],
    "title": nbc["title"],
    "summary": nbc["summary"],
    "url": nbc["url"],
    "source": "NBC"
})

# -----------------------------------------
# 4. Combine datasets
# -----------------------------------------
news = pd.concat([nyt_clean, nbc_clean], ignore_index=True)

# -----------------------------------------
# 5. Basic cleaning
# -----------------------------------------
news["title"] = news["title"].fillna("").astype(str).str.strip()
news["summary"] = news["summary"].fillna("").astype(str).str.strip()
news["url"] = news["url"].fillna("").astype(str).str.strip()

news = news.dropna(subset=["date"]).copy()
news = news[news["title"] != ""].copy()
news = news[news["url"] != ""].copy()

news = news.drop_duplicates(subset=["url"]).copy()
news = news.drop_duplicates(subset=["title", "date"]).copy()

# Restrict to project window
news = news[
    (news["date"] >= pd.Timestamp("2023-01-01")) &
    (news["date"] < pd.Timestamp("2024-02-01"))
].copy()

# -----------------------------------------
# 6. Build text field for filtering / NLP
# -----------------------------------------
news["text_for_nlp"] = (
    news["title"].fillna("") + " " + news["summary"].fillna("")
).str.strip()

# -----------------------------------------
# 7. Keyword-based filtering
# -----------------------------------------
tesla_keywords = [
    "tesla",
    "tsla",
    "autopilot",
    "model y",
    "model 3",
    "cybertruck"
]

musk_keywords = [
    "elon musk",
    "musk"
]

tesla_pattern = "|".join(tesla_keywords)
musk_pattern = "|".join(musk_keywords)

news["is_tesla_related"] = news["text_for_nlp"].str.contains(
    tesla_pattern, case=False, na=False
).astype(int)

news["is_musk_related"] = news["text_for_nlp"].str.contains(
    musk_pattern, case=False, na=False
).astype(int)

news_filtered = news[
    (news["is_tesla_related"] == 1) |
    (news["is_musk_related"] == 1)
].copy()

# -----------------------------------------
# 8. Final sort / preview
# -----------------------------------------
news_filtered = news_filtered.sort_values("date").reset_index(drop=True)

print("Combined rows before filter:", len(news))
print("Rows after filter:", len(news_filtered))
print("\nRows by source:")
print(news_filtered["source"].value_counts())

print("\nPreview:")
print(news_filtered[["date", "source", "title", "summary", "url"]].head(10))

# -----------------------------------------
# 9. Save cleaned combined file
# -----------------------------------------
news_filtered.to_csv("combined_tesla_news_clean.csv", index=False)
print("\nSaved combined_tesla_news_clean.csv")

# %%
