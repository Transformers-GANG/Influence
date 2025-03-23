import feedparser
from textblob import TextBlob
from datetime import datetime

def fetch_news_from_rss(person_name):
    rss_urls = [
        "https://www.forbes.com/real-time/feed2/",
        "https://www.bloomberg.com/feeds/podcasts/etf_report.xml",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.businessinsider.com/rss",
        "https://www.inc.com/rss",
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://feeds.bbci.co.uk/news/business/rss.xml",
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ]

    articles = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if person_name.lower() in entry.title.lower() or person_name.lower() in entry.summary.lower():
                articles.append({
                    "title": entry.title,
                    "url": entry.link,
                    "publishedAt": entry.get("published", "Unknown Date"),
                    "source": url.split("//")[1].split("/")[0]
                })
    return articles[:10]

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def get_sentiment_score(articles):
    total_score = sum(analyze_sentiment(article["title"]) for article in articles)
    return total_score / len(articles) if articles else 0

def evaluate_source_reliability(source):
    reliable_sources = ["forbes.com", "bloomberg.com", "wsj.com", "businessinsider.com", "inc.com", "bbc.co.uk"]
    return 1.0 if source in reliable_sources else 0.5

def evaluate_recency(published_at):
    try:
        published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return 0.5
    days_since_published = (datetime.now() - published_date).days
    return 1.0 if days_since_published <= 7 else (0.7 if days_since_published <= 30 else 0.3)

def calculate_credibility_score(articles, sentiment_score):
    total_score = sum((evaluate_source_reliability(article["source"]) + evaluate_recency(article["publishedAt"])) / 2 for article in articles)
    average_score = total_score / len(articles) if articles else 0
    return (average_score + sentiment_score) / 2

def evaluate_credibility(credibility_score):
    if credibility_score > 0.7:
        return "High Credibility", "Positive"
    elif 0.4 <= credibility_score <= 0.7:
        return "Moderate Credibility", "Neutral"
    else:
        return "Low Credibility", "Negative"

def main():
    person_name = input("Enter the business personality's name: ").strip()
    if not person_name:
        print("No name provided. Exiting...")
        return

    articles = fetch_news_from_rss(person_name)
    if not articles:
        print("No articles found.")
        return

    sentiment_score = get_sentiment_score(articles)
    credibility_score = calculate_credibility_score(articles, sentiment_score)
    credibility, nature = evaluate_credibility(credibility_score)

    print(f"Credibility Score: {credibility_score:.2f}")
    print(f"Overall Nature of News: {nature}")
    print(f"Credibility: {credibility}")

if __name__ == "__main__":
    main()
