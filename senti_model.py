import requests
import json
from textblob import TextBlob  # For sentiment analysis

# Step 1: Identify Reliable News Sources
RELIABLE_SOURCES = [
    "forbes.com",
    "bloomberg.com",
    "wsj.com",  # The Wall Street Journal
    "businessinsider.com",
    "inc.com"
]

# Step 2: Monitor News Mentions
def fetch_news_mentions(person_name, api_key):
    """Fetch news articles mentioning the person from reliable sources."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": person_name,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10  # Limit to 10 articles
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Filter articles from reliable sources
        articles = [
            article for article in data.get("articles", [])
            if any(source in article["url"] for source in RELIABLE_SOURCES)
        ]
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

# Step 3: Conduct Sentiment Analysis
def analyze_sentiment(text):
    """Analyze the sentiment of a given text using TextBlob."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a score between -1 (negative) and 1 (positive)

def get_sentiment_score(articles):
    """Calculate the overall sentiment score for a list of articles."""
    total_score = 0
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        content = f"{title}. {description}"
        total_score += analyze_sentiment(content)
    
    if len(articles) > 0:
        return total_score / len(articles)  # Average sentiment score
    return 0

# Step 4: Evaluate Credibility Score
def evaluate_credibility(sentiment_score):
    """Evaluate the credibility score based on sentiment analysis."""
    if sentiment_score > 0.2:
        return "High Credibility"
    elif -0.2 <= sentiment_score <= 0.2:
        return "Moderate Credibility"
    else:
        return "Low Credibility"

# Main Function
def main():
    # Input: Person's name and NewsAPI key
    person_name = input("Enter the business personality's name: ")
    api_key = "5e3cbae1061b43ed8049fa3e35980c90"  # Replace with your NewsAPI key

    # Step 2: Fetch news mentions
    print(f"Fetching news mentions for {person_name}...")
    articles = fetch_news_mentions(person_name, api_key)

    if not articles:
        print("No articles found from reliable sources.")
        return

    # Step 3: Conduct sentiment analysis
    print("Analyzing sentiment of news articles...")
    sentiment_score = get_sentiment_score(articles)
    print(f"Average Sentiment Score: {sentiment_score:.2f}")

    # Step 4: Evaluate credibility score
    credibility = evaluate_credibility(sentiment_score)
    print(f"Credibility Score: {credibility}")

    # Display top articles
    print("\nTop Articles:")
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title']} ({article['source']['name']})")
        print(f"   URL: {article['url']}")
        print(f"   Published At: {article['publishedAt']}")
        print()

# Run the program
if __name__ == "__main__":
    main()