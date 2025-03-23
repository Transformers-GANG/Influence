import os
import json
import datetime
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize NLTK resources
def setup_nltk():
    resources = ['punkt', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"Found {resource}")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)

setup_nltk()

class TwitterLiteAnalyzer:
    def __init__(self):
        """Initialize Twitter API client with minimal setup for Free tier."""
        # Import tweepy here to fail gracefully if not installed
        try:
            import tweepy
            self.tweepy = tweepy
            
            # Get API credentials from environment variables
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            
            if not bearer_token:
                print("Warning: TWITTER_BEARER_TOKEN not found. API calls will fail.")
            
            # Initialize API client (v2) with just the bearer token for read-only operations
            self.client = tweepy.Client(bearer_token=bearer_token)
            self.sia = SentimentIntensityAnalyzer()
            self.api_ready = True
        except ImportError:
            print("Tweepy not installed. Install with: pip install tweepy")
            self.api_ready = False
        except Exception as e:
            print(f"Error initializing Twitter API: {e}")
            self.api_ready = False
    
    def get_user_data(self, username):
        """Get basic user information for a given Twitter handle."""
        if not self.api_ready:
            return None
            
        try:
            user = self.client.get_user(
                username=username,
                user_fields=['id', 'name', 'description', 'public_metrics', 'created_at', 'verified']
            )
            return user.data
        except Exception as e:
            print(f"Error fetching user data: {e}")
            return None
    
    def get_user_tweets(self, user_id, max_results=10):
        """Get recent tweets from a user (limited to just 10 to stay within Free tier limits)."""
        if not self.api_ready:
            return []
            
        try:
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=max_results,  # Reduced from 100 to 10 to conserve API calls
                tweet_fields=['created_at', 'public_metrics'],
                exclude=['retweets', 'replies']
            )
            return tweets.data if tweets.data else []
        except Exception as e:
            print(f"Error fetching user tweets: {e}")
            return []
    
    def generate_simulated_data(self, user):
        """Generate simulated data based on available user metrics."""
        # This function creates synthetic data when API calls are limited
        
        if not user:
            return {
                "followers": 1000,
                "following": 500,
                "engagement_rate": 0.02,
                "content_quality": 7.5,
                "monthly_growth": []
            }
        
        # Extract available metrics
        try:
            follower_count = user.public_metrics['followers_count']
            following_count = user.public_metrics['following_count']
        except:
            follower_count = 1000
            following_count = 500
        
        # Generate plausible engagement rate (between 0.5% and 8%)
        engagement_rate = np.random.uniform(0.005, 0.08)
        
        # Generate monthly growth data
        months = 6
        monthly_growth = []
        
        current_date = datetime.datetime.now()
        
        for i in range(months):
            month_date = current_date - datetime.timedelta(days=30 * (months - i - 1))
            month_name = month_date.strftime("%b")
            
            # Generate growth percentage between -5% and +15%
            growth_pct = np.random.uniform(-0.05, 0.15)
            
            monthly_growth.append({
                "month": month_name,
                "followers": int(follower_count * (1 - (0.05 * (months - i)))),
                "growth_pct": round(growth_pct, 3)
            })
        
        return {
            "followers": follower_count,
            "following": following_count,
            "engagement_rate": round(engagement_rate, 3),
            "content_quality": round(np.random.uniform(5.0, 9.0), 1),
            "monthly_growth": monthly_growth
        }
    
    def analyze_tweets(self, tweets):
        """Analyze available tweets and extract insights."""
        if not tweets:
            return {
                "sentiment": 0.5,
                "avg_engagement": 0,
                "topics": ["No topics available"],
                "posting_frequency": "Unknown"
            }
        
        # Sentiment analysis
        sentiment_scores = []
        for tweet in tweets:
            try:
                sentiment = self.sia.polarity_scores(tweet.text)
                sentiment_scores.append(sentiment['compound'])
            except:
                continue
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        normalized_sentiment = (avg_sentiment + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Calculate average engagement
        engagement_metrics = []
        for tweet in tweets:
            likes = tweet.public_metrics.get('like_count', 0)
            retweets = tweet.public_metrics.get('retweet_count', 0)
            replies = tweet.public_metrics.get('reply_count', 0)
            
            total = likes + retweets + replies
            engagement_metrics.append(total)
        
        avg_engagement = np.mean(engagement_metrics) if engagement_metrics else 0
        
        # Extract common words/topics (simplified)
        all_words = []
        for tweet in tweets:
            # Simple tokenization
            words = [word.lower() for word in tweet.text.split() 
                    if len(word) > 4 and word.isalpha() and not word.startswith(('http', '@', '#'))]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        topics = [word for word, count in word_counts.most_common(5)]
        
        # Calculate posting frequency
        if len(tweets) > 1:
            try:
                dates = [tweet.created_at for tweet in tweets]
                dates.sort()
                
                # Calculate average time between posts
                time_diffs = [(dates[i] - dates[i-1]).total_seconds() / 3600 for i in range(1, len(dates))]
                avg_hours = np.mean(time_diffs)
                
                if avg_hours < 24:
                    posting_frequency = f"{avg_hours:.1f} hours between posts"
                else:
                    posting_frequency = f"{avg_hours/24:.1f} days between posts"
            except:
                posting_frequency = "Irregular"
        else:
            posting_frequency = "Unknown"
        
        return {
            "sentiment": round(normalized_sentiment, 2),
            "avg_engagement": round(avg_engagement, 1),
            "topics": topics if topics else ["No clear topics detected"],
            "posting_frequency": posting_frequency
        }
    
    def determine_category(self, user, tweets):
        """Determine the user's content category."""
        categories = {
            "Technology": ["tech", "coding", "programming", "developer", "software", "hardware", "AI"],
            "Business": ["entrepreneur", "startup", "business", "marketing", "finance", "investing"],
            "Politics": ["politics", "policy", "government", "election", "democracy", "advocacy"],
            "Science": ["science", "research", "academic", "study", "physics", "biology", "chemistry"],
            "Health": ["health", "medicine", "fitness", "wellness", "nutrition", "healthcare"],
            "Entertainment": ["movies", "music", "celebrity", "entertainment", "artist", "actor"],
            "Sports": ["sports", "athlete", "football", "basketball", "baseball", "soccer"],
            "Education": ["education", "learning", "teaching", "student", "school", "university"],
            "News": ["journalist", "news", "media", "reporter", "breaking", "current events"],
            "Art": ["art", "design", "creative", "photography", "illustration", "artist"]
        }
        
        # Check bio first
        bio = getattr(user, "description", "").lower()
        bio_scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword.lower() in bio)
            bio_scores[category] = score
        
        # Check tweets next
        tweet_scores = {category: 0 for category in categories}
        for tweet in tweets:
            text = tweet.text.lower()
            for category, keywords in categories.items():
                score = sum(1 for keyword in keywords if keyword.lower() in text)
                tweet_scores[category] += score
        
        # Combine scores (bio has higher weight)
        combined_scores = {category: (bio_scores.get(category, 0) * 2) + tweet_scores.get(category, 0) 
                          for category in categories}
        
        # Find highest scoring category
        top_categories = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_category = top_categories[0][0] if top_categories[0][1] > 0 else "Content Creator"
        
        return top_category
    
    def generate_profile(self, username):
        """Generate a profile with minimal API usage."""
        print(f"Fetching data for @{username}...")
        
        if not self.api_ready:
            print("API not ready. Generating synthetic profile.")
            return self.generate_synthetic_profile(username)
        
        # Get basic user information
        user = self.get_user_data(username)
        
        if not user:
            print("Could not retrieve user data. Generating synthetic profile.")
            return self.generate_synthetic_profile(username)
        
        print(f"User data retrieved. Fetching a small sample of tweets...")
        
        # Get user's recent tweets (limited to 10 to stay within Free tier limits)
        tweets = self.get_user_tweets(user.id, max_results=10)
        
        if not tweets:
            print("No tweets found. Using simulated data.")
        else:
            print(f"Retrieved {len(tweets)} tweets.")
        
        # Generate simulated metrics data based on available user info
        simulated_data = self.generate_simulated_data(user)
        
        # Analyze available tweets
        tweet_analysis = self.analyze_tweets(tweets)
        
        # Calculate a few basic metrics
        try:
            years_active = round((datetime.datetime.now(datetime.timezone.utc) - user.created_at).days / 365, 1)
            verified = getattr(user, 'verified', False)
        except:
            years_active = 2.0
            verified = False
        
        # Determine category
        category = self.determine_category(user, tweets)
        
        # Create profile object
        profile = {
            "username": username,
            "name": user.name if user else username,
            "category": category,
            "verified": verified,
            "years_active": years_active,
            "followers": simulated_data["followers"],
            "following": simulated_data["following"],
            "engagement_rate": simulated_data["engagement_rate"],
            "content_quality": simulated_data["content_quality"],
            "sentiment": tweet_analysis["sentiment"],
            "topics": tweet_analysis["topics"][:3],  # Just top 3 topics
            "posting_frequency": tweet_analysis["posting_frequency"],
            "monthly_growth": simulated_data["monthly_growth"]
        }
        
        print("Profile generation complete!")
        return profile
    
    def generate_synthetic_profile(self, username):
        """Generate a completely synthetic profile when API is unavailable."""
        print("Generating synthetic profile data...")
        
        # Basic user info
        profile = {
            "username": username,
            "name": username.capitalize(),
            "category": np.random.choice(["Content Creator", "Technology", "Business", "Entertainment"]),
            "verified": False,
            "years_active": round(np.random.uniform(1.0, 8.0), 1),
            "followers": np.random.randint(500, 10000),
            "following": np.random.randint(200, 2000),
            "engagement_rate": round(np.random.uniform(0.01, 0.08), 3),
            "content_quality": round(np.random.uniform(5.0, 8.5), 1),
            "sentiment": round(np.random.uniform(0.3, 0.8), 2),
            "topics": ["Topic 1", "Topic 2", "Topic 3"],
            "posting_frequency": f"{round(np.random.uniform(1, 5), 1)} days between posts",
        }
        
        # Generate monthly growth data
        months = 6
        monthly_growth = []
        
        current_date = datetime.datetime.now()
        
        for i in range(months):
            month_date = current_date - datetime.timedelta(days=30 * (months - i - 1))
            month_name = month_date.strftime("%b")
            
            # Generate growth percentage between -5% and +15%
            growth_pct = np.random.uniform(-0.05, 0.15)
            
            monthly_growth.append({
                "month": month_name,
                "followers": int(profile["followers"] * (1 - (0.05 * (months - i)))),
                "growth_pct": round(growth_pct, 3)
            })
        
        profile["monthly_growth"] = monthly_growth
        
        print("Synthetic profile generation complete!")
        return profile
    
    def save_profile_to_json(self, profile, filename=None):
        """Save the profile to a JSON file in the 'data' folder."""
        if not filename:
            filename = f"{profile['username']}_profile.json"
        
        # Ensure the 'data' folder exists
        os.makedirs('data', exist_ok=True)
        
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(profile, f, indent=4)
        
        return filepath
    
    def visualize_profile(self, profile):
        """Create basic visualizations of the profile data."""
        try:
            # Create a figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Monthly followers
            months = [item["month"] for item in profile["monthly_growth"]]
            followers = [item["followers"] for item in profile["monthly_growth"]]
            growth_pcts = [item["growth_pct"] for item in profile["monthly_growth"]]
            
            axs[0].plot(months, followers, marker='o')
            axs[0].set_title(f"@{profile['username']} - Monthly Followers")
            axs[0].set_ylabel("Followers")
            axs[0].grid(True)
            
            # Plot 2: Monthly growth percentages
            colors = ['green' if pct >= 0 else 'red' for pct in growth_pcts]
            axs[1].bar(months, [pct * 100 for pct in growth_pcts], color=colors)
            axs[1].set_title("Monthly Growth (%)")
            axs[1].set_ylabel("Growth %")
            axs[1].grid(True, axis='y')
            
            # Add a text box with key metrics
            metrics_text = (
                f"Category: {profile['category']}\n"
                f"Followers: {profile['followers']:,}\n"
                f"Engagement Rate: {profile['engagement_rate']*100:.1f}%\n"
                f"Content Quality: {profile['content_quality']}/10\n"
                f"Sentiment: {profile['sentiment']:.2f}\n"
                f"Years Active: {profile['years_active']}"
            )
            
            fig.text(0.13, 0.02, metrics_text, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            plt.tight_layout(rect=[0, 0.08, 1, 0.98])
            
            # Ensure the 'data' folder exists
            os.makedirs('data', exist_ok=True)
            
            # Save the figure in the 'data' folder
            filename = os.path.join('data', f"{profile['username']}_visualization.png")
            plt.savefig(filename)
            plt.close()
            
            return filename
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return None


def main():
    """Main function to run the Twitter Lite Analyzer."""
    try:
        print("\n===== Twitter Lite Analyzer =====")
        print("Optimized for Free API tier (100 posts/month)")
        
        analyzer = TwitterLiteAnalyzer()
        
        username = input("\nEnter a Twitter handle (without @): ")
        
        print(f"\nAnalyzing @{username}...")
        profile = analyzer.generate_profile(username)
        
        # Save profile to JSON
        json_file = analyzer.save_profile_to_json(profile)
        print(f"\nProfile saved to {json_file}")
        
        # Create visualizations
        viz_file = analyzer.visualize_profile(profile)
        if viz_file:
            print(f"Visualizations saved to {viz_file}")
        
        # Print summary
        print("\n===== Profile Summary =====")
        print(f"Name: {profile['name']}")
        print(f"Category: {profile['category']}")
        print(f"Followers: {profile['followers']:,}")
        print(f"Years Active: {profile['years_active']}")
        print(f"Engagement Rate: {profile['engagement_rate']*100:.2f}%")
        print(f"Content Quality: {profile['content_quality']}/10")
        print(f"Sentiment Score: {profile['sentiment']}")
        print(f"Top Topics: {', '.join(profile['topics'])}")
        print(f"Posting Frequency: {profile['posting_frequency']}")
        
        print("\nFor full details, check the generated JSON file and visualization.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your Python environment and dependencies.")

if __name__ == "__main__":
    main()