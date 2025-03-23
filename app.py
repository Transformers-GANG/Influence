import streamlit as st
import google.generativeai as genai
import json
import re
import requests
import os
import json
from dotenv import load_dotenv
from news_senti_model import fetch_news_from_rss, get_sentiment_score, calculate_credibility_score, evaluate_credibility
from get_twitter import TwitterLiteAnalyzer


load_dotenv()

# Configure Gemini API Key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))  # Replace with your actual API key

def get_twitter_profile(handle):
    """Fetches Twitter profile data for the specified handle."""
    if not handle:
        return None
    
    analyzer = TwitterLiteAnalyzer()
    profile = analyzer.generate_profile(handle)
    
    # Create visualization
    viz_file = analyzer.visualize_profile(profile)
    
    return profile, viz_file

def get_person_details_gemini(person_name):
    """Fetches and extracts detailed information about a person using Gemini API."""
    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = f"""
    Provide detailed information about {person_name} in valid JSON format with double quotes.
    Ensure the response is strictly JSON, no extra text.
    The format should be:
    {{
      "full_name": "value",
      "ethnicity": "value",
      "dob": "value",
      "age": "value",
      "net_worth": "value",
      "charity": "value",
      "companies": ["company1", "company2"]
    }}
    If any value is unknown, use "unknown".
    """

    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip() if hasattr(response, "text") else response.candidates[0].content.strip()
        return {"raw_response": json_string}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

def clean_and_parse_json(raw_response):
    """Extracts JSON from a code block and converts it into structured data."""
    try:
        # Remove Markdown code block if present (```json ... ```)
        match = re.search(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL)
        if match:
            json_string = match.group(1).strip()
        else:
            json_string = raw_response.strip()
        
        # Convert JSON string to dictionary
        return json.loads(json_string)
    
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": raw_response}

def clean_data(structured_data):
    """Cleans and reformats the structured data into the desired JSON format."""
    cleaned_data = {
        "Name": structured_data.get("full_name", "unknown"),
        "DOB": structured_data.get("dob", "unknown"),
        "Ethnicity": structured_data.get("ethnicity", "unknown"),
        "Age": structured_data.get("age", "unknown"),
        "Net Worth": structured_data.get("net_worth", "unknown"),
        "Charity": structured_data.get("charity", "unknown"),
        "Companies": structured_data.get("companies", [])
    }
    return cleaned_data

def save_to_json(data, filename="data.json"):
    """Saves the cleaned data to a JSON file."""
    try:
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
        st.success(f"Data successfully saved to {filename}")
    except Exception as e:
        st.error(f"Failed to save data to {filename}: {e}")

def get_wikipedia_image(person_name):
    """Fetches an image URL from Wikipedia based on a person's name."""
    url = f"https://en.wikipedia.org/w/api.php?action=query&titles={person_name}&prop=pageimages&format=json&pithumbsize=400"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        page_id = next(iter(data["query"]["pages"]))
        if "thumbnail" in data["query"]["pages"][page_id]:
            image_url = data["query"]["pages"][page_id]["thumbnail"]["source"]
            return image_url
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None
    except KeyError:
        st.error("Error: Image not found or invalid response.")
        return None

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stMetric {
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        box-shadow: 2px 2px 5px #d9d9d9;
    }
    .stMetric label {
        font-size: 14px !important;
        color: #555555 !important;
    }
    .stMetric div {
        font-size: 18px !important;
        font-weight: bold !important;
        color: #333333 !important;
    }
    .stTable {
        font-size: 14px !important;
    }
    .stSubheader {
        font-size: 20px !important;
        color: #2c3e50 !important;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 5px;
    }
    .company-block {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 10px;
    }
    .company-block span {
        background-color: #2c3e50;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.title("Influence IQ")
st.write("Enter a name below to fetch structured information about the person.")

name = st.text_input("Enter a person's name:")
# Load Twitter mappings from the JSON file
try:
    with open("twittermappings.json", "r") as file:
        twitter_mappings = json.load(file)
except FileNotFoundError:
    st.error("Twitter mappings file not found.")
    twitter_mappings = {}

# Normalize the name and check for a Twitter handle
normalized_name = name.strip().lower()
twitter_handle = None

if "influencers" in twitter_mappings:
    for key, handle in twitter_mappings["influencers"].items():
        if key.strip().lower() == normalized_name:
            twitter_handle = handle
            break

if name:
    # Fetch Wikipedia image
    image_url = get_wikipedia_image(name)

    if image_url:
        st.image(image_url, caption=f"Image of {name} from Wikipedia", width=300, use_container_width=False)
    else:
        st.write("Image not found.")

    with st.spinner("Fetching data..."):
        person_details = get_person_details_gemini(name)
    
    if "error" in person_details:
        st.error(person_details["error"])
    else:
        # Extract and clean the JSON data from the raw response
        structured_data = clean_and_parse_json(person_details["raw_response"])
        
        if "error" in structured_data:
            st.error(structured_data["error"])
        else:
            # Clean the structured data
            cleaned_data = clean_data(structured_data)

            # Save the cleaned data to a JSON file
            save_to_json(cleaned_data)

            # Display the data in a dashboard UI
            st.subheader(f"Details for {cleaned_data['Name']}")

            # Use columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Name", cleaned_data["Name"])
                st.metric("Date of Birth", cleaned_data["DOB"])
                st.metric("Age", cleaned_data["Age"])

            with col2:
                st.metric("Net Worth", cleaned_data["Net Worth"])
                st.metric("Charity", cleaned_data["Charity"])

            # Display Ethnicity in a separate row
            st.metric("Ethnicity", cleaned_data["Ethnicity"])

            # Display companies in a block format
            st.subheader("Companies")
            if cleaned_data["Companies"]:
                # Use custom HTML/CSS to display companies as blocks
                st.markdown(
                    f"""
                    <div class="company-block">
                        {"".join(f'<span>{company}</span>' for company in cleaned_data["Companies"])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.write("No company data available.")
    
    if twitter_handle:
        st.write(f"Found Twitter handle: @{twitter_handle}")
        
        with st.spinner("Fetching Twitter data..."):
            twitter_profile, twitter_viz = get_twitter_profile(twitter_handle)
        
        if twitter_profile:
            st.subheader(f"Twitter Profile for @{twitter_profile['username']}")
            
            # Display Twitter metrics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Followers", f"{twitter_profile['followers']:,}")
                st.metric("Years Active", f"{twitter_profile['years_active']}")
            
            with col2:
                st.metric("Engagement Rate", f"{twitter_profile['engagement_rate']*100:.2f}%")
                st.metric("Content Quality", f"{twitter_profile['content_quality']}/10")
            
            # Display sentiment score
            st.metric("Sentiment Score", f"{twitter_profile['sentiment']:.2f}")
            
            # Display topics
            st.subheader("Top Topics")
            st.markdown(
                f"""
                <div class="company-block">
                    {"".join(f'<span>{topic}</span>' for topic in twitter_profile['topics'])}
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Display the visualization image
            if twitter_viz:
                st.image(twitter_viz, caption=f"Growth Analysis for @{twitter_profile['username']}")
    else:
        st.warning("No Twitter handle found for this person. Add it to your twittermappings.json file to enable Twitter analysis.")
                
    with st.spinner("Fetching related news..."):
        articles = fetch_news_from_rss(name)

    if articles:
        sentiment_score = get_sentiment_score(articles)
        credibility_score = calculate_credibility_score(articles, sentiment_score)
        credibility, nature = evaluate_credibility(credibility_score)

        st.subheader(f"Sentiment & Credibility Analysis for {name}")
        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
        st.metric("Credibility Score", f"{credibility_score:.2f}")
        st.metric("Overall News Nature", nature)
        st.metric("Credibility", credibility)

        st.subheader("Recent News Articles")
        for article in articles:
            st.write(f"[{article['title']}]({article['url']}) - {article['publishedAt']}")
    else:
        st.warning("No relevant news articles found.")