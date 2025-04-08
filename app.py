
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:07:12 2025

@author: JosephineGitau
"""



# Standard Libraries
import sqlite3   
# import json   
import logging   

# API Requests
import requests  # Fetch data from Google Places and People Data Labs
from urllib.parse import urlparse  #  Extract domain from URLs

# Data Processing
import pandas as pd  

# Machine Learning (Lead Scoring)
import xgboost as xgb  

# Streamlit for Web UI
import streamlit as st  

# Visualization (Maps)
import folium  
from folium.plugins import MarkerCluster  
from streamlit_folium import st_folium  

# Timestamp
from datetime import datetime
import pytz  # For timezone conversion

# People Data Labs
from peopledatalabs import PDLPY

# OS operations
import os

# Logging Configuration 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



#  API Keys 
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PDL_API_KEY = st.secrets["PDL_API_KEY"]




#  Define File Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODEL_PATH = os.path.join(BASE_DIR, "xgboost_lead_ranker.json")  # JSON path
DATABASE_FILE = os.path.join(BASE_DIR, "leads_data.db")  # Database



# DATABASE. Storage and retrieval of business leads using SQLite


def get_database_connection():
    """Establishes a connection to the SQLite database in the correct directory."""
    return sqlite3.connect(DATABASE_FILE)


# Define Ottawa timezone
ottawa_tz = pytz.timezone("America/Toronto")

def get_current_time():
    """Returns the current time in Ottawa timezone."""
    return datetime.now(pytz.utc).astimezone(ottawa_tz).strftime("%Y-%m-%d %H:%M:%S")


#  Column order for database

COLUMN_ORDER = [
    "business_type", "city",  #  User inputs
    "name", "address", "phone", "website", "domain", "latitude", "longitude",
    "types", "rating", "reviews", "company_name", "company_id", "employee_count",
    "linkedin", "founded", "decision_maker_name", "decision_maker_title",
    "decision_maker_email", "decision_maker_phone", "has_decision_maker_phone",
    "has_email", "has_decision_maker_name", "size_priority",
    "normalized_rating", "normalized_reviews", "lead_probability", "lead_rank"
]




#  Ensure database schema exists
def initialize_database():
    """Creates SQLite database tables if they do not exist."""
    
    with sqlite3.connect(DATABASE_FILE) as conn:
    
    
        cursor = conn.cursor()
    
        cursor.execute('''
   
    CREATE TABLE IF NOT EXISTS business_leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        business_type TEXT,
        city TEXT,
        name TEXT,
        address TEXT,
        phone TEXT,
        website TEXT UNIQUE,
        domain TEXT,
        latitude REAL,
        longitude REAL,
        types TEXT,
        rating REAL,
        reviews INTEGER,
        company_name TEXT,
        company_id TEXT,
        employee_count INTEGER,
        linkedin TEXT,
        founded INTEGER,
        decision_maker_name TEXT,
        decision_maker_title TEXT,
        decision_maker_email TEXT,
        decision_maker_phone TEXT,
        has_decision_maker_phone INTEGER,
        has_email INTEGER,
        has_decision_maker_name INTEGER,
        size_priority REAL,
        normalized_rating REAL,
        normalized_reviews REAL,
        lead_probability REAL,
        lead_rank REAL,
        created_at TIMESTAMP DEFAULT (DATETIME('now', 'localtime'))  -- Convert to local time
    )
''')



    conn.commit()
    logging.info(" Database initialized with necessary tables.")




#  Save new business leads to SQLite

def save_to_db(df):
    """Saves new business leads to SQLite while avoiding duplicates."""
    new_entries = []  # Track new businesses

    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()

        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT INTO business_leads (
                        business_type, city,  
                        name, address, phone, website, domain, latitude, longitude, types,
                        rating, reviews, company_name, company_id, employee_count, linkedin,
                        founded, decision_maker_name, decision_maker_title, decision_maker_email,
                        decision_maker_phone, has_decision_maker_phone, has_email, has_decision_maker_name,
                        size_priority, normalized_rating, normalized_reviews, lead_probability, lead_rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row["business_type"],  
                    row["city"],  
                    row["name"],
                    row["address"],
                    row["phone"],
                    row["website"],
                    row["domain"],
                    row["latitude"],
                    row["longitude"],
                    row["types"],
                    row["rating"],
                    row["reviews"],
                    row.get("company_name", "N/A"),  
                    row.get("company_id", None),  
                    row.get("employee_count", None),  
                    row.get("linkedin", "N/A"),
                    row.get("founded", None),
                    row.get("decision_maker_name", "N/A"),
                    row.get("decision_maker_title", "N/A"),
                    row.get("decision_maker_email", "N/A"),
                    row.get("decision_maker_phone", "N/A"),
                    int(row.get("has_decision_maker_phone", 0)),
                    int(row.get("has_email", 0)),
                    int(row.get("has_decision_maker_name", 0)),
                    row.get("size_priority", None),
                    row.get("normalized_rating", None),
                    row.get("normalized_reviews", None),
                    row.get("lead_probability", None),
                    row.get("lead_rank", None)
                ))

                new_entries.append(row["website"])

            except sqlite3.IntegrityError:
                logging.warning(f" Duplicate entry skipped for: {row['website']}")

        conn.commit()  # Commit inside the `with` block

    logging.info(f" Successfully saved {len(new_entries)} new businesses to SQLite.")
    return new_entries  #  Returns newly added businesses for enrichment & ranking





############ GOOGLE PLACES . BUSINESS DATA EXTRACTION.



#  Extract domain from URL
def extract_domain(url):
    """Safely extracts the domain from a URL."""
    if not url or url in ["N/A", "Unknown"]:
        return "N/A"
    return urlparse(url).netloc.replace("www.", "")



# Fetch business data from Google Places API and save to SQLite
def get_companies_from_google(business_type, city, max_results):
    """Fetches businesses from Google Places API and saves only new ones to SQLite."""
    logging.info(f" Searching for: {business_type} in {city}....")

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{business_type} in {city}", "key": GOOGLE_API_KEY}

    try:
        response = requests.get(url, params=params).json()
        businesses = []

        if not response.get("results"):
            logging.warning(f"No businesses found for {business_type} in {city}.")
            return pd.DataFrame()

        for place in response["results"][:max_results]:
            place_id = place.get("place_id", "N/A")
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {
                "place_id": place_id,
                "fields": "name,formatted_address,formatted_phone_number,website,geometry,types,rating,user_ratings_total",
                "key": GOOGLE_API_KEY
            }

            details = requests.get(details_url, params=details_params).json().get("result", {})
            website_url = details.get("website", "N/A")
            domain = extract_domain(website_url)

            businesses.append({
                "business_type": business_type,  # User input
                "city": city,  #  User input
                "name": details.get("name", "N/A"),
                "address": details.get("formatted_address", "N/A"),
                "phone": details.get("formatted_phone_number", "N/A"),
                "website": website_url,
                "domain": domain if domain else "N/A",  
                "latitude": details.get("geometry", {}).get("location", {}).get("lat", "N/A"),
                "longitude": details.get("geometry", {}).get("location", {}).get("lng", "N/A"),
                "types": ", ".join(details.get("types", [])) if details.get("types") else "N/A",
                "rating": details.get("rating", 0), 
                "reviews": details.get("user_ratings_total", 0)  
            })

        # Save only new businesses to SQLite
        df = pd.DataFrame(businesses)
        new_websites = save_to_db(df)  
        
        #  Ensure we always return the requested number of leads without recursion
        attempts = 0
        while len(new_websites) < max_results and attempts < 3:  # Limit attempts to prevent infinite loops
            logging.info(f" Only {len(new_websites)} unique leads saved. Fetching more to meet {max_results} leads.")
            additional_results = max_results - len(new_websites)
            
            #  Fetch additional leads but without recursion
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {"query": f"{business_type} in {city}", "key": GOOGLE_API_KEY}
            response = requests.get(url, params=params).json()
            
            if not response.get("results"):
                logging.warning(" No more new businesses found. Returning available leads.")
                break  # Stop fetching if no more leads are available

            extra_businesses = []
            for place in response["results"][:additional_results]:
                place_id = place.get("place_id", "N/A")
                details_params = {
                    "place_id": place_id,
                    "fields": "name,formatted_address,formatted_phone_number,website,geometry,types,rating,user_ratings_total",
                    "key": GOOGLE_API_KEY
                }
                details = requests.get(details_url, params=details_params).json().get("result", {})
                website_url = details.get("website", "N/A")
                domain = extract_domain(website_url)

                extra_businesses.append({
                    "business_type": business_type,
                    "city": city,
                    "name": details.get("name", "N/A"),
                    "address": details.get("formatted_address", "N/A"),
                    "phone": details.get("formatted_phone_number", "N/A"),
                    "website": website_url,
                    "domain": domain if domain else "N/A",
                    "latitude": details.get("geometry", {}).get("location", {}).get("lat", "N/A"),
                    "longitude": details.get("geometry", {}).get("location", {}).get("lng", "N/A"),
                    "types": ", ".join(details.get("types", [])) if details.get("types") else "N/A",
                    "rating": details.get("rating", 0),
                    "reviews": details.get("user_ratings_total", 0)
                })

            extra_df = pd.DataFrame(extra_businesses)
            extra_websites = save_to_db(extra_df)  # Save additional leads to DB
            new_websites.extend(extra_websites)  # Update the list with newly added leads
            
            attempts += 1  # Prevent infinite loops

        logging.info(f"Successfully fetched and saved {len(new_websites)} new businesses.")
        return df[df["website"].isin(new_websites)]  #  Return exact count of leads

    except requests.RequestException as e:
        logging.error(f"API Error: {e}")
        return pd.DataFrame()

        


        

############### PEOPLE DATA LABS. ENRICHMENT OF DATA.



# ✅ API Setup
PDL_COMPANY_ENRICH_URL = "https://api.peopledatalabs.com/v5/company/enrich"
PDL_PERSON_SEARCH_URL = "https://api.peopledatalabs.com/v5/person/search"

PDL_HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "X-API-Key": PDL_API_KEY
}

# Fetch only unenriched businesses from SQLite
def fetch_unenriched_businesses():
    """Fetches businesses that have not been enriched yet (no employee_count)."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        df = pd.read_sql("SELECT * FROM business_leads WHERE employee_count IS NULL", conn)
    return df


# Enrich and update companies in SQLite

def enrich_new_companies(new_websites , business_type, city):
    """
    Fetches company details and a decision-maker per company from People Data Labs,
    then updates the SQLite database dynamically.
    """
   
    # Fetch unenriched businesses first
    df = fetch_unenriched_businesses()

    logging.info(f"Fetching company data & decision-makers from People Data Labs for {len(df)} businesses...")

    if df.empty:
        logging.info(" No new businesses to enrich.")
        return
    
    #  Get only businesses that need enrichment
    df = fetch_unenriched_businesses()  
    if df.empty:
        logging.info(" No new businesses to enrich.")
        return

    results = []
    for _, row in df.iterrows():
        domain = extract_domain(row["website"])
        if domain == "N/A":
            continue  #  Skip businesses without a valid domain

        # Fetch company enrichment details
        company_data, company_id = enrich_company(domain)
        if not company_id:
            logging.warning(f" No company ID found for domain {domain}. Skipping decision-maker search.")
            decision_maker = {}
        else:
            decision_maker = search_decision_maker(company_id)

        #  Fetch founded year 
        founded = company_data.get("founded", None)
        founded = int(founded) if isinstance(founded, (int, float)) else 0  
        

        # Ensure phone_numbers is a list (Avoid "bool" errors)
        phone_numbers = decision_maker.get("phone_numbers", [])
        if isinstance(phone_numbers, bool):  
            phone_numbers = []

        # Assign first phone number if available, otherwise "N/A"
        phone_number = phone_numbers[0] if isinstance(phone_numbers, list) and phone_numbers else "N/A"

        #  Store Enriched Data
         
        
        enriched_entry = {
    "website": row["website"],  #  Used for updating the right business
    "employee_count": int(company_data.get("employee_count", 0) or 0),
    "company_name": company_data.get("display_name", "N/A"),  #  Correctly assigns company name
    "company_id": company_id,
    "linkedin": company_data.get("linkedin_url", "N/A"),
    "founded": founded,
    
    "decision_maker_name": decision_maker.get("full_name", "N/A"),
    "decision_maker_title": decision_maker.get("job_title", "N/A"),
    "decision_maker_email": decision_maker.get("work_email", "N/A"),
    "decision_maker_phone": phone_number
}
        
        results.append(enriched_entry)
        


    # Update the database with enriched data
    conn = sqlite3.connect("leads_data.db")
    cursor = conn.cursor()

    for record in results:
        
        cursor.execute('''
    UPDATE business_leads 
    SET employee_count = ?, company_name = ?, company_id = ?, linkedin = ?, founded = ?, 
        decision_maker_name = ?, decision_maker_title = ?, decision_maker_email = ?, decision_maker_phone = ? 
    WHERE website = ?
''', (
    record["employee_count"], record["company_name"], record["company_id"], record["linkedin"], record["founded"],
    record["decision_maker_name"], record["decision_maker_title"], record["decision_maker_email"],
    record["decision_maker_phone"], record["website"]
))
    
    if new_websites:  #  Prevent errors when no new businesses are found
        st.session_state["current_run_websites"] = new_websites
        st.session_state["current_run_business_type"] = business_type
        st.session_state["current_run_city"] = city
    else:
        logging.warning("No new businesses were found. Skipping session state update.")
    

    conn.commit()
    conn.close()

    logging.info(f"Successfully enriched {len(results)} new businesses.")





#  Create PDL client
CLIENT = PDLPY(api_key=PDL_API_KEY)


def enrich_company(domain):
    """Fetches company details using People Data Labs API (PDLPY Client)."""
    try:
        logging.info(f"Fetching company data for domain: {domain}")

        #  Call Company Enrichment API using PDLPY
        response = CLIENT.company.enrichment(website=domain).json()

        #  Debugging: Print API Response
        logging.info(f"PDL Response for {domain}: {response}")

        if response.get("status") == 200 and "id" in response:
            logging.info(f"Successfully enriched company data for {domain}")
            return response, response.get("id", None)
        else:
            logging.warning(f" No company data found for {domain}: {response}")
            return {}, None
    except Exception as e:
        logging.error(f" API Request Failed: {e}")
        return {}, None




def search_decision_maker(company_id):
    """Finds a decision-maker using People Data Labs API (PDLPY Client)."""
    if not company_id:
        logging.warning(" No company ID provided, skipping decision-maker search.")
        return {}

    try:
        logging.info(f" Searching for decision-makers in company ID: {company_id}")

        #  Construct elasticsearch query
        es_query = {
            "query": {
                "bool": {
                    "must": [{"term": {"job_company_id": company_id}}]
                }
            }
        }

        # Call PDL person search API
        response = CLIENT.person.search(query=es_query, size=1).json()

        if response.get("status") == 200 and "data" in response:
            logging.info(f" Found decision-maker for company ID {company_id}")
            return response["data"][0] if response["data"] else {}
        else:
            logging.warning(f" No decision-maker found for company ID {company_id}: {response}")
            return {}

    except Exception as e:
        logging.error(f" API Request Failed: {e}")
        return {}





############## FEATURE ENGINEERING



def feature_engineering():
    """Applies feature engineering ONLY on the most recent businesses that haven't been processed yet."""

    #  Fetch only businesses where lead_probability is NULL (new businesses)
    with sqlite3.connect(DATABASE_FILE) as conn:
        query = '''
            SELECT * FROM business_leads 
            WHERE lead_probability IS NULL
            ORDER BY created_at DESC
        '''
        df = pd.read_sql(query, conn)

    if df.empty:
        logging.info(" No new businesses require feature engineering.")
        return

    logging.info(f"Processing {len(df)} new businesses for feature engineering...")

    #  Ensure employee count is numeric
    df["employee_count"] = pd.to_numeric(df["employee_count"], errors="coerce").fillna(0).astype(int)

    #  Convert decision maker columns to binary
    def is_valid_entry(x):
        """Check if the string is a valid entry (not empty, N/A, unknown, etc.)."""
        return isinstance(x, str) and x.strip() and x.strip().lower() not in ["n/a", "unknown", "none", "false", ""]

    df["has_decision_maker_name"] = df["decision_maker_name"].apply(lambda x: 1 if is_valid_entry(x) else 0)
    df["has_email"] = df["decision_maker_email"].apply(lambda x: 1 if is_valid_entry(x) and "@" in x else 0)
    df["has_decision_maker_phone"] = df["decision_maker_phone"].apply(lambda x: 1 if is_valid_entry(x) and any(char.isdigit() for char in x) else 0)

    #  Assign business size category
    def business_size_priority(size):
        if 21 <= size <= 50:
            return 3  # Medium Business - Highest priority
        elif 1 <= size <= 20:
            return 2  # Small Business - Medium priority
        elif size > 50:
            return 1  # Large Business - Lower priority
        else:
            return 0.5  # Unknown size

    df["size_priority"] = df["employee_count"].apply(business_size_priority)

    #  Normalize Rating and Number of Reviews
    df["normalized_rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0) / 5  # Normalize rating (0 to 1)
    max_reviews = df["reviews"].max() + 1  # Prevent division by zero
    df["normalized_reviews"] = pd.to_numeric(df["reviews"], errors="coerce").fillna(0) / max_reviews  # Normalize reviews

    #  Store Engineered Features in Database 
    with sqlite3.connect(DATABASE_FILE) as conn:
        for _, row in df.iterrows():
            conn.execute('''
                UPDATE business_leads
                SET size_priority = ?, normalized_rating = ?, normalized_reviews = ?, 
                    has_decision_maker_name = ?, has_email = ?, has_decision_maker_phone = ?
                WHERE id = ?
            ''', (
                row["size_priority"], row["normalized_rating"], row["normalized_reviews"],
                row["has_decision_maker_name"], row["has_email"], row["has_decision_maker_phone"],
                row["id"]
            ))
        conn.commit()

    logging.info(f" Feature Engineering Completed for {len(df)} new businesses.")

#  Run Feature Engineering for New Leads Only
feature_engineering()






################## XGBOOST MACHINE LEARNING MODEL


#  Machine Learning model loading

@st.cache_resource()
def load_model():
    """Loads the trained XGBoost model, ensuring it is loaded only once."""
    if not os.path.exists(TRAINED_MODEL_PATH):
        st.error(f"❌ Model file not found: {TRAINED_MODEL_PATH}")
        return None  

    # Check if model is already in session state
    if "loaded_model" in st.session_state:
        return st.session_state["loaded_model"]  # Use cached model

    try:
        model = xgb.XGBClassifier()
        model.load_model(TRAINED_MODEL_PATH)
        st.session_state["loaded_model"] = model  # Store in session state
        logging.info(" Model loaded successfully!")
        return model  
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None  


#  Fetch feature-engineered leads needing ranking
def fetch_unranked_leads():
    """Fetches only new business leads that have NOT been ranked yet."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        query = '''
            SELECT * FROM business_leads 
            WHERE lead_probability IS NULL 
            ORDER BY created_at DESC
        '''
        df = pd.read_sql(query, conn)

    return df if not df.empty else pd.DataFrame()


#  Rank new leads using XGBoost
def rank_new_leads():
    """Ranks only new business leads using the trained XGBoost model."""

    # Model is loaded once
    if "loaded_model" not in st.session_state:
        model = load_model()
        if model is None:
            return  # Stop if model is missing
        st.session_state["loaded_model"] = model  # Store in session state
    else:
        model = st.session_state["loaded_model"]

    df = fetch_unranked_leads()
    if df.empty:
        st.warning(" No new leads available for ranking.")
        return

    logging.info(f" Ranking {len(df)} new businesses...")

    # Only current run leads are processed
    if "current_run_websites" in st.session_state and st.session_state["current_run_websites"]:
        df = df[df["website"].isin(st.session_state["current_run_websites"])]

    if df.empty:
        logging.warning(" No matching leads found for the current run.")
        return

    #  Feature order 
    feature_order = [
        "has_decision_maker_phone", "has_email", "has_decision_maker_name",
        "size_priority", "normalized_rating", "normalized_reviews"
    ]
    
    # DataFrame to contain required features (Convert NaN to numeric)
    X = df[feature_order].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Predict Lead Probability
    try:
        predictions = model.predict_proba(X)[:, 1]  # Ensure model outputs probabilities
        df["lead_probability"] = pd.Series(predictions, index=df.index).fillna(0)

        #  lead_rank is calculation 
        if df["lead_probability"].sum() == 0:
            df["lead_rank"] = 0  # If all probabilities are zero, set rank to 0
        else:
            df["lead_rank"] = df["lead_probability"].rank(ascending=False).fillna(0).astype(int)

    except Exception as e:
        st.error(f" Error during lead ranking: {e}")
        return

    #  Update SQLite with the new rankings
    with sqlite3.connect(DATABASE_FILE) as conn:
        for _, row in df.iterrows():
            conn.execute('''
                UPDATE business_leads 
                SET lead_probability = ?, lead_rank = ? 
                WHERE website = ?
            ''', (row["lead_probability"], row["lead_rank"], row["website"]))

        conn.commit()

    logging.info(f" Lead ranking updated in SQLite for {len(df)} new leads.")
    st.success(f" Lead ranking updated for {len(df)} new leads.")






########## CURRENT USER INPUT RUN/LEADS


#  Display Ranked Leads from SQLite

#  Fetch Current Run Leads from SQLite
def fetch_current_run_leads():
    """Fetches ranked business leads for the current run only."""
    if "current_run_websites" not in st.session_state or not st.session_state["current_run_websites"]:
        st.warning(" No businesses from the current run detected. Please run lead generation first.")
        return pd.DataFrame()  # Return empty dataFrame if no leads exist

    placeholders = ", ".join(["?"] * len(st.session_state["current_run_websites"]))
    query = f'''
        SELECT name, address, phone, website, rating, reviews AS "Number of Reviews", 
               linkedin AS "LinkedIn URL", founded AS "Founded", employee_count, 
               decision_maker_name, decision_maker_title, decision_maker_email, 
               decision_maker_phone, lead_rank, latitude, longitude
        FROM business_leads
        WHERE website IN ({placeholders})
        ORDER BY lead_rank ASC
    '''

    with sqlite3.connect(DATABASE_FILE) as conn:
        df = pd.read_sql(query, conn, params=tuple(st.session_state["current_run_websites"]))

    return df

#  Display Leads in Streamlit Table
def display_leads_table(df):
    """Displays the business leads in a formatted Streamlit table."""
    if df.empty:
        st.warning(" No ranked leads found for this run.")
        return

    # Select Columns for Display
    required_columns = [
        "name", "address", "phone", "website", "rating", "Number of Reviews", "LinkedIn URL", "Founded",
        "employee_count", "decision_maker_name", "decision_maker_title", "decision_maker_email", 
        "decision_maker_phone", "lead_rank"
    ]

    #  Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = "N/A" if "decision_maker" in col or col in ["LinkedIn URL", "phone"] else 0  

    #  Display Dataframe in Streamlit
    st.subheader("Business Leads")
    st.dataframe(df[required_columns])




################ MAP



# Display leads on a map
def display_leads_map(df):
    """Displays only businesses from the current run on a map."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        st.warning("No valid locations to display on the map.")
        return

    st.subheader("Business Locations")

    #  Compute Map Center
    map_center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    #  Add Markers for Businesses
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"<b>{row['name']}</b><br>{row['address']}",
            tooltip=row["name"],
            icon=folium.Icon(color="blue")
        ).add_to(marker_cluster)

    #  Display Map in Streamlit
    st_folium(m, width=700, height=500)





############### MAIN. User Interface


# Main Streamlit App
def main():
    initialize_database()  # Ensure database is initialized
    
    st.title("GIS AI-Driven Lead Generation")

    #  User Inputs
    business_type = st.text_input("Enter Business Type (e.g., Construction, Plumbing, Roofing):")
    city = st.text_input("Enter City (e.g., Ottawa, Toronto, Vancouver):")
    max_results = st.number_input("Enter the number of results to fetch:", min_value=1, max_value=20, value=10, step=1)

    if st.button(" Find & Rank Leads"):
        if not business_type or not city:
            st.warning("Please enter a Business Type and City to proceed.")
            return

        with st.spinner("Fetching businesses..."):
            businesses_df = get_companies_from_google(business_type, city, max_results)

        #  Extract new websites that were inserted into the database
        new_websites = businesses_df["website"].tolist() if not businesses_df.empty else []

        if businesses_df.empty:
            st.warning(f" No businesses found for: {business_type} in {city}. Try a different category or city.")
            logging.error(f" Google API returned no businesses for: {business_type} in {city}")
            return
        
        #  Store the new businesses in session state
        st.session_state["current_run_websites"] = new_websites

        #  Enrich only new businesses
        with st.spinner(" Enriching data..."):
            enrich_new_companies(new_websites, business_type, city)

        #  Feature engineering step (Only for the current run)
        with st.spinner(" Applying Feature Engineering.."):
            feature_engineering()

        # Load model and rank leads
        model = load_model()
        if model:
            with st.spinner(" Ranking new leads..."):
                rank_new_leads()  # Only ranks the new leads

        #  Mark data as fetched
        st.session_state["data_fetched"] = True

    #  Display ranked leads and map for the current run
    if "current_run_websites" in st.session_state and st.session_state["current_run_websites"]:
        df = fetch_current_run_leads()  # Fetch ranked leads for the current run
        
        if not df.empty:
            display_leads_table(df)  # Show leads in table format
            display_leads_map(df)  # Show leads on map

#  Run Streamlit App
if __name__ == "__main__":
    main()









