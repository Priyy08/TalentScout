# TalentScout
TalentScout Chatbot is a Streamlit-based web application that conducts initial screenings for tech candidates. It collects candidate information, validates inputs, generates technical questions based on the provided tech stack, collects answers in a single response, and stores data in an SQLite database. The app uses Groq's Llama3 model for question generation and NLTK for sentiment analysis, featuring a user-friendly interface with a chat system that terminates after answers are submitted.

# Features

Candidate Information Collection: Sequentially gathers full name, email, phone number, years of experience, desired position, current location, and tech stack (e.g., Python, React).
Input Validation: Ensures email follows a valid format (e.g., user@domain.com) and phone number is exactly 10 digits (e.g., 1234567890).
Technical Question Generation: Generates 3–5 technical questions based on the candidate's tech stack, presented as a numbered list using Groq's Llama3 model.
Single-Response Answers: Collects all answers in one numbered response (e.g., 1. [Answer 1] 2. [Answer 2]), parsed for storage.
SQLite Database: Stores candidate details (with hashed email and phone for privacy) and answers in candidates and candidate_answers tables.
Sentiment Analysis: Uses NLTK's VADER to analyze user responses and display sentiment emojis (😊 for positive, 😟 for negative, 😐 for neutral).
Chat Termination: Disables the chat input after answers are submitted or on exit commands (e.g., "bye", "exit", "quit").
UI Styling: Custom CSS for chat messages, sidebar progress tracking, and a clean layout.

Project Structure
talentscout-chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes .env, candidates.db, etc.
├── README.md           # Project documentation

Setup

Clone the Repository:
git clone https://github.com/your-username/talentscout-chatbot.git
cd talentscout-chatbot


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root with your Groq API key:
GROQ_API_KEY=your_groq_api_key_here


Run the App:
streamlit run app.py

Access the app at http://localhost:8501.


# Usage

Enter Candidate Details: Provide your full name, email, phone number, years of experience, desired position (e.g., Software Engineer), location, and tech stack.
Receive Technical Questions: The app generates 3–5 questions based on your tech stack, displayed as a numbered list.
Submit Answers: Enter all answers in a single response, numbering each to match the questions (e.g., 1. [Answer 1] 2. [Answer 2]).
Completion: The chat input is disabled after submitting answers or using an exit command. Data is saved to an SQLite database (candidates.db).

Database Schema

candidates:
id: INTEGER PRIMARY KEY AUTOINCREMENT
full_name: TEXT
email_hash: TEXT
phone_hash: TEXT
years_experience: TEXT
desired_position: TEXT
current_location: TEXT
tech_stack: TEXT
created_at: TIMESTAMP


candidate_answers:
id: INTEGER PRIMARY KEY AUTOINCREMENT
candidate_id: INTEGER (FOREIGN KEY references candidates(id))
question: TEXT
answer: TEXT
created_at: TIMESTAMP



# Dependencies

streamlit==1.39.0: Powers the web interface.
groq==0.11.0: Interfaces with Llama3 for question generation.
python-dotenv==1.0.1: Loads the Groq API key from .env.
nltk==3.9.1: Performs sentiment analysis on user responses.
sqlite3: Built-in Python library for database storage.

# Notes

The SQLite database (candidates.db) is created locally and excluded from the repository via .gitignore.
Email and phone numbers are hashed using SHA-256 for privacy.
The app expects numbered answers to match question numbers for accurate parsing.
