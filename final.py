import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import hashlib
from datetime import datetime
import logging
import firebase_admin
from firebase_admin import credentials, firestore
import json
import httpx

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="TalentScout Assistant Enhanced",
    page_icon="✨",
    layout="wide"
)

# --- Firebase Setup ---
def init_database():
    """Initialize Firebase Firestore client."""
    try:
        # Load Firebase service account key from environment variable
        firebase_key = os.getenv("FIREBASE_SERVICE_ACCOUNT")
        if not firebase_key:
            raise ValueError("FIREBASE_SERVICE_ACCOUNT is not set in environment variables")
        
        # Parse the JSON string
        cred_dict = json.loads(firebase_key)
        cred = credentials.Certificate(cred_dict)
        
        # Check if Firebase app is already initialized
        try:
            firebase_admin.get_app(name="[DEFAULT]")
            logger.info("Firebase app already initialized.")
        except ValueError:
            # Initialize Firebase app if not already initialized
            firebase_admin.initialize_app(cred, name="[DEFAULT]")
            logger.info("Firebase app initialized successfully.")
        
        db = firestore.client()
        logger.info("Firebase Firestore client created successfully.")
        return db
    except Exception as e:
        logger.error(f"Error initializing Firebase: {e}")
        st.error(f"Failed to initialize Firebase: {e}. Candidate data will not be saved.", icon="❌")
        return None

# Initialize database on startup
db = init_database()

def hash_sensitive_info(data):
    """Hash sensitive information like email and phone number."""
    return hashlib.sha256(data.encode()).hexdigest()

def save_candidate_data(candidate_info=None, answers=None, candidate_id=None):
    """Save candidate information and/or their answers to Firebase Firestore."""
    if not db:
        logger.error("Firebase not initialized. Cannot save candidate data.")
        st.error("Failed to save candidate data: Firebase not initialized.", icon="❌")
        return None
    
    try:
        # Save candidate information if provided
        if candidate_info:
            candidate_data = {
                "full_name": candidate_info.get("full_name", ""),
                "email_hash": hash_sensitive_info(candidate_info.get("email", "")) if candidate_info.get("email") else "",
                "phone_hash": hash_sensitive_info(candidate_info.get("phone_number", "")) if candidate_info.get("phone_number") else "",
                "years_experience": candidate_info.get("years_experience", ""),
                "desired_position": candidate_info.get("desired_position", ""),
                "current_location": candidate_info.get("current_location", ""),
                "tech_stack": candidate_info.get("tech_stack", ""),
                "created_at": datetime.now().isoformat()
            }
            # Add candidate to Firestore and get document ID
            candidate_ref = db.collection("candidates").add(candidate_data)[1]
            candidate_id = candidate_ref.id
            logger.info(f"Saved candidate data for ID {candidate_id} to Firebase.")
        
        # Save answers if provided and candidate_id is available
        if answers and candidate_id:
            for question, answer in answers:
                answer_data = {
                    "candidate_id": candidate_id,
                    "question": question,
                    "answer": answer,
                    "created_at": datetime.now().isoformat()
                }
                db.collection("candidate_answers").add(answer_data)
            logger.info(f"Saved {len(answers)} answers for candidate ID {candidate_id} to Firebase.")
        
        return candidate_id
    except Exception as e:
        logger.error(f"Error saving candidate data to Firebase: {e}")
        st.error(f"Failed to save candidate data to Firebase: {e}", icon="❌")
        return None

# --- NLTK Setup (for Sentiment Analysis) ---
# Set custom NLTK data path to a writable directory
nltk_data_path = "/tmp/nltk_data"
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

_sentiment_analyzer = None
try:
    # Check if VADER lexicon exists in custom path
    lexicon_file = nltk.data.find('sentiment/vader_lexicon.zip')
    logger.info(f"VADER lexicon found at {lexicon_file}")
    _sentiment_analyzer = SentimentIntensityAnalyzer()
    logger.info("Sentiment Analyzer initialized.")
except LookupError:
    logger.info("VADER lexicon not found. Attempting download to /tmp/nltk_data...")
    try:
        # Download VADER lexicon to custom path
        nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
        _sentiment_analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER lexicon downloaded and Sentiment Analyzer initialized.")
    except Exception as download_e:
        logger.warning(f"Failed to download VADER lexicon to {nltk_data_path}: {download_e}")
        st.warning(f"Sentiment analysis disabled due to lexicon download failure: {download_e}", icon="⚠️")
except Exception as e:
    logger.warning(f"Error during NLTK setup: {e}")
    st.warning(f"Sentiment analysis disabled due to NLTK setup error: {e}", icon="⚠️")

# --- Configuration and Setup ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# --- Constants ---
ASSISTANT_PERSONA = """
You are 'TalentScout Assistant', a friendly, professional, and efficient AI hiring assistant for TalentScout, a tech recruitment agency.
Your primary goal is to conduct an initial screening of candidates.
Your tasks are:
1. Greet the candidate and explain your purpose clearly.
2. Collect the following information sequentially: Full Name, Email Address, Phone Number, Years of Experience, Desired Position(s), Current Location, and Tech Stack (programming languages, frameworks, databases, tools). Ask one question at a time.
3. Validate Email Address: Ensure it follows a valid format (e.g., user@domain.com). If invalid, politely ask again until a valid email is provided.
4. Validate Phone Number: Ensure it is exactly 10 digits (e.g., 1234567890). If invalid, politely ask again until a valid phone number is provided.
5. Based *only* on the provided Tech Stack, generate 3-5 relevant technical screening questions suitable for assessing basic to intermediate proficiency. Present all questions at once as a numbered list in a single response.
6. Collect all answers to the technical questions in a single response from the candidate, expecting answers to be numbered corresponding to the questions.
7. Maintain context and manage the conversation flow smoothly. Be concise.
8. Handle unexpected inputs gracefully by gently guiding the user back to the current question or process step. Stick strictly to the screening tasks. Do not answer general knowledge questions, engage in off-topic chats, or provide recruitment advice beyond the scope of this initial screening.
9. Conclude the conversation politely, explaining that their profile and answers will be reviewed by a human recruiter.
10. If the user indicates they want to end the conversation (e.g., "bye", "exit", "quit", "thanks that's all"), acknowledge and end the conversation gracefully.
"""

CONVERSATION_STAGES = [
    "greeting", "get_name", "get_email", "get_phone", "get_experience",
    "get_position", "get_location", "get_tech_stack", "generate_questions",
    "collect_answers", "finalize", "end"
]
EXIT_KEYWORDS = ["bye", "exit", "quit", "goodbye", "thanks that's all", "thank you", "done", "stop"]

# --- LLM Client Initialization ---
client = None
llm_ready_flag = False
MODEL_NAME = "llama3-8b-8192"
try:
    # Configure client with custom timeout and retries
    client = Groq(
        api_key=API_KEY,
        timeout=httpx.Timeout(30.0, connect=5.0, read=20.0, write=5.0),
        max_retries=3
    )
    # Test API key with a simple request
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test connection"}],
        max_tokens=10
    )
    llm_ready_flag = True
    logger.info("Groq client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Groq client: {e}")
    st.error(f"Failed to initialize Groq client: {e}. Please check your GROQ_API_KEY in the .env file or Hugging Face secrets.", icon="❌")
    llm_ready_flag = False

# --- Helper Functions ---
def get_llm_response(prompt, context_messages):
    """Gets a response from the LLM based on the prompt and context."""
    if not llm_ready_flag or not client:
        return "LLM client not initialized."
    messages = [{"role": "system", "content": ASSISTANT_PERSONA}] + context_messages + [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.5,
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in get_llm_response: {e}")
        st.error(f"Failed to communicate with Groq API: {e}. Please try again later.", icon="📡")
        return "Sorry, I encountered an issue processing your request via Groq. Please try again later."

def generate_technical_questions(tech_stack, context_messages):
    """Generates technical questions based on the tech stack using Groq."""
    if not llm_ready_flag or not client:
        return "LLM client not initialized."
    question_prompt = f"""
    Based *only* on the following tech stack provided by the candidate: "{tech_stack}".
    Generate exactly 3 to 5 relevant technical screening questions suitable for assessing basic to intermediate proficiency in these specific technologies.
    Present the questions clearly as a numbered list in a single paragraph-like response.
    Do *not* ask the candidate to answer them now. Just generate the list of questions.
    Example format:
    Okay, based on your tech stack, here are a few technical questions:
    1. [Question 1 related to stack]
    2. [Question 2 related to stack]
    3. [Question 3 related to stack]
    """
    messages = [{"role": "system", "content": ASSISTANT_PERSONA}] + context_messages + [{"role": "user", "content": question_prompt}]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.6,
            max_tokens=350
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in generate_technical_questions: {e}")
        st.error(f"Failed to generate technical questions via Groq: {e}", icon="❔")
        return "Sorry, I couldn't generate the technical questions at this moment."

def parse_answers(questions, answer_text):
    """Parse a single response into individual answers, assuming numbered format."""
    answers = []
    # Split answer text by lines and look for numbered entries
    lines = answer_text.strip().split("\n")
    current_answer = []
    current_number = 1
    
    for line in lines:
        line = line.strip()
        if line and re.match(r'^\d+\.\s*', line):
            if current_answer:  # Save previous answer
                answers.append(" ".join(current_answer).strip())
                current_answer = []
            current_answer.append(line.lstrip("12345.").strip())
            current_number += 1
        elif line:
            current_answer.append(line)
    
    # Append the last answer
    if current_answer:
        answers.append(" ".join(current_answer).strip())
    
    # Pair questions with answers, handling mismatched lengths
    result = []
    for i in range(min(len(questions), len(answers))):
        question = questions[i].lstrip("12345.").strip()
        answer = answers[i]
        result.append((question, answer))
    
    return result

def is_exit_command(text):
    """Checks if the user input indicates a desire to exit."""
    return any(keyword in text.lower().strip() for keyword in EXIT_KEYWORDS)

def get_sentiment_emoji(text):
    """Analyzes text sentiment and returns an emoji representation."""
    if _sentiment_analyzer is None:
        return ""
    score = _sentiment_analyzer.polarity_scores(text)['compound']
    if score >= 0.1:
        return "😊"  # Positive
    elif score <= -0.1:
        return "😟"  # Negative
    else:
        return "😐"  # Neutral

# --- Custom CSS ---
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 15px;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 85%;
    }
    [data-testid="chatAvatarIcon-assistant"] + div .stChatMessage {
        background-color: #f0f2f6;
        color: #333;
        border: 1px solid #e0e0e0;
        margin-right: auto;
        margin-left: 0;
    }
    [data-testid="chatAvatarIcon-user"] + div .stChatMessage {
        background-color: #dcf8c6;
        color: #333;
        border: 1px solid #c5e5a4;
        margin-left: auto;
        margin-right: 0;
    }
    .stChatMessage + .stChatMessage {
        margin-top: 10px;
    }
    .sentiment-emoji {
        font-size: 0.8em;
        margin-left: 5px;
        opacity: 0.7;
        display: inline-block;
    }
    .stSidebar [data-testid="stExpander"] details {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #fafafa;
    }
    .stSidebar [data-testid="stExpander"] summary {
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- Streamlit App UI ---
st.title("✨ TalentScout Hiring Assistant")
st.caption("Enhanced with Sentiment Analysis, Firebase Storage & UI Improvements | Powered by Groq ⚡")

# --- Sidebar for Collected Information ---
with st.sidebar:
    st.header("Screening Progress")
    with st.expander("View Collected Information", expanded=True):
        if not st.session_state.get("candidate_info", {}):
            st.write("No information collected yet.")
        else:
            info_map = {
                "full_name": "👤 Name", "email": "✉️ Email", "phone_number": "📞 Phone",
                "years_experience": "⏳ Experience", "desired_position": "🎯 Position(s)",
                "current_location": "📍 Location", "tech_stack": "💻 Tech Stack"
            }
            for key, value in st.session_state.candidate_info.items():
                display_key = info_map.get(key, key.replace("_", " ").title())
                st.markdown(f"**{display_key}:** {value}")

    st.info("Chatbot is using the Llama3 8B model via Groq for fast responses.", icon="💡")
    if not llm_ready_flag:
        st.error("Groq LLM Client is not available. Check API Key.", icon="🚨")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stage" not in st.session_state:
    st.session_state.stage = "greeting"
if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {}
if "technical_questions" not in st.session_state:
    st.session_state.technical_questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "candidate_id" not in st.session_state:
    st.session_state.candidate_id = None

# --- Display Chat History ---
# Deduplicate messages while preserving order
seen_messages = set()
unique_messages = []
for message in st.session_state.messages:
    message_tuple = (message["role"], message["content"])
    if message_tuple not in seen_messages:
        seen_messages.add(message_tuple)
        unique_messages.append(message)

for message in unique_messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if message["role"] == "user" and _sentiment_analyzer:
            sentiment_emoji = get_sentiment_emoji(content)
            st.markdown(f"{content} <span class='sentiment-emoji'>{sentiment_emoji}</span>", unsafe_allow_html=True)
        else:
            st.markdown(content)

# --- Initial Greeting or Next Step Prompt ---
if st.session_state.stage == "greeting" and not st.session_state.messages:
    initial_prompt = "Greet the candidate and briefly explain your purpose (initial screening for TalentScout tech roles)."
    if llm_ready_flag:
        assistant_response = get_llm_response(initial_prompt, [])
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.stage = "get_name"
        time.sleep(0.5)
        first_question = "To begin, could you please share your full name?"
        st.session_state.messages.append({"role": "assistant", "content": first_question})
        st.rerun()

# --- Handle Finalize Stage Separately ---
if st.session_state.stage == "finalize":
    closing_prompt = "The candidate has provided answers to all technical questions. Provide concluding remarks. Thank the candidate for their time and answers. Explain that their profile and responses will be reviewed by a TalentScout recruiter who will contact them if there's a suitable match. Wish them good luck in their job search."
    assistant_response = get_llm_response(closing_prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.session_state.stage = "end"
    st.rerun()

# --- Handle User Input ---
input_disabled = not llm_ready_flag or st.session_state.stage == "end"
input_placeholder = "Chat ended." if st.session_state.stage == "end" else "Your response..."

if prompt := st.chat_input(input_placeholder, disabled=input_disabled):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check for exit command
    if is_exit_command(prompt):
        exit_confirmation_prompt = "The user indicated they want to end the conversation (e.g., said 'bye' or 'quit'). Acknowledge this politely and briefly confirm the chat is ending."
        exit_confirmation = get_llm_response(exit_confirmation_prompt, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": exit_confirmation})
        # Save candidate data only if not already saved
        if st.session_state.candidate_info and not st.session_state.candidate_id:
            st.session_state.candidate_id = save_candidate_data(st.session_state.candidate_info, st.session_state.answers)
        st.session_state.stage = "end"
        st.rerun()

    assistant_response = None
    if st.session_state.stage != "end":
        current_stage = st.session_state.stage
        user_input = prompt

        # --- State Machine Logic ---
        if current_stage == "get_name":
            st.session_state.candidate_info["full_name"] = user_input
            next_prompt = f"I've received the name '{user_input}'. Now ask for their email address."
            assistant_response = get_llm_response(next_prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.session_state.stage = "get_email"

        elif current_stage == "get_email":
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(email_pattern, user_input):
                st.session_state.candidate_info["email"] = user_input
                next_prompt = "Email received. Now ask for their phone number."
                assistant_response = get_llm_response(next_prompt, st.session_state.messages)
                st.session_state.stage = "get_phone"
            else:
                retry_prompt = f"The user provided '{user_input}' which doesn't seem like a valid email address. Politely ask them again for a correct email address, explaining it should be in the format 'user@domain.com'."
                assistant_response = get_llm_response(retry_prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        elif current_stage == "get_phone":
            cleaned_phone = re.sub(r'[\s\-\(\)+]', '', user_input)
            if re.match(r'^\d{10}$', cleaned_phone):
                st.session_state.candidate_info["phone_number"] = user_input
                next_prompt = "Phone number received. Now ask for their total years of professional technical experience."
                assistant_response = get_llm_response(next_prompt, st.session_state.messages)
                st.session_state.stage = "get_experience"
            else:
                retry_prompt = f"The user provided '{user_input}' which doesn't seem like a valid 10-digit phone number. Politely ask them again for a correct phone number, explaining it should be exactly 10 digits (e.g., 1234567890)."
                assistant_response = get_llm_response(retry_prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        elif current_stage == "get_experience":
            st.session_state.candidate_info["years_experience"] = user_input
            next_prompt = "Experience noted. Now ask what specific technical position(s) they are interested in (like Software Engineer, Data Scientist, etc.)."
            assistant_response = get_llm_response(next_prompt, st.session_state.messages)
            st.session_state.stage = "get_position"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        elif current_stage == "get_position":
            st.session_state.candidate_info["desired_position"] = user_input
            next_prompt = "Desired position noted. Now ask for their current location (City, Country)."
            assistant_response = get_llm_response(next_prompt, st.session_state.messages)
            st.session_state.stage = "get_location"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        elif current_stage == "get_location":
            st.session_state.candidate_info["current_location"] = user_input
            next_prompt = "Location noted. Now ask them to list their primary technical skills and technologies (e.g., Python, React, Node.js, AWS, SQL, Docker)."
            assistant_response = get_llm_response(next_prompt, st.session_state.messages)
            st.session_state.stage = "get_tech_stack"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        elif current_stage == "get_tech_stack":
            st.session_state.candidate_info["tech_stack"] = user_input
            tech_stack_confirmation = "Thanks for sharing your tech stack! I will now generate technical questions based on these skills for you to answer."
            st.session_state.messages.append({"role": "assistant", "content": tech_stack_confirmation})

            questions_response = generate_technical_questions(user_input, st.session_state.messages)
            # Parse questions into a list for answer pairing
            questions = [q.strip() for q in questions_response.split("\n") if q.strip() and q[0].isdigit()]
            st.session_state.technical_questions = questions
            
            # Save candidate data and store candidate_id
            st.session_state.candidate_id = save_candidate_data(st.session_state.candidate_info, [])
            
            if st.session_state.technical_questions:
                assistant_response = f"Based on your tech stack, here are the technical questions:\n\n{questions_response}\n\nPlease provide your answers to all questions in a single response, numbering each answer to match the question numbers."
                st.session_state.stage = "collect_answers"
            else:
                assistant_response = "I couldn't generate questions at this time. Let's conclude the screening."
                st.session_state.stage = "finalize"
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        elif current_stage == "collect_answers":
            # Parse the single response into individual answers
            questions = st.session_state.technical_questions
            st.session_state.answers = parse_answers(questions, user_input)
            
            # Save answers using existing candidate_id
            if st.session_state.candidate_id and st.session_state.answers:
                save_candidate_data(answers=st.session_state.answers, candidate_id=st.session_state.candidate_id)
            
            # Move to finalize stage to handle concluding remarks separately
            st.session_state.stage = "finalize"
            assistant_response = None

        else:
            fallback_prompt = f"The user provided: '{user_input}'. My current goal is '{current_stage}'. Respond politely, acknowledge if necessary, but gently guide them back to providing the information needed for '{current_stage}'. Avoid answering off-topic questions. If unsure, ask them to clarify or repeat the expected question for '{current_stage}'."
            assistant_response = get_llm_response(fallback_prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        st.rerun()

if st.session_state.stage == "end" and 'prompt' not in locals():
    st.chat_input("Chat ended.", disabled=True)
