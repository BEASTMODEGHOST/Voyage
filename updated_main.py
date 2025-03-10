import os
from flask import Flask, request, jsonify, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from collections import defaultdict
from groq import Groq
from google.cloud import texttospeech
import google.generativeai as genai
import tempfile
from pydub import AudioSegment
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from rapidfuzz import process as fuzz_process
from google.cloud import translate_v2 as translate
import re

app = Flask(__name__)


GOOGLE_API_KEY = "AIzaSyDGgx6hG0kXex56TlL9z5UfLDdpxjaDirk"
model = genai.GenerativeModel(model_name="gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key="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")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"translation.json"
translate_client = translate.Client()

contributions = {
    "Mohan": 0,
    "Ayush": 0,
    "Siddharth": 0,
    "Ravi": 0
}
dues = defaultdict(lambda: defaultdict(float))  
categories = defaultdict(float) 
chat_history = []

def plot_contributions():

    plt.figure(figsize=(8, 5))
    plt.bar(contributions.keys(), contributions.values(), color='skyblue')
    plt.xlabel("Participants")
    plt.ylabel("Contribution (INR)")
    plt.title("Individual Contributions")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return base64_image

def plot_due_chart():
    users = list(contributions.keys())
    num_users = len(users)
    bar_width = 0.2
    index = range(num_users)

    plt.figure(figsize=(10, 6))
    for i, user in enumerate(users):
        values = [dues[user][other] for other in users]
        positions = [x + (i * bar_width) for x in index]
        plt.bar(positions, values, width=bar_width, label=f"Dues from {user}")

    plt.xlabel("Users")
    plt.ylabel("Amount Due (INR)")
    plt.title("Due Amounts Between Users")
    plt.xticks([r + bar_width for r in range(num_users)], users)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return base64_image


def calculate_total_expenditure():
    
    return sum(categories.values())
def plot_categories():
    plt.figure(figsize=(8, 5))
    plt.pie(
        categories.values(),
        labels=categories.keys(),
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Expense Categories")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return base64_image

def clean_text(text):
    text = re.sub(r"&#39;", "'", text) 
    text = re.sub(r"[^\w\s.,'-]", "", text)  
    return text.strip()

def translate_to_english(text):
    detection = translate_client.detect_language(text)
    detected_language = detection['language']

    if detected_language != "en": 
        print(f"Detected {detected_language}, translating to English...")
        translation = translate_client.translate(text, target_language="en")
        return clean_text(translation['translatedText']), detected_language

    print("Detected English, no translation needed.")
    return clean_text(text), detected_language

def transcribe_audio(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio = AudioSegment.from_file(audio_file)
            audio.export(temp_audio_file.name, format="wav")
            with open(temp_audio_file.name, "rb") as audio:
                transcription = groq_client.audio.transcriptions.create(model="whisper-large-v3", file=audio)
            return transcription.text
    except Exception as e:
        raise RuntimeError(f"Error during audio transcription: {str(e)}")

def get_closest_name(name, valid_names, threshold=5):

    result = fuzz_process.extractOne(name, valid_names)
    if result is not None:
        match, score = result[0], result[1]
        if score >= threshold:
            return match
    return None

def process_speech(audio_file):
    global dues, contributions, categories
    try:
        text = transcribe_audio(audio_file)
        translated_text, detected_language = translate_to_english(text)
        chat_history.append(translated_text)
        print(f"Translated Text: {translated_text} (Detected Language: {detected_language})")

        
        prompt = (
            f"Extract the following information from the text and return as JSON:\n"
            f"For example in the command 'Add 3000 rupees for ayush contribution for trnasportation of ayush, siddharth and jay'.\n"
            f"- payer: The name of the person who made the payment. from the example it is Ayush\n"
            f"- payees: A list of names of persons for whom the payment was made. from the example they are 'Ayush','Siddharth','Jay'\n"
            f"- amount: The total amount in numeric format.\n"
            f"- category: The category of the expense (e.g., Food, Transportation, etc.).\n"
            f"- split_equally: A boolean indicating if the amount is to be split equally.\n"
            f"Text: '{translated_text}'\n\n"
            f"Output JSON:"
        )
        llm_response = llm.invoke(prompt)
        response_content = llm_response.content.strip()

    
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(response_content)

        payer = get_closest_name(parsed_response.get("payer"), contributions.keys())
        if not payer:
            return f"Error: Payer '{parsed_response.get('payer')}' not recognized."

        payees = [
            get_closest_name(payee, contributions.keys())
            for payee in parsed_response.get("payees", [])
        ]
        invalid_payees = [parsed_response.get("payees", [])[i] for i, payee in enumerate(payees) if payee is None]
        if invalid_payees:
            return f"Error: Invalid payees: {', '.join(invalid_payees)}"

        amount = parsed_response.get("amount", 0)
        category = parsed_response.get("category", "Others")
        split_equally = parsed_response.get("split_equally", False)

    
        categories[category] += amount

        if split_equally:
            individual_due = amount / len(payees)
            for payee in payees:
                if payee and payee != payer:  
                    dues[payee][payer] += individual_due  


    
        contributions[payer] += amount

        print(f"Dues Updated: {dict(dues)}")
        print(f"Contributions Updated: {dict(contributions)}")
        return f"{payer} paid {amount} for {', '.join(payees)} in {category}."
    except Exception as e:
        print(f"Error processing speech: {str(e)}")
        return f"Error processing speech: {str(e)}"


def search_chat_history(query):
    
    try:
        
        prompt = (
            f"From the following chat history, find and return entries that match the query:\n"
            f"Chat History:\n{json.dumps(chat_history, indent=2)}\n\n"
            f"Query: '{query}'\n\n"
            f"Output: A JSON array of relevant chat entries."
        )

        llm_response = llm.invoke(prompt)
        response_content = llm_response.content.strip()

       
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()

        try:
            matched_entries = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response Content: {response_content}")
            return []

        matched_entries = [str(entry) for entry in matched_entries]

        return matched_entries

    except Exception as e:
        print(f"Error in search_chat_history: {str(e)}")
        return []

@app.route("/")
def index():
    contribution_chart = plot_contributions()
    category_chart = plot_categories()
    due_chart = plot_due_chart()
    total_expenditure = calculate_total_expenditure()
    return render_template(
        "indexvoy.html",
        contribution_chart=contribution_chart,
        category_chart=category_chart,
        due_chart=due_chart,
        chat_history=chat_history,
        total_expenditure=total_expenditure,
    )


@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    response_message = process_speech(audio_file)

    contribution_chart = plot_contributions()
    category_chart = plot_categories()
    due_chart = plot_due_chart()
    total_expenditure = calculate_total_expenditure()

    return jsonify(
        {
            "message": response_message,
            "contribution_chart": contribution_chart,
            "category_chart": category_chart,
            "due_chart": due_chart,
            "chat_history": chat_history,
            "total_expenditure": total_expenditure,
        }
    )
@app.route("/search", methods=["POST"])
def search():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    try:
        query = transcribe_audio(request.files["audio"])
        translated_query, _ = translate_to_english(query)
        results = search_chat_history(translated_query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    

if __name__ == '__main__':
    app.run(debug=True)
