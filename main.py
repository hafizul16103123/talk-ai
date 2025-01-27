import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pyttsx3
import speech_recognition as sr

# Initialize the app
app = Flask(__name__)

# Load local LLM model and tokenizer
MODEL_NAME = "gpt2"  # Replace with your desired open-source model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create a pipeline for text generation
conversation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Initialize TTS engine
tts_engine = pyttsx3.init()

def text_to_speech(text):
    """Convert text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

# Initialize speech recognizer
recognizer = sr.Recognizer()

def speech_to_text(audio_file):
    """Convert speech to text from an audio file."""
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        return recognizer.recognize_google(audio)

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle user messages and provide AI responses."""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "Please provide a message."}), 400

    try:
        # Generate response using LLM
        inputs = tokenizer.encode(user_message, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Optionally convert response to speech
        if request.json.get('speak', False):
            text_to_speech(bot_reply)

        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/speech_to_text', methods=['POST'])
def handle_speech_to_text():
    """Endpoint to convert user audio to text."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    audio_file = request.files['audio']
    try:
        text = speech_to_text(audio_file)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ielts_tips', methods=['GET'])
def ielts_tips():
    """Provide IELTS preparation tips."""
    tips = [
        "Practice speaking on common IELTS topics like technology, health, and education.",
        "Record yourself to analyze your pronunciation and fluency.",
        "Expand your vocabulary by reading newspapers, blogs, and articles.",
        "Familiarize yourself with IELTS speaking test formats.",
        "Focus on giving structured answers with a clear introduction, body, and conclusion."
    ]
    return jsonify({"tips": tips})

if __name__ == '__main__':
    # Download the model locally if not already available
    if not os.path.exists(f"./{MODEL_NAME}"):
        print("Downloading model...")
        AutoModelForCausalLM.from_pretrained(MODEL_NAME).save_pretrained(f"./{MODEL_NAME}")
        AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(f"./{MODEL_NAME}")
    app.run(debug=True)
