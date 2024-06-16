from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from newsapi import NewsApiClient
import pyttsx3
import subprocess
from PIL import Image, ImageDraw, ImageFont
import io
from diffusers import StableDiffusionInpaintPipeline
import torch
import base64
import openai
from flask import send_file
import requests
import signal
import time
import platform
import threading


# Flag to control the running state
running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("Jarvis is stopping...")

# Attach the signal handler to handle termination signals
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

openai.api_key='YOUR-OPENAI-KEY'


import google.generativeai as genai

API_KEY = "YOUR-GEMINI-KEY"

genai.configure(api_key="YOUR-GEMINI-KEY")
newsapi = NewsApiClient(api_key='YOUR-NEWS-KEY')

engine=pyttsx3.init()

jarvis_process = None


# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  safety_settings=safety_settings,
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "HELLO\n\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Hello! How can I help you today? \n",
      ],
    },
  ]
)
app = Flask(__name__)

# Ensure the 'static/images' directory exists
if not os.path.exists('static/images'):
    os.makedirs('static/images')

# Global variable to store the captured image path
captured_image_path = None

def capture_image():
    global captured_image_path
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the height
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        image_path = f'static/images/captured_image_{timestamp}.jpg'
        cv2.imwrite(image_path, frame)
        captured_image_path = image_path
        engine.say("Please ask any question ..?")
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path, question):
    image = Image.open(image_path).convert('RGB')
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    engine.say(answer)
    return answer

@app.route('/')
def index():
    return render_template('index.html')

def get_sources_and_domains():
    all_sources = newsapi.get_sources()['sources']
    sources = []
    domains = []
    for e in all_sources:
        id = e['id']
        domain = e['url'].replace("http://", "")
        domain = domain.replace("https://", "")
        domain = domain.replace("www.", "")
        slash = domain.find('/')
        if slash != -1:
            domain = domain[:slash]
        sources.append(id)
        domains.append(domain)
    sources = ", ".join(sources)
    domains = ", ".join(domains)
    return sources, domains

@app.route('/news', methods=['GET', 'POST'])
def news():
    if request.method == 'POST':
        sources, domains = get_sources_and_domains()
        keyword = request.form["keyword"]
        related_news = newsapi.get_everything(q=keyword,
                                      sources=sources,
                                      domains=domains,
                                      language='en',
                                      sort_by='relevancy')
        no_of_articles = related_news['totalResults']
        if no_of_articles > 100:
            no_of_articles = 100
        all_articles = newsapi.get_everything(q=keyword,
                                      sources=sources,
                                      domains=domains,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size = no_of_articles)['articles']
        return render_template("news.html", all_articles = all_articles, 
                               keyword=keyword)
    else:
        top_headlines = newsapi.get_top_headlines(country="in", language="en")
        total_results = top_headlines['totalResults']
        if total_results > 100:
            total_results = 100
        all_headlines = newsapi.get_top_headlines(country="in",
                                                     language="en", 
                                                     page_size=total_results)['articles']
        return render_template("news.html", all_headlines = all_headlines)
    return render_template("home.html")

@app.route('/qa', methods=['GET', 'POST'])
def qa():
    global captured_image_path
    if request.method == 'POST':
        # Handle form submission and get the question and image data
        image_data = request.json['image_data']
        question = request.json['question']

        # Process the question and image data to get the answer
        if image_data:
            capture_image()
            answer = process_image(captured_image_path, question)
            return jsonify({'answer': answer})
        else:
            return jsonify({'error': 'Failed to capture image.'})
    else:
        return render_template('qa.html')

@app.route('/speech', methods=['POST'])
def speech():
    if request.method == 'POST':
        user_input = request.json.get('input')
        ai_response = chat_session.send_message(user_input)
        engine.say(ai_response.text)
        engine.runAndWait()
        return jsonify({'response': ai_response.text})
    else:
        return redirect(url_for('index'))

@app.route('/gaming', methods=['GET'])
def gaming():
    # Create or update the HTML template for the gaming page
    return render_template('TicTacToe.html')

@app.route('/scramble', methods=['GET'])
def scramble():
    # Create or update the HTML template for the gaming page
    return render_template('scramble.html')


@app.route('/exit')
def exit():
    global captured_image_path
    captured_image_path = None
    return redirect(url_for('index'))

@app.route('/response', methods=['POST'])
def response():
    user_input = request.json.get('input')
    ai_response = response= chat_session.send_message(user_input)
    engine.say(ai_response.text)
    # engine.runAndWait()
    return jsonify({'response': ai_response.text})


@app.route('/bot')
def bot():
    return render_template('bot.html')


@app.route('/generate-image', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json"
        )

        image_data = response['data'][0]['b64_json']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_jarvis_script():
    global jarvis_process
    jarvis_process = subprocess.Popen(['python', 'jarvis.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@app.route('/run_jarvis', methods=['POST'])
def run_jarvis():
    global jarvis_process
    if jarvis_process is None:
        if platform.system() == "Windows":
            jarvis_process = subprocess.Popen(['python', 'jarvis.py'], creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif platform.system() == "Darwin":  # macOS
            jarvis_process = subprocess.Popen(['osascript', '-e', 'tell application "Terminal" to do script "python3 jarvis.py"'], shell=False)
        elif platform.system() == "Linux":
            jarvis_process = subprocess.Popen(['gnome-terminal', '--', 'python3', 'jarvis.py'])
        else:
            return jsonify({'status': 'Unsupported OS'})
        
        return jsonify({'status': 'Jarvis started'})
    else:
        return jsonify({'status': 'Jarvis is already running'})

@app.route('/terminate_jarvis', methods=['POST'])
def terminate_jarvis():
    global jarvis_process
    if jarvis_process is not None:
        if platform.system() == "Windows":
            jarvis_process.terminate()
        else:
            os.kill(jarvis_process.pid, signal.SIGTERM)
        jarvis_process = None
        return jsonify({'status': 'Jarvis terminated'})
    else:
        return jsonify({'status': 'Jarvis is not running'})

if __name__ == '__main__':
    app.run(debug=True)
    