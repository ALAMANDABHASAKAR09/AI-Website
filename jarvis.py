import pyttsx3
import speech_recognition as sr
import requests
import json
import time
from datetime import datetime
import threading
import pygame
import pytube
import os
import cv2  # Import OpenCV
import subprocess  # Import subprocess for opening applications
import google.generativeai as genai
pygame.mixer.init()
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import uuid
import webbrowser

# Configure the Gemini model
genai.configure(api_key="AIzaSyCk24LgF1T9VW4fqltn8rubr2wYFFmJpEk")
instruction2llm = '''
"Please respond only in JSON format. When I provide text for analysis, respond with the appropriate module, its parameters, the count of those parameters, a success flag, and a brief description. If you cannot understand the input, respond with 'Sorry, I can't understand.' Here are additional modules to be integrated into our talking robot program, along with examples of inputs and their expected outputs:

Modules:
1. 'Weather' - Get weather updates with 'City'.
2. 'Reminder' - Set reminders with 'Time', 'Message'.
3. 'Translator' - Translate text with 'Source Language', 'Target Language'.
4. 'News' - Fetch news with 'Category', 'Region'.
5. 'Jokes' - Tell jokes with 'Category', 'Language'.
6. 'CurrencyConverter' - Convert currency with 'From Currency', 'To Currency'.
7. 'SportsScores' - Provide sports scores with 'Sport Type', 'League'.
8. 'MusicPlayer' - Play music with 'Song Title', 'Artist'.
9. 'Navigation' - Provide directions with 'Start Location', 'End Location'.
10. 'Timer' - Set a timer with 'Duration'.
11. 'CookingRecipes' - Suggest recipes with 'Dish Name', 'Cuisine'.
12. 'FitnessTracker' - Track fitness activities with 'Activity Type', 'Duration'.
13. 'Quiz' - Conduct quizzes with 'Topic', 'Difficulty Level'.
14. 'HealthAdvice' - Give health advice with 'Symptom', 'Age Group'.
15. 'AppointmentScheduler' - Schedule appointments with 'Date', 'Time', 'Purpose'.
16. 'LearningAssistant' - Provide educational content with 'Subject', 'Grade Level'.
17. 'VoiceControl' - Control devices with 'Device Name', 'Action'.
18. 'EmergencyAlert' - Send emergency alerts with 'Type of Emergency', 'Location'.
19. 'TravelPlanner' - Plan trips with 'Destination', 'Departure Date', 'Budget'.
20. 'PetCare' - Provide pet care advice with 'Pet Type', 'Issue'.
21. 'EventReminder' - Remind about events with 'Event Name', 'Event Date'.
22. 'LanguageLearning' - Assist in language learning with 'Language', 'Proficiency Level'.
23. 'MoodTracker' - Track mood with 'Current Mood', 'Time of Day'.
24. 'ProductFinder' - Help find products with 'Product Name', 'Category'.
25. 'Horoscope' - Provide horoscope with 'Zodiac Sign'.
26. 'BookRecommendations' - Suggest books with 'Genre', 'Author'.
27. 'MovieSuggestions' - Recommend movies with 'Genre', 'Age Group'.
28. 'HomeworkHelper' - Assist with homework with 'Subject', 'Complexity Level'.
29. 'GardeningTips' - Offer gardening advice with 'Plant Type', 'Season'.
30. 'ArtificialIntelligence' - Explain AI concepts with 'Concept', 'Difficulty Level'.
31. 'PublicTransportInfo' - Provide transport information with 'Transport Mode', 'Route'.
32. 'StockMarketUpdates' - Give stock updates with 'Stock Symbol', 'Market'.
33. 'PersonalFinance' - Offer finance advice with 'Finance Topic', 'Experience Level'.
34. 'Time' - Provide the current time.
35. 'TakePicture' - Take a picture with the webcam.
36. 'VoiceControl' - Control devices with 'Device Name', 'Action'.

Examples:
1. Input: "What is the weather in New York?"
   Output:
   json
   {
     "module": "Weather",
     "parameters": ["New York"],
     "parameter_count": 1,
     "success": true,
     "description": "Providing weather updates for New York."
   }

2. Input: "Set a reminder for a meeting at 3 PM."
   Output:
   json
   {
     "module": "Reminder",
     "parameters": ["3 PM", "meeting"],
     "parameter_count": 2,
     "success": true,
     "description": "Reminder set for a meeting at 3 PM."
   }

3. Input: "Translate 'Hello' from English to Spanish."
   Output:
   json
   {
     "module": "Translator",
     "parameters": ["English", "Spanish", "Hello"],
     "parameter_count": 3,
     "success": true,
     "description": "Translating 'Hello' from English to Spanish."
   }

4. Input: "Tell me a joke about animals."
   Output:
   json
   {
     "module": "Jokes",
     "parameters": ["animals"],
     "parameter_count": 1,
     "success": true,
     "description": "Telling a joke about animals."
   }

5. Input: "How do I invest in Apple stocks?"
   Output:
   json
   {
     "module": "StockMarketUpdates",
     "parameters": ["Apple"],
     "parameter_count": 1,
     "success": true,
     "description": "Providing investment information for Apple stocks."
   }

6. Input: "What is the current time?"
   Output:
   json
   {
     "module": "Time",
     "parameters": [],
     "parameter_count": 0,
     "success": true,
     "description": "Providing the current time."
   }

7. Input: "Open Notepad"
   Output:
   json
   {
     "module": "VoiceControl",
     "parameters": ["Notepad", "open"],
     "parameter_count": 2,
     "success": true,
     "description": "Opening Notepad."
   }

8. Input: "I don't know what to do."
   Output:
   json
   {
     "response": "Sorry, I can't understand.",
     "success": false,
     "description": "The input could not be understood."
   }

'''

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
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
    model_name="gemini-1.5-flash-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Change the voice index (0 for male, 1 for female)
engine.setProperty('voice', voices[1].id)

# Function to recognize speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("Please say something...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed (e.g., 5 seconds)
        except sr.WaitTimeoutError:
            print("Timeout. No speech detected.")
            return ""

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return ""
    except sr.RequestError:
        print("Sorry, the speech recognition service is unavailable.")
        return ""

# Function to play audio from a YouTube URL
time_up_message_path = r"C:\Users\HP\Downloads\twirling-intime-lenovo-k8-note-alarm-tone-41440.mp3"

# Function to take a picture
def take_picture():
        cap = cv2.VideoCapture(0)

        # Capture a frame
        ret, frame = cap.read()

        # If the frame is captured successfully
        if ret:
            # Generate a unique filename
            filename = f"captured_image_{uuid.uuid4()}.jpg"
            # Save the image
            cv2.imwrite(filename, frame)
            cap.release()
            return {"success": True, "filename": filename}
        else:
            cap.release()
            return {"success": False, "error": "Failed to capture image"}

    # Capture an image
response = take_picture()

      
# Function to handle API requests based on the module

def handle_api_request(json_response):
    print(f"from gemini {json_response}")
    module = json_response.get("module")
    parameters = json_response.get("parameters")
    if module == "Weather":
        city = parameters[0]
        api_key = "84190506d5bf0843188d4a9531d7117c"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return {"error": response.get("message", "Error fetching weather data.")}

        weather = response["weather"][0]["description"]
        temperature = response["main"]["temp"]
        humidity = response["main"]["humidity"]
        wind_speed = response["wind"]["speed"]

        weather_info = (f"Current weather in {city}: {weather}. "
                        f"Temperature: {temperature}°C, "
                        f"Humidity: {humidity}%, "
                        f"Wind Speed: {wind_speed} m/s.")

        return {"weather_info": weather_info}
    elif module == "CurrencyConverter":
        # Implement currency converter API functionality here
        response = "Implement currency converter API functionality here"
        return response
    elif module == "StockMarketUpdates":
        stock_symbol = parameters[0]
        api_key = "S8GRXZIRC5TAC1IY"
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval=5min&apikey={api_key}"
        response = requests.get(url).json()

        if "Time Series (5min)" not in response:
            return {"error": "Error fetching stock data."}

        time_series = response["Time Series (5min)"]
        latest_timestamp = next(iter(time_series))
        latest_data = time_series[latest_timestamp]

        stock_info = (f"Stock update for {stock_symbol}: "
                      f"Price: {latest_data['1. open']}, "
                      f"High: {latest_data['2. high']}, "
                      f"Low: {latest_data['3. low']}, "
                      f"Close: {latest_data['4. close']}, "
                      f"Volume: {latest_data['5. volume']}.")

        return {"stock_info": stock_info}
    elif module == "Jokes":
        # Fetch a joke from JokeAPI
        url = "https://v2.jokeapi.dev/joke/Any"
        response = requests.get(url).json()
        if response.get("error"):
            return {"error": "Error fetching joke."}

        if response["type"] == "single":
            joke = response["joke"]
        else:
            joke = f"{response['setup']} ... {response['delivery']}"

        return {"joke": joke}
    elif module == "Time":
        current_time = datetime.now().strftime("%H:%M:%S")
        return {"time": current_time}
    elif module == "Timer":
        duration = parameters[0]
        try:
            duration_parts = duration.split()
            value = int(duration_parts[0])
            unit = duration_parts[1].lower() if len(duration_parts) > 1 else "seconds"
            if unit.startswith("minute"):
                seconds = value * 60
            elif unit.startswith("second"):
                seconds = value
            else:
                return {"error": "Invalid duration unit."}

            # Start a timer to play the time-up message
            threading.Timer(seconds, play_time_up_message).start()

            return {"timer": seconds}
        except (IndexError, ValueError):
            return {"error": "Invalid duration format."}
    elif module == "TakePicture":
        engine.say("Sure, I'll take a picture now, Keep Smile.")
        engine.runAndWait()
        webbrowser.open('AIMERS\VAmodels\Q&A\templates\qa.html',new=2)
        response = take_picture()
        if response.get("success"):
            engine.say("Image captured successfully.")
            engine.runAndWait()
            # Ask a question related to the image
            engine.say("Now, please ask a question about the image.")
            engine.runAndWait()
            question = recognize_speech()  # Use the speech recognition function
            if question:
                # Define the processor for question-answering
                processor = BlipProcessor.from_pretrained("C:/Code/AIMERS/VAmodels/")
                # Define the model for question-answering
                model = BlipForQuestionAnswering.from_pretrained("C:/Code/AIMERS/VAmodels/")
                
                # Process the captured image with the question
                inputs = processor(response['filename'], question, return_tensors="pt")
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True)

                # Provide the answer
                engine.say("The answer to your question is:")
                engine.say(answer)
                engine.runAndWait()
            else:
                engine.say("No question detected. Exiting.")
                engine.runAndWait()
        else:
            engine.say("Failed to capture image. Exiting.")
            engine.runAndWait()  
            return response

    elif module == "VoiceControl":
        device_name = parameters[0]
        action = parameters[1]
        if action.lower() == "open":
            if device_name.lower() == "calculator":
                subprocess.Popen("calc.exe")
                return "Opening Calculator."
            elif device_name.lower() == "chrome":
                subprocess.Popen("chrome.exe")
                return "Opening Browser."
            elif device_name.lower() == "notepad":
                subprocess.Popen("notepad.exe")
                return "Opening Notepad."
            response = {"module": "VoiceControl", "parameters": parameters, "parameter_count": len(parameters),"success": True, "description": f"Opening {device_name}."}

        elif action.lower() == "close":
            if device_name.lower() == "calculator":
                subprocess.Popen("taskkill /IM calc.exe /F")
                return "Closing Calculator"
            elif device_name.lower() == "chrome":
                subprocess.Popen("taskkill /IM chrome.exe /F")
                return "Closing Chrome"
            elif device_name.lower() == "notepad":
                subprocess.Popen("taskkill /IM notepad.exe /F")
                return "Closing Notepad"
            response = {"module": "VoiceControl", "parameters": parameters, "parameter_count": len(parameters),"success": True, "description": f"Closing {device_name}."}

    elif module=="terminate":
        subprocess.run(['python', 'app.py'],  check=True)


    # Add other API handling code for different modules here

    else:
        return "Sorry, I can't understand."

def play_time_up_message():
    try:
        # Play the time-up message audio
        pygame.mixer.music.load(time_up_message_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print("Error:", e)

def main():
    extracted_json_data = None  # Initialize the variable

    # Add initial greeting
    engine.say("Myself Buuujji from Kaalki 2898AD.")
    engine.runAndWait()

    while True:
        # Get speech input from user
        user_input = recognize_speech()

        # Check for "thank you" input
        if "thank you" in user_input.lower():
            engine.say("You're welcome! If you have any more questions or need further assistance, feel free to ask. Have a great day!")
            engine.runAndWait()
            continue
        if "how are you" in user_input.lower():
            engine.say("I'm doing well, thank you! How can I assist you today?")
            engine.runAndWait()
            continue
        if "good morning" in user_input.lower():
            engine.say("Good morning! What's up? How can I help you today?")
            engine.runAndWait()
            continue
        if "good afternoon" in user_input.lower():
            engine.say("Good afternoon! Had your lunch yet? What's on your mind today?")
            engine.runAndWait()
            continue
        if "good evening" in user_input.lower():
            engine.say("good evening! What's up bujji")
            engine.runAndWait()
            continue
        if "terminate" in user_input.lower():
            subprocess.run(['python', 'app.py'],  check=True)
            exit()


        if user_input:
            # Start a chat session with initial history
            chat_session = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [user_input],
                    }
                ]
            )

            # Send a message to the chat session and get the response
            response = chat_session.send_message(instruction2llm + user_input)
            llmresponse = response.text
            print(llmresponse)

            # Extract the JSON string from the response
            start_index = llmresponse.find('{')
            end_index = llmresponse.rfind('}')
            extracted_json_string = llmresponse[start_index:end_index + 1] if start_index != -1 and end_index != -1 else None

            # Parse the extracted JSON string
            if extracted_json_string:
                extracted_json_data = json.loads(extracted_json_string)
                print(extracted_json_data)

                # Handle API request and convert the response to speech
                try:
                    extracted_json_string = extracted_json_string.replace("'", "\"")
                    json_response = json.loads(extracted_json_string)
                    api_response = handle_api_request(json_response)
                    print(api_response)

                    if isinstance(api_response, dict):
                        if "weather_info" in api_response:
                            engine.say(api_response["weather_info"])
                        elif "stock_info" in api_response:
                            engine.say(api_response["stock_info"])
                        elif "joke" in api_response:
                            engine.say(api_response["joke"])
                        elif "time" in api_response:
                            engine.say(f"The current time is {api_response['time']}")
                        elif "timer" in api_response:
                            seconds = api_response["timer"]
                            minutes = seconds // 60
                            if minutes > 0:
                                engine.say(f"Timer set for {minutes} minutes.")
                            else:
                                engine.say(f"Timer set for {seconds} seconds.")
                            engine.runAndWait()
                            # Start a timer for the time-up message
                            threading.Timer(seconds, play_time_up_message).start()
                        elif "picture_path" in api_response:
                            engine.say(api_response["message"])
                            engine.say("Picture saved as succesfully.")
                        elif "description" in api_response:
                            engine.say(api_response["description"])
                    else:
                        engine.say(response.text)
                    engine.runAndWait()
                except json.JSONDecodeError:
                    print("Error decoding the JSON response.")
            else:
                print("No valid input was recognized. Waiting for speech input...")

if __name__ == "__main__":
    main()
