import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

def process_image_and_question(image, question):
    # Load the BLIP processor and model from the specified local directory
    processor = BlipProcessor.from_pretrained("C:/Code/AIMERS/VAmodels/")
    model = BlipForQuestionAnswering.from_pretrained("C:/Code/AIMERS/VAmodels/")

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("C:/Code/AIMERS/VAmodels/")
model = BlipForQuestionAnswering.from_pretrained("C:/Code/AIMERS/VAmodels/")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))























    # Process the image and question
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)

    return answer

def main():
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    cap.release()
    cv2.destroyAllWindows()

    # Convert the captured frame to a PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Specify the question
    question = "how many dogs are in the picture?"

    # Process the image and question
    answer = process_image_and_question(image, question)

    # Print the answer
    print("Answer:", answer)

if __name__ == "__main__":
    main()
