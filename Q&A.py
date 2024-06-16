import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    image_path = 'captured_image.jpg'
    cv2.imwrite(image_path, frame)
    cap.release()
    cv2.destroyAllWindows()
    return image_path

def process_image(image_path, question):
    image = Image.open(image_path).convert('RGB')
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    image_path = capture_image()
    question = input("Enter your question about the image: ")
    answer = process_image(image_path, question)
    print("Answer:", answer)
