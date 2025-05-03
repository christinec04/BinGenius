from flask import Blueprint, flash, redirect, render_template, request, session, url_for, make_response, Flask, abort


#imports for data processing and training
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import os



views = Blueprint('views', __name__)



@views.route('/', methods=['GET', 'POST'], endpoint='home')
def home():

    picture = 'output.png'
    predClass = 'None'
    confidence = 0.0
    index = -1
    cssFilename = 'home.css'

    if request.method == 'POST': 

        file = request.files['image']
        filename = file.filename
        s1 = filename
        s2 = '/Users/pranavvangari/B351/FinalProject/Code/Static/Pictures/'
        s3 = "%s%s" % (s2, s1)

        file.save(s3)
        s4 = 'pictures/'

        picture = "%s%s" % (s4, s1)

        # Perform image classification
        predClass, confidence, index = image_classify(s3)
        round(confidence, 5)








    return render_template("home.html", picture=picture, predClass=predClass, confidence=confidence, index=index, cssFilename=cssFilename)


@views.route('/result', methods=['GET', 'POST'], endpoint='result')
def result():
    
    

    return render_template("result.html")



# code for model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes=6):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)  # Adapt to TrashNet
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def image_classify(path):
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    model = load_model('Code/models/mobilenetv2_trashnet2.pth', num_classes=len(class_names))

    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        # print("Predicted class index:", predicted.item())

        predClass = class_names[int(predicted.item())]
    
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)  # Shape: [1, num_classes]

        # Get the predicted class index and confidence
        # predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted].item() * 100

        return predClass, confidence, int(predicted.item())
    
# cla, con =image_classify('data/paper/paper1.jpg')

# print(f"Class: {cla}  Confidence: {con:.4f}")

