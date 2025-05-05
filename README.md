Group Members: Christine, Atmikha, Pranav
CSCI-B 351: Introduction to Artificial Intelligence


What to install to run our application. 
1. NOTE: Due to the size of one of the files in the virtual environment, we were not able to push it up to GitHub, so you first need to start a virtual environment before starting to download all of the dependencies
2. Start a venv like such: python3 -m venv .venv
3. Then run this: source .venv/bin/activate
4. Requirements.txt has all of the dependencies you need to run our app, so run this:
  pip install -r requirements.txt



We are using the framework Flask for our web application. 

How our app works:
1. You run app.py, which runs a Flask function called create_app() from __init__.py
2. __init__.py has views.py registered as a blueprint with prefix '/', meaning any function ran in views.py that returns a html file will be directed with '/'
3. In views.py is where we have all of our model code. Here is where we have the classify_image() method that uses our model to classify the image uploaded in the home screen.
4. When a image is uploaded in the home screen and the 'analyze' button is clicked, it runs a POST method that runs code in the home() method in views.py
5. In this method, the picture uploaded is put into the /static/pictures folder. From this folder, it uses the classify_image() method to classify the uploaded image, returning the category of the image and the confidence level of the image (percentage).
6. The home() method then sends this information to the frontend html file using Django, which the html file can use to output the correct data. 
