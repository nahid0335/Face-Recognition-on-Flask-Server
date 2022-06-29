from copyreg import constructor
import os
import re
import io
import zlib
from werkzeug.utils import secure_filename
from flask import Response
from cs50 import SQL
from flask import Flask, flash, jsonify, redirect, render_template, request, session ,url_for
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime

#import basic library
from PIL import Image
from base64 import b64encode, b64decode
import re
import numpy as np

#import tensorflow and function
import tensorflow as tf
from layers import L1Dist

from helpers import apology, login_required



# Configure application
app = Flask(__name__)



# Configure the one-shot learning siamese model
model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist':L1Dist})



# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True



# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response



# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///data.db")



@app.route("/")
@login_required
def home():
    return redirect("/home")



@app.route("/home")
@login_required
def index():
    return render_template("index.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    # Forget any user_id
    session.clear()
    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        # Ensure username was submitted
        if not input_username:
            return render_template("login.html",messager = 1)

        # Ensure password was submitted
        elif not input_password:
             return render_template("login.html",messager = 2)

        # Query database for username
        username = db.execute("SELECT * FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username exists and password is correct
        if len(username) != 1 or not check_password_hash(username[0]["hash"], input_password):
            return render_template("login.html",messager = 3)

        # Remember which user has logged in
        session["user_id"] = username[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")



@app.route("/logout")
def logout():
    """Log user out"""
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")



@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not input_username:
            return render_template("register.html",messager = 1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("register.html",messager = 2)

        # Ensure passwsord confirmation was submitted
        elif not input_confirmation:
            return render_template("register.html",messager = 4)

        elif not input_password == input_confirmation:
            return render_template("register.html",messager = 3)

        # Query database for username
        username = db.execute("SELECT username FROM users WHERE username = :username",
                              username=input_username)

        # Ensure username is not already taken
        if len(username) == 1:
            return render_template("register.html",messager = 5)

        # Query database to insert new user
        else:
            new_user = db.execute("INSERT INTO users (username, hash) VALUES (:username, :password)",
                                  username=input_username,
                                  password=generate_password_hash(input_password, method="pbkdf2:sha256", salt_length=8),)

            if new_user:
                # Keep newly registered user logged in
                session["user_id"] = new_user

            # Flash info for the user
            flash(f"Registered as {input_username}")

            # Redirect user to homepage
            return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")




# Load image from file and convert to 100x100px
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img,channels=3)
    
    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0
    
    # Return image
    return img





@app.route("/facereg", methods=["GET", "POST"])
def facereg():
    session.clear()
    if request.method == "POST":

        encoded_image = (request.form.get("pic")+"==").encode('utf-8')
        username = request.form.get("name")
        name = db.execute("SELECT * FROM users WHERE username = :username",
                        username=username)
              
        if len(name) != 1:
            return render_template("camera.html",message = 1)

        id_ = name[0]['id']    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/unknown/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()


        # Specify thresholds
        detection_threshold = 0.50

        input_img = preprocess('./static/face/unknown/'+str(id_)+'.jpg')
        validation_img = preprocess('./static/face/'+str(id_)+'.jpg')
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        print(result)

        if result>detection_threshold:
            username = db.execute("SELECT * FROM users WHERE username = :username",
                              username="swa")
            session["user_id"] = username[0]["id"]
            return redirect("/")
        else:
            return render_template("camera.html",message=3)

    else:
        return render_template("camera.html")





@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":

        encoded_image = (request.form.get("pic")+"==").encode('utf-8')


        id_=db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]
        # id_ = db.execute("SELECT id FROM users WHERE id = :user_id", user_id=session["user_id"])[0]["id"]    
        compressed_data = zlib.compress(encoded_image, 9) 
        
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)
        
        new_image_handle = open('./static/face/'+str(id_)+'.jpg', 'wb')
        
        new_image_handle.write(decoded_data)
        new_image_handle.close()
        return redirect("/home")

    else:
        return render_template("face.html")



def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html",e = e)



# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)




if __name__ == '__main__':
      app.run()
