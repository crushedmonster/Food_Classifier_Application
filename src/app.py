####################
# Required Modules #
####################

# Generic / Built-in 
import os

# Libraries 
import logging
from waitress import serve
from flask import Flask, render_template, jsonify, flash, \
    request, url_for, redirect
from werkzeug.utils import secure_filename

# Custom 
## import inference.py from src 
from .inference import Inference

##################
# Configurations #
##################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create the application
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

DEV = False # <--- TOGGLE THIS

if DEV:
    model_name = '../model.h5'
    app.config["UPLOAD_FOLDER"] = 'static/uploads/'
    read_me_path = '../README.md'
    
else:
    model_name = 'model.h5'
    app.config["UPLOAD_FOLDER"] = 'src/static/uploads/'
    read_me_path = 'README.md'

# instantiate the inference class
inference = Inference(model_name)
logger.info(f'Model has been successfully loaded.')

#############
# Functions #
#############

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """ Displays the index page accessible at '/'
    """
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    """ Displays the about page accessible at '/'
    """
    return render_template('about.html')

@app.route('/info', methods=['GET'])
def short_description():
    """Returns information about the model and what input it expects in JSON
    """
    return jsonify({
        'model': 'MobileNetv2',
        'input-size': '256x256x3',
        'num-classes': 12,
        'pretrained-on': 'ImageNet'
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # make prediction
            input_arr = inference.load_img(image_path)
            pred_class, pred_proba = inference.prediction(input_arr)

            return render_template('index.html', food = pred_class, \
                probability_pct = round(pred_proba *100,2), filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

###########
# Scripts #
###########

if __name__ == "__main__":
    logger.info("Session Started")
    if DEV:
        app.run(host="0.0.0.0", debug=True, port=8000)
    else:
        # For production mode, comment the line above and uncomment below
        serve(app, host="0.0.0.0", port=8000)
    logger.info("Session Ended")
