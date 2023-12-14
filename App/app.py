from symbol import tfpdef
from flask import Flask
from flask import render_template, request, jsonify
app=Flask(__name__)

from werkzeug.utils import secure_filename

import os
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from PIL import Image
#from keras.applications.vgg19 import VGG19
#from keras.preprocessing import image
#from keras.applications.vgg19 import preprocess_input
#from keras.models import Model
import numpy as np
#from tensorflow.keras.utils import load_img
import scipy.io.wavfile as wavfile
import numpy
import os.path
from os import walk
from scipy import stats
import numpy as np
import librosa 
import numpy as np
from scipy.stats import norm
import pickle
import base64
import matplotlib.pyplot as plt
# Import the libraries
import matplotlib.pyplot as plt
from sklearn import svm
#from keras.preprocessing.image import img_to_array
#from keras.applications.vgg19 import preprocess_input

UPLOAD_FOLDER = os.getcwd()  
ALLOWED_EXTENSIONS = {'wav', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    print("no existing")

@app.route("/")
def home():
    return "Hello, Flask!"
    
@app.route('/upload')
def upload():
   return render_template("form.html")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 
           

	
@app.route('/uploader', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            signal, rate = librosa.load(file_path)
            S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=2048, hop_length=512, n_mels=128)
            S_DB = librosa.power_to_db(S, ref=np.max)
            S_DB = S_DB.flatten()[:1200]
            clf = pickle.load(open('SVM.pkl', 'rb'))
            ans = clf.predict([S_DB])[0]
            music_class = str(ans)

            # Encode the file to base64
            with open(file_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

            result = {
                'music_class': music_class,
                'base64_audio': encoded_audio
            }

            os.remove(file_path)  # Remove the uploaded file after processing

            return jsonify(result)



'''@app.route('/uploadervgg', methods = ['GET','POST'])
def classify_vgg():
    if request.method == 'GET':
           return "hello"
    if request.method == 'POST':
         file = request.files['file']
         if file.filename == '':
              return 'No file selected'
         if file and allowed_file(file.filename):
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                   
          vgg19 = tfpdef.keras.models.load_model("vgg19.h5")
          image = load_img(UPLOAD_FOLDER+'/'+filename, target_size=(224, 224, 3))
          np.expand_dims(image, axis=0)
          image = img_to_array(image)
          image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

          image = preprocess_input(image)
          yhat = model.predict(image)
    # create a list containing the class labels
          class_labels = ["blues", "classical", "country", "disco", "hiphop", "metal", "pop", "reggae", "rock"]
    # find the index of the class with maximum score
          pred = np.argmax(class_labels, axis=-1)
    # print the label of the class with maximum score
          return class_labels[pred]'''
          
          
    
               
      
		
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  
   
