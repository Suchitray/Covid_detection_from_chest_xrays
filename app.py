import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image
import torchvision
import torchvision.transforms as transforms
import PIL
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('models/model_resnet.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5001/')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    show_img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    imgs = []
    imgg = Image.open(img_path)
    imgg = imgg.convert('RGB')
    transform = transforms.Compose (
          [transforms.Resize((224, 224)),  
           transforms.ToTensor(),
           transforms.Normalize (mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
           )
    imgg = transform (imgg)
    imgs.append(imgg)
    imgs = np.reshape (np.stack(imgs, axis = 0), (1,224,224,3))
    preds = model.predict(imgs)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Normal','Pneumonia','Covid']
        a = preds[0]
        ind=np.argmax(a)
        print((np.max(a))*100)
        print('Prediction:', disease_class[ind])
        result=(str)(round(((np.max(a))*100),2)) + '% ' + disease_class[ind]
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)

    # Serve the app with gevent
    #http_server = WSGIServer(('', 5001), app)
    #http_server.serve_forever()
    #app.run()
