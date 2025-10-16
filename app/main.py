from flask import Flask, request, jsonify
from torch_utils import transform_image, get_prediction
app = Flask(__name__)


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS
#rsplit splits from the right the first(1) hit it finds (one is for only once)
#square bracket chooses right part of the split



#in the context of a wel--designed RESTful API each major resource
#should have its own endpoint
@app.route('/predict', methods=['POST'])
def predict():
    #load image from request
    #turn image to tensor
    #make prediction using model
    #return prediction as json
    # Perform prediction using the model

    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        if not allowed_file(file.filename):
            return jsonify({"error": "format not supported"})
        try:
            errorline = 0
            img_bytes = file.read()
            errorline = 1
            tensor = transform_image(img_bytes)
            errorline = 2
            prediction = get_prediction(tensor)
            errorline = 3
            data = {"prediction":prediction, "class_name": str(prediction)}
            errorline = 4
            return jsonify(data)
        #can#t put tensor in the json but it only has one item anyways

        
        except:
            return jsonify({"error": str(errorline)})