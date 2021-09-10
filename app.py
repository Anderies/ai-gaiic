from flask import Flask, request, jsonify
from flask_marshmallow import Marshmallow
import pickle
# import sklearn.external.joblib as extjoblib
# from sklearn.svm import SVR
import numpy as np
import os
import json
import sklearn

# Init app
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


# # Init ma
ma = Marshmallow(app)

# Product Schema
class ResponseSchema(ma.Schema):
    class Meta:
        fields = ('farm_meter', 'rain', 'sunny', 'cloudy')


# Load Model for prediction
# model_iris = load_model('model.h5') 
filename = "svr_crop.pt"
model = pickle.load(open(filename, 'rb'))

print('The scikit-learn version is {}.'.format(sklearn.__version__))
# Predict
@app.route('/model', methods=['POST'])
def predict_flower():

    pred = {"success": False}

    farm_meter  = request.json['farm_meter']
    rain  = request.json['rain']
    sunny  = request.json['sunny']
    cloudy  = request.json['cloudy']
    crop_type = request.json['crop_type']
    # CATEGORIES = ['setosa','versicolor','virginica']
    input = [[farm_meter,rain,sunny,cloudy]]
    # print([petal_length,petal_width,sepal_length,sepal_width])

    print("dam =============",input)
    output = model.predict(input)
    # class_name = CATEGORIES[np.argmax(output[0])]

    print(output,"===========")
    pred= {
        "success": True,
        "data_sended" : {
            "farm_meter": farm_meter,
            "rain": rain,
            "sunny": sunny,
            "cloudy": cloudy,
        },
        "hasil_prediksi" : {
            "crop_type": crop_type,
            "crop_yield_result": output[0]
        }
    }

    return json.dumps(str(pred))
    # return jsonify(pred)

# Run Server
if __name__ == '__main__':
    app.run(debug=True)