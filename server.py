from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin
import pickle
app = Flask(__name__)
CORS(app, support_credentials=True)
from model_run import Export_DTC, Export_RFC

model_dtc = Export_DTC()
model_rfc = Export_RFC()
               
@app.route('/', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def hello_world():
    if request.method == 'POST':
        try:
            
            json_data_array = request.get_json()
            
            df = pd.DataFrame(json_data_array)
            ans1 = model_rfc.predict(df)
            ans2 = model_dtc.predict(df)
            print(ans1,ans2)
            return jsonify(ans1.tolist())
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    elif request.method == 'GET':
        return 'Hello, World!'


if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)