from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
from model_run import Export_DTC, Export_RFC

model_dtc = Export_DTC()
model_rfc = Export_RFC()
               
@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        try:
            
            json_data_array = request.get_json()
            
            df = pd.DataFrame(json_data_array)
            ans1 = model_rfc.predict(df)
            ans2 = model_dtc.predict(df)
            print(ans1,ans2)
            return str(ans1)+","+str(ans2),200
            # return jsonify({'prediction': ans}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    elif request.method == 'GET':
        return 'Hello, World!'


if __name__ == '__main__':
   app.run(host='127.0.0.1', port=5000)