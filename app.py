from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from pipeline.predict_pipeline import CustomData,PredictPipeline



app = Flask(__name__)

## route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            ram = float(request.form.get('ram')),
            battery_power = float(request.form.get('battery_power')),
            px_width= float(request.form.get('px_width')),
            px_height= float(request.form.get('px_height')),
            mobile_wt = float(request.form.get('mobile_wt')),
            int_memory= float(request.form.get('int_memory')),
         )
        pred_data = data.get_data_as_data_frame()
        print(pred_data)

        Predict_pipeline=PredictPipeline()
        result = Predict_pipeline.predict(pred_data)

        if result[0] == 0:
            result = 'Low cost'
        elif result[0] == 1:
            result = 'Medium cost'
        elif result[0] == 2:
            result = 'High cost'
        elif result[0] == 3:
            result = 'Expensive'

        return render_template('home.html',result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000,debug=True)

    

