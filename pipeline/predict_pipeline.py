import sys
import pandas as pd
import pickle
import os



class PredictPipeline:
    def __init__(self):
        pass
        
    def predict(self,features):
        model_path=os.path.join("artifacts","model.pkl")
        preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
        print("Before Loading")
        model=pickle.load(open(model_path, 'rb'))
        preprocessor=pickle.load(open(preprocessor_path, 'rb'))
        print("After Loading")
        data_scaled=preprocessor.transform(features)
        preds=model.predict(data_scaled)
        return preds



class CustomData:
    def __init__(self,ram:int,battery_power:int,px_width:int,px_height:int,mobile_wt:int,int_memory:int):
        self.ram = ram
        self.battery_power = battery_power
        self.px_width= px_width
        self.px_height= px_height
        self.mobile_wt = mobile_wt
        self.int_memory= int_memory

    def get_data_as_data_frame(self):
        
        custom_data_input_dict = {
                "ram": [self.ram],
                "battery_power": [self.battery_power],
                "px_width": [self.px_width],
                "px_height": [self.px_height],
                "mobile_wt": [self.mobile_wt],
                "int_memory": [self.int_memory]
                
            }

        return pd.DataFrame(custom_data_input_dict)

