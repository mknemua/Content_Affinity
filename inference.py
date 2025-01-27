import joblib
import os
import json
import pandas as pd
import  numpy as np


def model(model_dir):
    model=joblib.load(os.path.join(model_dir,'iris_model.joblib'))
    return model

def input_fn(request_body,request_content_type):
    if request_content_type=='application/json':
        input_data=json.loads(request_body)
        data=np.array(input_data['data']).reshape(-1,1)
        return data
    raise ValueError(f'Unsupported content type:{request_content_type}')

def predict_fn(input_data,model):
    return model.predict(input_data)

def output_fn(prediction,content_type):
    return json.dumps(prediction.tolist())
                      