import requests
import numpy as np

url_cloud_arthur_0 = 'https://contrails-2pojbkqtxa-ew.a.run.app'

def api_call_predict(X) :
    X_bytes = X.tobytes()
    result = requests.post(url_cloud_arthur_0 + "/upload_image/", files= {"img" : X_bytes} )
    print(f"status_code is : {result.status_code}")
    X_mask_pred = np.frombuffer(result.content, dtype = np.float32).reshape(256,256,1)
    return X_mask_pred
