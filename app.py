from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
from bone_age_assessment_model.test import test
import os

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def index():
    return 'bone-age-assessment-server'

@app.route("/api/v1/predict/", methods=['POST'])
def predict():
    image_data_url = request.json['dataUrl']
    predicted_bone_age = test(image_data_url)
    return jsonify({'predicted_bone_age': predicted_bone_age})

if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug = True, host = '0.0.0.0', port=port) # 배포 시 이 부분 주석 해제