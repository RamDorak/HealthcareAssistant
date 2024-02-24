from flask import Flask, request, jsonify
from flask_cors import CORS
from prog.heartdisease import train_and_predict

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the frontend
    input_data = request.json['input_data']
    print(input_data)
    input_data = [int(x) for x in input_data]
    # # Call the heartdisease.py program and pass input data
    result = train_and_predict(input_data)
    return jsonify({'output': result})

if __name__ == '__main__':
    app.run(debug=True)
