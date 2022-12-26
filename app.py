from flask import Flask, jsonify, request
from classifier import get_prediction
app = Flask(__name__)

@app.route('Predict-Digit', methods = ['POST'])
def predict_data():
  Image = request.files.get('Digit')
  prediction  = get_prediction(Image)
  return jsonify({
    'prediction': prediction
  }), 200
  
if __name__ == '__main__':
    app.run(debug = True)