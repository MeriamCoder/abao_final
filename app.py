import pandas as pd
from joblib import load 
from flask import Flask, request, jsonify
from flask_cors import CORS
from category_encoders import BinaryEncoder

#import the model
model = load("decision_tree_model.joblib")

#import the dataset

x= pd.read_csv("accident.csv")

# fit the categorigal features to the encoder
categorical_features = ['Gender', 'Helmet_Used', 'Seatbelt_Used']
encoder= BinaryEncoder()
x_encoded = encoder.fit_transform(x[categorical_features])

api = Flask(__name__)
CORS(api)

@api.route('/api/prediction', methods=['POST'])
def predict_heart_failure():
    data = request.json['inputs']
    input_df = pd.DataFrame(data)
    
    input_encoded = encoder.transform(input_df[categorical_features])

    input_df = input_df.drop(categorical_features, axis=1)
    input_encoded = input_encoded.reset_index(drop=True)

    final_input = pd.concat([input_df, input_encoded], axis=1)

    prediction = model.predict_proba(final_input)
    class_labels = model.classes_
    response = []
    for prob in prediction:
        prob_dict = {}
        for k, v in zip(class_labels, prob):
            prob_dict[str(k)]= round(float(v) *100, 2)
        response.append(prob_dict)

    return jsonify({'Prediction': response})

if __name__ == "__main__":
    api.run(port=8080, debug=True, host="0.0.0.0")

