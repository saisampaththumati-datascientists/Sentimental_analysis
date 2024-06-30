# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import json

# app = Flask(__name__)
# CORS(app)

# # Load your model
# model = tf.keras.models.load_model('my_rnn_model.h5')

# # Load the tokenizer
# with open('tokenizer.json') as f:
#     data = json.load(f)
#     tokenizer = tokenizer_from_json(data)

# # Preprocessing function
# def preprocess_text(text):
#     sequences = tokenizer.texts_to_sequences([text])
#     padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen based on your training
#     return padded_sequences

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         text = data['text']
#         input_data = preprocess_text(text)
#         prediction = model.predict(input_data)
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf

# app = Flask(__name__)
# CORS(app)

# # Load your trained model
# model = tf.keras.models.load_model('my_rnn_model.h5')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         text = data['text']
#         input_data = tf.constant([text])  # Pass raw text directly to the model
#         prediction = model.predict(input_data)
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import pandas as pd

# app = Flask(__name__)
# CORS(app)
# def load_model():
#     global model
#     model = tf.keras.models.load_model('my_rnn_model.h5')

# # Initialize the model when the application starts
# load_model()

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         print(data)
#         text = data['text']
#         input_series = pd.Series([text])
#         input_data =input_series
#         print(type(input_data))
#         prediction = model.predict(input_data)
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         return jsonify({'error': '{}'.format(e)}), 500
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# Load your trained model
model = tf.keras.models.load_model('my_rnn_model.h5')

# Load your dataset (assuming you want to preprocess new inputs similar to training data)
df = pd.read_csv("/Users/saisampath/Sentimental_analysis/train (1).csv")
X = df["text"]
Y = df["target"]

# Initialize the TextVectorization layer
vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=100)
vectorizer.adapt(X)

# Define the model (you can reuse the same model architecture)
embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, mask_zero=True)
model = tf.keras.Sequential([
    vectorizer,
    embedding,
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model (you can choose to fit it again or keep the loaded weights)
model.fit(X, Y, epochs=5, batch_size=32)

@app.route('/predict', methods=['POST'])

def predict():
    try:
        data = request.json
        text = data['text']

        # Preprocess the input text
        input_series = pd.Series([text])
        # input_data = vectorizer(input_series)
        # input_data = tf.expand_dims(input_data, axis=-1)  # Add a dimension to match expected input shape
        
        # Perform prediction
        prediction = model.predict(input_series)

        # Assuming the prediction is a probability, convert it to a class label if needed
        # Example: predicted_class = 1 if prediction > 0.5 else 0
        
        return jsonify({'prediction': prediction.tolist()})
    
    except KeyError as e:
        return jsonify({'error': 'KeyError: {}'.format(str(e))}), 400
    
    except Exception as e:
        return jsonify({'error': 'Prediction failed: {}'.format(str(e))}), 500
if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# import pandas as pd

# app = Flask(__name__)
# CORS(app)

# # Define the path where TextVectorization layer was saved
# vectorizer_load_path = '/Users/saisampath/Sentimental_analysis/vectorizer_config'

# # Load the TextVectorization layer from saved model
# loaded_vectorizer = tf.saved_model.load(vectorizer_load_path)

# # Load your trained model
# model = tf.keras.models.load_model('my_rnn_model.h5')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         text = data['text']

#         # Preprocess the input text
#         input_series = pd.Series([text])
#         input_data = loaded_vectorizer(input_series)['input_text']
#         print(input_data)
#         # Perform prediction
#         prediction = model.predict(input_data)
#         print(predict)
#         return jsonify({'prediction': prediction.tolist()})
    
#     except KeyError as e:
#         return jsonify({'error': 'KeyError: {}'.format(str(e))}), 400
    
#     except Exception as e:
#         return jsonify({'error': 'Prediction failed: {}'.format(str(e))}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

