# app.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from flask import Flask, request, jsonify
app = Flask(__name__)

###### model loading #####
loaded_model = None
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)
    
vectorizer = None
with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)
######################


@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    text = request.args.get("text", None)

    # For debugging
    print(f"got name {text}")

    response = {}
    
    
    prediction = loaded_model.predict(vectorizer.transform(['This is fake news']))[0]
    

    if not prediction:
        response["ERROR"] = "no model output"

    else:
        response["MESSAGE"] = f"{prediction}"

    # Return the response in json format
    return jsonify(response)



# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
