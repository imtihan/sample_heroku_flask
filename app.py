# app.py
from flask import Flask, request, jsonify
app = Flask(__name__)

###### model loading #####
from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-fake-news-detection")

model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-finetuned-fake-news-detection")
######################


@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    text = request.args.get("text", None)

    # For debugging
    print(f"got name {text}")

    response = {}
    output = model(text)
    

    if not output:
        response["ERROR"] = "no model output"

    else:
        response["MESSAGE"] = f"{output}"

    # Return the response in json format
    return jsonify(response)



# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
