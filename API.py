from flask import Flask, request, jsonify, json
import random
from main import retrival_qa

app = Flask(__name__)

json_file_path = 'contexts.json'

with open(json_file_path, 'r') as json_file:
    chatbot_data = json.load(json_file)

@app.route('/contexts',methods=['GET'])
def get_intents():
    return jsonify(chatbot_data['contexts'])

def chatbot_response(user_message):
    intents = chatbot_data['intents']
    
    response = {"message": ""}

    for intent in intents:
        # print(intent)
        for pattern in intent['patterns']:
            if user_message.lower() in pattern.lower():
                print('workds')
                response["message"] = random.choice(intent['responses'])
                return response
    response["message"] = "NA"
    return response

@app.route('/intents/chat', methods=['POST'])
def chat():    
    str = request.args.get('message')
    response = chatbot_response(str)
    return jsonify(response)
  
@app.route('/qa', methods=['POST'])
def qa_endpoint():
    try:
        if request.headers['Content-Type'] == 'application/json':
            data = request.get_json()
            crop = data['crop']
            question = data['question']
            
            # Call the retrieval-based QA function
            answer= retrival_qa(crop, question)
            
            return jsonify({"answer": answer})
        else:
            return jsonify({"error": "Invalid content type. Please use 'application/json'."}), 415
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3500)


