from flask import Flask, request, jsonify
from chat_flow import trigger_flow

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def handle_query():
    """Handles incoming queries from the frontend."""
    data = request.get_json()
    print(data)
    user_query = data.get('query')
    llm_response, images = trigger_flow(user_query)
    return jsonify({"response": llm_response, "images":images}),200

if __name__ == '__main__':
    # This is for running locally during development
    # In a production environment, you would use a production-ready WSGI server
    # app.run(debug=True)
     app.run(host="0.0.0.0", port=5000, debug=True)
