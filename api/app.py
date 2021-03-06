from flask import Flask, jsonify, request, send_from_directory
import pickle
import os
import numpy as np
from questions import generate_multiple_choice
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

SWAGGER_URL = ''
API_URL = '/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Question generation app"
    },
)


@app.route("/swagger.json")
def specs():
  #  print('he')
    return send_from_directory(os.getcwd(), "swagger.json")

    
    
@app.route('/api', methods=['POST'])
def get():
    # use parser and find the user's query
    data = request.get_json(force=True)
    if not ('text' in data):
        return {'error': 'text is required'}
    if not ('count' in data):
        return {'error': 'count is required'}

    output = generate_multiple_choice(data['text'], data['count'])
    #output = data

    return jsonify(output)

app.register_blueprint(swaggerui_blueprint)

app.run()


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)