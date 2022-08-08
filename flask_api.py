## Flask-RESTful API
## Generate book content through GET request

from flask import Flask
from flask_restful import Api, Resource
import threading
import sys
import os
import datetime

currentDir = os.getcwd()
sys.path.insert(0, currentDir)

import combined_generator

app = Flask(__name__)
api = Api(app)

## Main
class book(Resource):
    def get(self, prompt):
        threading.Thread(target=combined_generator.generateBook(prompt)).start()
        timeReady = "Book finished created at: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"data": prompt, "timeReady": timeReady}

## Test resource
class hello(Resource):
    def get(self):
        timeNow = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {"data": "Hello! API up and running...", "time": timeNow}

api.add_resource(book, "/book/<string:prompt>")
api.add_resource(hello, "/hello")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
