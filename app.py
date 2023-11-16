from flask import Flask
app = Flask(__name__)

@app.route('/')
def welcome():
    return " Welcome to project file"

if __name__ =='__main__':
    app.run(debug=True)