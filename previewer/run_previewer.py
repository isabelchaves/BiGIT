from flask import Flask, request, render_template

from src.configs.models_const import ModelsConsts

app = Flask(__name__, template_folder='./templates')


@app.route('/', methods=['GET'])
def main():
    return render_template('main.html',
                           last_query='',
                           last_model=ModelsConsts.TF_IDF)


@app.route('/', methods=['POST'])
def get_response():
    query = str(request.form['query'])
    return render_template('main.html',
                           last_query=query,
                           result='I will list the results here')
