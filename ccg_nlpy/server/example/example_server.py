from ccg_nlpy.server.example.example_annotator import ExampleAnnotator
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# necessary for testing on localhost
CORS(app)


def main():
    # create your model object here, see the DummyModel class for a minimal example.
    annotator = ExampleAnnotator(model=mymodel, provided_view="DUMMYVIEW", required_views=["TOKENS"])

    # Expose wrapper.annotate method through a Flask server
    app.add_url_rule(rule='/annotate', endpoint='annotate', view_func=annotator.annotate, methods=['GET'])
    app.run(host='localhost', port=5000)
    # On running this main(), you should be able to visit the following URL and see a json text annotation returned
    # http://127.0.0.1:5000/annotate?text="Stephen Mayhew is a person's name"&views=DUMMYVIEW


if __name__ == "__main__":
    main()
