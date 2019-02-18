This folder contains the neccessary code for serving your python model through a visualization tool like [apelles](https://github.com/CogComp/apelles) which can consume text annotation jsons.

To make a demo using your fancy pytorch / tensorflow / dynet model, you need to 
- [Write your annotator](#create-your-annotator), 
- [Write a method to create new views in your model](#add-method-to-create-view-in-your-model), and 
- [Write the server](#write-the-server).

If you are trying to serve a multilingual model, you should also look at [serving multiple models](#serving-multiple-models).

## Create Your Annotator 
Create a Annotator by subclassing the Annotator class in `annotator.py`. 
This class wraps around your model, and specifies what view will be provided by your model and what views are required.

You need to implement the `add_view` method, that will internally call your model.
For example, the `ExampleAnnotator` in `example/example_annotator` implements a `add_view` method that calls the model to get a new view that is then added to the text annotation. 

```python
    def add_view(self, docta):
        # ask the model to create the new view
        new_view = self.model.get_view_from_text_annotation(docta)
        # add it to the text annotation
        new_view.view_name = self.provided_view
        docta.view_dictionary[self.provided_view] = new_view
        return docta
```

You also need to implement a `get_text_annotation_for_model` method that creates a text annotation (by calling either a local or remote pipeline) that contains all the neccessary views needed by your model (for instance, Wikifier needs NER view).  

```python
    def get_text_annotation_for_model(self, text: str, required_views: List[str]):
        text = text.replace("\n", "")
        pretokenized_text = [text.split(" ")]
        required_views = ",".join(required_views)
        ta_json = self.pipeline.call_server_pretokenized(pretokenized_text=pretokenized_text, views=required_views)
        ta = TextAnnotation(json_str=ta_json)
        return ta
```

## Add Method to Create View in your Model

Write a method similar to get_view_from_model in `example/example_model.py`. This method name could be anything, you are responsible for calling this in the `add_view` method above.

```python
    def get_view_from_model(self, docta:TextAnnotation) -> View:
        # This upcases each token. Test for TokenLabelView
        new_view = copy.deepcopy(docta.get_view("TOKENS"))
        tokens = docta.get_tokens
        for token, cons in zip(tokens, new_view.cons_list):
            cons["label"] = token.upper()
        return new_view
```

## Write the Server
Write a `server.py` similar to `example/example_server.py`. 
This is where you instantiate your model, wrap it into the annotator class you wrote, and expose its annotate method using flask server.


```python
mymodel = ExampleModel()
# this could have been a remote pipeline. 
pipeline = local_pipeline.LocalPipeline()
# specify the view that your annotator will provide, and the views that it will require. 
annotator = ExampleAnnotator(model=mymodel, pipeline=pipeline, provided_view="DUMMYVIEW", required_views=["TOKENS"])

# expose the annotate method using flask.
app.add_url_rule(rule='/annotate', endpoint='annotate', view_func=annotator.annotate, methods=['GET'])
app.run(host='localhost', port=5000)
```
Running server.py will host the server on [localhost](http://127.0.0.1:5000/) and you can get your text annotation in json format by 
sending requests to the server like so,
```
http://localhost/annotate?text="Shyam is a person and Apple is an organization"&views=DUMMYVIEW
```

## Serving Multiple Models

For serving multiple models using a single server, as in the case of multilingual models, there is a utility class `multi_annotator.py` that wraps around several annotator instances. 
For instance, you can serve NER_English, NER_Spanish, etc. all through a single server using the `MultiAnnotator` class in `multi_annotator.py`. 
You can use it to write your `server.py` that provides multiple views as follows,

```python
    annotators: List[Annotator] = []
    langs = ["es", "zh", "fr", "it", "de"]
    model_paths = [...]
    for lang, model_path in zip(langs, model_paths):
        annotator = ... # Create your language specific annotators here
        annotators.append(annotator)

    multi_annotator = MultiAnnotator(annotators=annotators)
    app.add_url_rule(rule='/annotate', endpoint='annotate', view_func=multi_annotator.annotate, methods=['GET'])
    app.run(host='0.0.0.0', port=8009)
    
```