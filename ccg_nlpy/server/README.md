To make a demo using your fancy pytorch / tensorflow / dynet model, you need to

1. Create a Annotator by subclassing the Annotator class in `annotator.py`.
    - You need to implement the `add_view` method, that will internally call your model. 

2. Write a method similar to get_view_from_model in `example/example_model.py`. This method name could be anything, you are responsible for calling this in the `add_view` method above.

3. Write a `server.py` similar to `example/example_server.py`, where you create your model, wrap it into the annotator class you wrote, and expose its annotate method using flask server. 