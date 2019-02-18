from typing import List
from ccg_nlpy.pipeline_base import PipelineBase
from ccg_nlpy.core.text_annotation import TextAnnotation
import json
from flask import request
import logging


class Annotator:
    """
    Wraps around your python model, and calls it to get new views that are provided by it.
    The annotate method implemented below is exposed through the flask server.

    You should subclass this class and implement add_view and get_text_annotation_for_model methods.
    If you are serving a multilingual model, please see multi_annotator.py also.
    """

    def __init__(self, pipeline: PipelineBase, provided_view: str, required_views: List[str]):
        # the viewname provided by the model
        self.provided_view = provided_view
        # the views required by the model (e.g. NER_CONLL for Wikifier)
        self.required_views = required_views
        # right now, we call the model load inside the init of server
        # this could have been done outside. Cannot say which is a better choice.
        # self.load_params()
        # We need a pipeline to create views that are required by our model (e.g. NER is needed for WIKIFIER etc.)
        self.pipeline = pipeline
        logging.info("required views: %s", self.get_required_views())
        logging.info("provides view: %s", self.get_view_name())
        logging.info("ready!")

    def get_required_views(self) -> List[str]:
        """
        The list of viewnames required by model (e.g. NER_CONLL is needed by Wikifier)
        :return: list of viewnames
        """
        return self.required_views

    def get_view_name(self) -> str:
        """
        The viewname provided by model (e.g. NER_CONLL)
        :return: viewname
        """
        return self.provided_view

    def add_view(self, docta: TextAnnotation) -> TextAnnotation:
        """
        Takes a text annotation and adds the view provided by this model to it.
        :return: TextAnnotation
        """
        raise NotImplementedError

    def annotate(self) -> str:
        """
        The method exposed through the flask interface.
        :return: json of a text annotation
        """
        # we get something like "?text=<text>&views=<views>". Below two lines extract these.
        text = request.args.get('text')
        views = request.args.get('views')
        logging.info("request args views:%s", views)
        if text is None or views is None:
            return "The parameters 'text' and/or 'views' are not specified. Here is a sample input: ?text=\"This is a " \
                   "sample sentence. I'm happy.\"&views=POS,NER "
        views = views.split(",")
        if self.provided_view not in views:
            logging.info("desired view not provided by this server.")
            # After discussing with Daniel, this is the proper discipline to handle views not provided by this.
            # The appelles server will fallback to the next remote server.
            return "VIEW NOT PROVIDED"

        # create a text ann with the required views for the model
        docta = self.get_text_annotation_for_model(text=text, required_views=self.get_required_views())

        # send it to your model for inference
        docta = self.add_view(docta=docta)

        # make the returned text ann to a json
        ta_json = json.dumps(docta.as_json)

        return ta_json

    def get_text_annotation_for_model(self, text: str, required_views: List[str]) -> TextAnnotation:
        """
        This takes text from the annotate api call and creates a text annotation with the views required by the model.
        :param text: text from the demo interface, coming through the annotate request call
        :param required_views: views required by the model
        :return: text annotation, to be sent to the model's inference on ta method
        """
        raise NotImplementedError
