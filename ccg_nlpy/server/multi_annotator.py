from typing import List
from ccg_nlpy.pipeline_base import PipelineBase
from ccg_nlpy.core.text_annotation import TextAnnotation
import json

from ccg_nlpy.server.annotator import Annotator
from flask import request
import logging


class MultiAnnotator:
    """
        Wraps around several Annotator instances for serving multiple models simultaneously from a single endpoint.
        The intended use case is for serving multilingual models (e.g., NER for English, Spanish, Chinese ...)

        All the annotators should have the same set of required views.

        The annotate method implemented below is exposed through the flask server.
        It identifies the relevant annotator to call.
        For instance, if the NER_zh view is requested, it will call the annotator that provides it.

        You should subclass this class and implement add_view method.
        """
    def __init__(self, annotators: List[Annotator]):
        self.annotators = annotators
        # all models should have the same set of required views
        self.required_views = annotators[0].get_required_views()
        self.provided_views = [m.get_view_name() for m in annotators]
        print("provided views", self.provided_views)

        # for each viewname (e.g. POS_Arabic) know which model to call (Arabic_POS_Tagger)
        self.view2annotator_dict = {}
        for m in self.annotators:
            self.view2annotator_dict[m.get_view_name()] = m

        logging.info("required views: %s", self.required_views)
        logging.info("provided views: %s", self.provided_views)
        logging.info("ready!")

    def get_required_views(self) -> List[str]:
        """
        The list of viewnames required by model (e.g. NER_CONLL is needed by Wikifier)
        :return: list of viewnames
        """
        return self.required_views

    def get_view_names(self) -> List[str]:
        """
        The list of viewnames provided by model (e.g. [NER_CONLL, NER_Ontonotes] or [POS_English, POS_French, POS_ ...])
        :return: list of viewnames
        """
        return self.provided_views

    def add_view(self, docta: TextAnnotation) -> TextAnnotation:
        """
        Takes a text annotation and adds the view provided by this model to it.
        :return: TextAnnotation
        """
        raise NotImplementedError

    def annotate(self):
        # we get something like "?text=<text>&views=<views>". Below two lines extract these.
        text = request.args.get('text')
        views = request.args.get('views')
        logging.info("request args views:%s", views)
        if text is None or views is None:
            return "The parameters 'text' and/or 'views' are not specified. Here is a sample input: ?text=\"This is a " \
                   "sample sentence. I'm happy.\"&views=POS,NER "
        views = views.split(",")

        for view in views:
            if view in self.provided_views:

                # select the correct model
                relevant_annotator: Annotator = self.view2annotator_dict[view]
                # create a text ann with the required views for the model
                docta = relevant_annotator.get_text_annotation_for_model(text=text, required_views=self.required_views)
                # send it to your model for inference
                docta = relevant_annotator.add_view(docta=docta)
                # make the returned text ann to a json
                ta_json = json.dumps(docta.as_json)
                # print("returning", ta_json)
                return ta_json
        # If we reached here, it means the requested view cannot be provided by this annotator
        return "VIEW NOT PROVIDED"
