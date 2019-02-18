import logging
from typing import List

from ccg_nlpy import local_pipeline, TextAnnotation
from ccg_nlpy.pipeline_base import PipelineBase
from ccg_nlpy.server.annotator import Annotator

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class ExampleAnnotator(Annotator):
    """
    A dummy model that is used with the model wrapper server You need to define two methods load_params and
    inference_on_ta when writing your own model, for it to be compatible with the model wrapper server.
    """
    def __init__(self, model, pipeline: PipelineBase, provided_view: str, required_views: List[str]):
        super().__init__(pipeline=pipeline, provided_view=provided_view, required_views=required_views)
        self.model = model

    # def load_params(self):
    #     logging.info("loading model params ...")
    #     raise NotImplementedError

    def add_view(self, docta):
        # ask the model to create the new view
        new_view = self.model.get_view_from_text_annotation(docta)
        # add it to the text annotation
        new_view.view_name = self.provided_view
        docta.view_dictionary[self.provided_view] = new_view
        return docta

    def get_text_annotation_for_model(self, text: str, required_views: List[str]):
        # TODO This is a problem with ccg_nlpy text annotation, it does not like newlines (e.g., marking paragraphs)
        text = text.replace("\n", "")
        pretokenized_text = [text.split(" ")]
        required_views = ",".join(required_views)
        ta_json = self.pipeline.call_server_pretokenized(pretokenized_text=pretokenized_text, views=required_views)
        ta = TextAnnotation(json_str=ta_json)
        return ta
