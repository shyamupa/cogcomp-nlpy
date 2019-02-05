from ccg_nlpy import TextAnnotation
import copy

from ccg_nlpy.core.view import View


class ExampleModel:
    """This would be your pytorch/dynet/tensorflow model"""
    def __init__(self):
        pass

    def get_view_from_model(self, docta:TextAnnotation) -> View:
        # This upcases each token. Test for TokenLabelView
        new_view = copy.deepcopy(docta.get_view("TOKENS"))
        tokens = docta.get_tokens
        for token, cons in zip(tokens, new_view.cons_list):
            cons["label"] = token.upper()

        # # This replaces each NER with its upcased tokens. Test for SpanLabelView
        # new_view = copy.deepcopy(docta.get_view("NER_CONLL"))
        # for nercons in new_view.cons_list:
        #     nercons["label"] = nercons["tokens"].upper()

        return new_view