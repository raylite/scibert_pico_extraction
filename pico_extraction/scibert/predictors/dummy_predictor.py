from allennlp.predictors.predictor import Predictor
@Predictor.register('dummy_predictor')
class DummyPredictor(Predictor):
    pass