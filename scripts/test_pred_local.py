from app.models.ner_prediction import NERPrediction

nr = NERPrediction(text='test', tokens=['a','b'], labels=['O','B-PER'], entities={'PER': ['a']}, formatted_output='<p>test</p>', format='html', inference_time_ms=10.5, user_id=None, extra_metadata={'language':'khmer'})
print('ok', nr.text, type(nr.tokens), type(nr.entities), nr.extra_metadata)
