from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

import transformers
print(transformers.__version__)
print(transformers.__file__)