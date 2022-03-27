
import json
import logging
import math
import os
import random

from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader




#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


DPR_DATASET_PATH = "data/mdd_dpr/dpr.multidoc2dial_all.structure.train.json"
MODEL_OUTPUT_PATH = "checkpoints/x-encoder-distilroberta"
POS_LABEL = 1
NEG_LABEL = 0
NEG_POS_RATIO = 4


with open(DPR_DATASET_PATH) as f:
    dpr_data = json.load(f)


len(dpr_data[0]["negative_ctxs"])


def textify_ctx_instance(ctx, question):
    return question + " // " + ctx["title"] + " // " + ctx["text"]


dpr_data[0]


random.shuffle(dpr_data)

train_data = dpr_data[:-5000]
dev_data = dpr_data[-5000:]

train_samples = []
dev_samples = {}

for instance in dev_data:
    qid = instance["qid"]
    dev_samples[qid] = {"query": instance["question"]}
    dev_samples[qid]["positive"] = [textify_ctx_instance(ctx, instance["question"]) for ctx in instance["positive_ctxs"]]
    dev_samples[qid]["negative"] = [textify_ctx_instance(ctx, instance["question"]) for ctx in instance["negative_ctxs"][:NEG_POS_RATIO]]

for instance in train_data:
    question = instance["question"]
    for ctx in instance["positive_ctxs"]:
        train_samples.append(InputExample(texts=[question, textify_ctx_instance(ctx, question)], label=POS_LABEL))
    for ctx in instance["negative_ctxs"]:
        train_samples.append(InputExample(texts=[question, textify_ctx_instance(ctx, question)], label=NEG_LABEL))




len(train_samples), len(dev_samples)

train_batch_size = 16
num_epochs = 4


#Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
model = CrossEncoder('distilroberta-base', num_labels=1)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
evaluator = CERerankingEvaluator(dev_samples, name='reranker-dev')


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=MODEL_OUTPUT_PATH)





