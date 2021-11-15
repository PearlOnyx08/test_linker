import numpy as np
from spacy.training import Example
batch_size = 2
import random


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

from spacy.training import Example
import random
import spacy
nlp = spacy.load("/home/spenser/prodigy/nasa_model/update_01")

optimizer = nlp.create_optimizer()
for itn in range(20):
    print(itn)
    losses = {}

    #shuffled = new_examples.sample(len(new_examples)) #check process for reproducibility


    for batch in iterate_minibatches(new_examples['text'] ,
                                     new_examples['spans'], 
                                     batchsize=64, 
                                     shuffle=True):

        batch_texts = batch[0].reset_index(drop=True)
        batch_annotations = batch[1].reset_index(drop=True)
        example_batch_list = []
        for i in range(0, len(batch)):


            doc = nlp.make_doc(batch_texts.iloc[i])
            annotations = batch_annotations.iloc[i]

            entity_list = []
            for i in range(0, len(annotations)):

                start = annotations[i]['start']
                end = annotations[i]['end']
                label = annotations[i]['label']

                entity_list.append((start, end, label))

            annotations_dict = {'entities': entity_list}
            example = Example.from_dict(doc, annotations_dict)
            example_batch_list.append(example)

        nlp.update(example_batch_list, drop=0.35, sgd=optimizer, losses=losses)
        print(losses)
