


import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from copy import deepcopy

file_path = "train.spacy"
doc_bin = DocBin().from_disk(file_path)

def parse_docbin(doc_bin):
    nlp = spacy.blank("en")
    annotations = []
    for doc in doc_bin.get_docs(nlp.vocab):
        spans = []
        for ent in doc.ents:
            end_adjustment = ent.end - 1
            spans.append({"start": ent.start, "end": ent.end, "label": ent.label_})
        annotations.append({"text": doc.text, "spans": spans})
    return annotations
  
  def character_spans(json_data):
    nlp = spacy.blank("en")
    character_annotations = []
    for row in deepcopy(json_data):
        doc = nlp(row["text"])
        spans = []
        for span in row["spans"]:
            entity_span = doc[span["start"] : span["end"]]
            start = entity_span.start_char
            end = entity_span.end_char
            spans.append({"start": start, "end": end, "label": span['label']})
        row['char_spans'] = spans
        row['token_spans'] = row.pop('spans')
        character_annotations.append(row)
    return character_annotations
  
  #char = character_spans(parsed_doc_bin)
  
  def character_spans_to_biluo(json_data):
    biluo_annotations = []
    nlp = spacy.blank("en")
    data_out = deepcopy(json_data)
    for row in data_out:
        doc = nlp(row['text'])
        spans = []
        for span in row['char_spans']:
            spans.append((span["start"], span["end"], span["label"]))
        annotation_dict = {"entities": spans}
        example = Example.from_dict(doc, annotation_dict)
        biluo = example.get_aligned_ner()
        row['biluo'] = biluo
    return data_out
 

def json_to_spacy(json_data):
    docbin = DocBin()
    
    nlp = spacy.blank('en')
    for row in json_data:
        doc = nlp.make_doc(row['text'])
        annotation_dict = {"entities": row["biluo"]}
        example = Example.from_dict(doc, annotation_dict)
        docbin.add(example.reference)
    return docbin
  
  
