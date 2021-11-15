"""
Utility functions to support NER model updates.
"""
import spacy

def format_annotations(doc_bin):
    """Loads and formats .spacy DocBin formatted annotations.

    Args:
    doc_bin -- .spacy data imported using spacy.tokens.DocBin

    Returns:
        A dict mapping keys to the corresponding text and span row data.
        Each row is represented as a tuple of strings. For example:

        {'text': 'Entity freeform annotation text',
         'spans': [{"start": 0, "end": 6, "label": label}]}
    """
    nlp = spacy.blank("en") # use blank nlp model for text parsing.
    annotations = []
    for doc in doc_bin.get_docs(nlp.vocab):
        spans = []
        for ent in doc.ents:
            span = [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
            ]
            spans.append(span)
            
        annotations.append({"text": doc.text, "spans": spans})
    return annotations


def filter_new_annotations(train_history, annotations):
    """Removes texts used in previous training iterations
       from the annotations data. 

    Args:
    train_history -- Dataframe of texts and spans previously used in training.
    annotations -- Dataframe of all texts and spans from updated train.spacy

    Returns:
        A pandas dataframe containing rows of new training texts and entity
        spans that can be used to update the NER model.
    """
    merged = train_history.merge(
        annotations, on=["text"], how="right", suffixes=("_in_train", "_new")
    )
    new_texts = merged[merged["train_index"].isnull()] # filter to new texts with no train 
    new_texts = new_texts[["text", "spans_new"]]
    new_texts.columns = ["text", "spans"]
    new_texts = new_texts.reset_index(drop=True) #reset indexes for batch processing
    return new_texts

def compare_train_spans(train_history, annotations):
    """Outputs a dataframe of duplicate texts with different spans 
       from the annotations data.

    Args:
    train_history -- Dataframe of texts and spans previously used in training.
    annotations -- Dataframe of all texts and spans from updated train.spacy

    Returns:
        A dataframe containing annotation examples with different span
        start and end points. 
    """
    merged = train_history.merge(
        annotations, on=["text"], how="inner", suffixes=("_in_train", "_new")
    )
    train_texts = merged[merged["train_index"].isnull() == False]
    new_spans = train_texts[train_texts["spans_in_train"] != train_texts["spans_new"]]
    # Format dataframe to contain a column for the new texts and span information.
    new_spans = new_spans[["text", "spans_in_train", "spans_new"]]
    # Reset indexes so that Spacy's batch iteration functions process the data.
    new_spans = new_spans.reset_index(drop=True)
    return new_spans
