import spacy
from spacy import displacy
from pathlib import Path

def tokenize(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    save_figures = False

    print("token".ljust(10), "lemma".ljust(10), "pos".ljust(6), "tag".ljust(6), "dep".ljust(10),
                "shape".ljust(10), "alpha", "stop")
    print("------------------------------------------------------------------------------")
    for token in doc:
        print(token.text.ljust(10), token.lemma_.ljust(10), token.pos_.ljust(6), token.tag_.ljust(6), token.dep_.ljust(10),
                token.shape_.ljust(10), token.is_alpha, token.is_stop)

        
def recognize(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    for ent in doc.ents:
        print(ent.text.ljust(12), ent.label_.ljust(10), ent.start_char, ent.end_char)

    html_ent = displacy.render(doc, style="ent", jupyter=True)