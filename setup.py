from setuptools import setup

setup(
    name='al3',
    version='0.1',
    
    entry_points='''
        [console_scripts]
        spacy_ann=spacy_ann.cli:__init__.py

        [spacy_factories]
        ann_linker=spacy_ann.ann_linker:AnnLinker
        remote_ann_linker=spacy_ann.remote_ann_linker:RemoteAnnLinker

        [spacy_kb]
        get_candidates=spacy_ann:get_candidates
    ''',
)
