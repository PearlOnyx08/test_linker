from setuptools import setup

setup(
    name="my_package",
    entry_points={
        "console_scripts": ["spacy_ann=spacy_ann.cli:main"],
        "spacy_factories": ["ann_linker=spacy_ann.ann_linker:AnnLinker"],
        "spacy_kb": ["get_candidates=spacy_ann:get_candidates"],
    },
)
