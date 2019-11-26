#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 12:26:37 2019

@author: ja18581
"""

from typing import Dict, List, Sequence, Iterable, Callable, Iterator
import os
import pandas as pd
import itertools
import logging

from overrides import overrides

from glob import glob

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import spacy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ebmnlp2")
class EBMNLPDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        
        logger.info("Reading instances from file at: %s", file_path)
        data = pd.read_csv(file_path)
        # Group into alternative divider / sentence chunks.
            
        for i, row in data.iterrows():
            for sent in row['tagged_tokens']:
                fields = [(token['token'], pos, tag, pico) for token in sent for pos, tag, pico in zip(*token['tag'].split())]
                fields = [list(field) for field in zip(*fields)]
                
                tokens_,  _, _, pico_tags  = fields
                # TextField requires ``Token`` objects
                        
                tokens = [Token(token) for token in tokens_]

                yield self.text_to_instance(tokens, pico_tags)
            
    @overrides
    def text_to_instance(self,
                         tokens: List[Token],
                         pico_tags: List[str] = None):
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        
        # Set the field 'labels' according to the specified PIO element
        if pico_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pico_tags, sequence, self.label_namespace)

        return Instance(instance_fields)
