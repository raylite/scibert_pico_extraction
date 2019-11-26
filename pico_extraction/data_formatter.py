#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:32:09 2019

@author: ja18581
"""

import spacy
import logging
import pandas as pd



logging.basicConfig(level=logging.DEBUG)

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
            
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        


class all_formats(object):
    def __init__(self, input_file_path):
        self.nlp = spacy.load('en_core_sci_lg', disable=['ner', 'tagger'])
        self.logger = logging.getLogger('pico_ner_formatter')
        self.spacy_doc_list = []
        self.config = Config(
                infile=input_file_path,
                outfile = None,
                )
        
    def spacyTokenize(self, abstractsdf: pd.DataFrame())-> None:
        pass
    
    
    def OutputTaggedToken(self):
        pass
          
    
    def load_input_file(self):
        self.logger.info(f"Reading data from input file at: {self.config.infile}")
        data: self.config.filepath_or_buffer = pd.read_csv(self.config.infile)
        self.logger.info(f"Input file at: {self.config.infile} reading successful")
        try:
            data = data.drop(['Sentences_1','Sentences_2','Sentences_3', 'Screeners'], axis=1)
            self.logger.info('Cleaned headers')
        except:
            self.logger.info('UNCleaned headers')
            return data
        
        else:
            self.logger.info("File read operation successful")
            return data
        
