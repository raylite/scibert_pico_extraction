#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:38:42 2019

@author: ja18581
"""

import subprocess
import logging

from txt_based_formatter  import txtNER
from csv_based_formatter import csvNER
from extract_pico import extract_pico

logging.basicConfig(level=logging.DEBUG)

def format_data(infile) -> None:
    
    formatters = [txtNER(infile), csvNER(infile)]
    
    for formatter in formatters:
        data = formatter.load_input_file()
        formatter.spacyTokenize(data)
        formatter.OutputTaggedToken()
        

def predict() -> None:
    prediction = subprocess.run(['allennlp predict --include-package scibert \
                   --predictor dummy_predictor --use-dataset-reader \
                   --output-file output/auto_label_test.json archive/model.tar.gz data/ner_tokenized.txt'],
    shell=True)
    
    return prediction
    
    
def extract_prediction_from_json(json_file, base_file, outfile) -> None:
    extractor = extract_pico(json_file, base_file, outfile)
    extractor.tag_and_write_spans()
    


if __name__=='__main__':
    logger = logging.getLogger('pico_main')
    
    json_file = 'output/auto_label_test.json'
    base_file = 'data/ner_tokenized.csv'
    initial_file = 'data/label_test_reprort.csv'
    final_outfile = 'output/interventions_suggestion.csv'
    
    logger.info('Calling data formmaters to prep the dataset for prediction')
    format_data(initial_file)
    
    logger.info('Invoking the NER trained model to predict on data')
    result = predict()
    logger.info(f'Result: {result}')
    
    logger.info('Going to extract the predictions and map predicted tags/span and tokens.')
    
    extract_prediction_from_json(json_file, base_file, final_outfile)
    logger.info('Predictions fully extracted')
    
    logger.info('PICO tagging and extraction opreations completed')
    
    

