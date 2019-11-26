#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:10:24 2019

@author: ja18581
"""

import pandas as pd
import json
from pandas.io.json import json_normalize
from collections import Counter
import logging

logging.basicConfig(level=logging.DEBUG)

class extract_pico():
    def __init__(self, pico_file, base_file, outfile):
        self.logger = logging.getLogger('json_extractor')
        #load required files
        self.logger.info('Initialising extractor instance')
        self.pico_json = [json.loads(rec) for rec in open(pico_file)]
        self.source_file = pd.read_csv(base_file)#to retrieve sentence length for each doc
                
        self.df = json_normalize(self.pico_json)
        self.df= self.df.drop(['logits', 'mask', 'loss'], axis = 1)
        self.outfile = outfile
        
  
    def get_INT_span_and_count(self, tokens_tag_df: pd.DataFrame) -> [list, Counter]:
        prev_i = 0
        pending = False
        counter = Counter()
        interventions = []
        
        for i, row in tokens_tag_df.iterrows():
            if row['tag'] == 'I-INT':
                
                pending = True
                
                if prev_i > i or prev_i == 0:
                    prev_i = 0
                    span = '' 
                    span = span + row["token"]
                
                elif i - prev_i == 1 and prev_i != 0:
                    span = span + ' ' + row["token"]
                    
                prev_i = i
                
            elif pending:
                interventions.append(span)
                counter[span.lower()] += 1
                pending = False
        
        return interventions, counter
    
    def _write_tagged_tokens(self, ids: list, abstracts: list, spans_list: list, counters: list, tags_list: list) -> None:
        #construct a csv of the abstract, interventions, intervention_count 
        self.logger.info('Write result to file')
        all_records = []
        for id, abstract, span_list, intervention_counter, token_tags in zip(ids, abstracts,spans_list, counters, tags_list):
            out_record = {'pmid': id,
                          'abstract': abstract, 
                          'unique_interventions': list(set([x.lower() for x in span_list])), 
                          'intervention_count': [(k, intervention_counter[k]) for k in intervention_counter],
                          'suggetsted_itervention': intervention_counter.most_common(2),
                          'predicted_tags': token_tags}
            
            all_records.append(out_record)
        
        df = pd.DataFrame(all_records)
        df.to_csv(self.outfile, index=False, )
        
    
    def tag_and_write_spans(self):
        spans_list, counters, abstracts, tags_list, ids = [], [], [],[], []
        
        num_of_sents_per_abstract = self.source_file['sents_length']
        self.logger.info(f'Sents num: {num_of_sents_per_abstract}')
        start = 0
        end = 0
        self.logger.info('Commencing span determination and tagging')
        for i in range(len(num_of_sents_per_abstract)):
            if i == 0:
                end += num_of_sents_per_abstract[i]
                sub_df = self.df.iloc[start : end]
                start = end
            else:
                end += num_of_sents_per_abstract[i]
                sub_df = self.df.iloc[start : end]
                start = end
                
            try:
                abstracts.append(self.source_file.at[i, 'abstract'])
            except:
                abstracts.append(self.source_file.at[i, 'Abstract'])
                
            ids.append((self.source_file.at[i, 'pmid']))
            paired = [{'token': x[0], 'tag': x[1]} for i, row in sub_df.iterrows() for x in zip(row['words'], row['tags'])]
            tokens_df = pd.DataFrame(paired)
            span_list, intervention_counter = self.get_INT_span_and_count(tokens_df)
            spans_list.append(span_list)
            counters.append(intervention_counter)
            tags_list.append([f'{x["token"]} ({x["tag"]})' if x['tag'] != 'O' else f'{x["token"]}' for x in paired])
        
        self._write_tagged_tokens(ids, abstracts, spans_list, counters, tags_list)
