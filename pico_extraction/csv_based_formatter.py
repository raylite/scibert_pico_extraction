#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:39:55 2019

@author: ja18581
"""

import pandas as pd
from data_formatter import all_formats



class csvNER(all_formats):
    
    def __init__(self, file_path):
        super(csvNER, self).__init__(file_path)
        self.logger.info('Initializing in csvNER')
        self.config.outfile = 'data/ner_tokenized.csv'
        
        
    def spacyTokenize(self, abstractsdf: pd.DataFrame())-> None:
        self.logger.info(f"In CSV spaceTokenize")
        self.logger.info(f"Creating tokens from input data")
        
        for i, row in abstractsdf.iterrows():
            try:
                doc = self.nlp(row['abstract'])
            except:
                doc = self.nlp(row['Abstract'])
            tokens_tag = [[{"token":token.text.strip(), "tag": "NN O O"} for token in sent if token.text.strip() != ''] for sent in doc.sents]
           
            self.spacy_doc_list.append({'tagged_tokens': tokens_tag, 'sents_length': len(tokens_tag)})
        self.org_df = abstractsdf
        
    
    
    def OutputTaggedToken(self):
        
        temp_df = pd.DataFrame(self.spacy_doc_list)
        
        self.logger.info(f"Writing output data from csvNER")
        self.org_df['tagged_tokens'] = temp_df['tagged_tokens'].values
        self.org_df['sents_length'] = temp_df['sents_length'].values
        
        self.org_df.to_csv(self.config.outfile, index=False)
      
            


#if __name__=='__main__':
#    logger = logging.getLogger('ner')
#
#    main()
#    logger.info('Run successful')
    