#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:41:51 2019

@author: ja18581
"""

import pandas as pd
import os
from data_formatter import all_formats


        
class txtNER(all_formats):
    def __init__(self, file_path):
        super(txtNER, self).__init__(file_path)
        self.logger.info('Initializing in txtNER')
        self.config.outfile = 'data/ner_tokenized.txt'
     
    def spacyTokenize(self,abstractsdf: pd.DataFrame())-> None:
        self.logger.info(f"In TXT spaceTokenize")
        self.logger.info(f"Creating tokens from input data")
        
        
        if os.path.exists(self.config.outfile):#housekeeping
            self.logger.info(f'File: {self.config.outfile} exists in the directory. Now being removed.\n')
            os.remove(self.config.outfile)
                
        for i, row in abstractsdf.iterrows():
            try:
                doc = self.nlp(row['abstract'])
            except:
                doc = self.nlp(row['Abstract'])
            
            self.spacy_doc_list.append([row['pmid'], doc])
    
    
    def OutputTaggedToken(self):
        self.logger.info(f"Writing output data from txtNER")
        with open(self.config.outfile, 'a') as f:
            for pmid, spacy_doc in self.spacy_doc_list:
                f.write(f'-DOCSTART- ({pmid})\n')
                #logger.info(f"Writing CONLL formmatted output to output file at: {config.outfile}")
                for sent in spacy_doc.sents:
                    tokens = [token.text.strip() +' NN O O' for token in sent if token.text.strip() != '']
                    f.write('\n')
                    for item in tokens:
                        f.write("%s\n" % item.strip())
                f.write('\n' * 3)        
                

    
# =============================================================================
# if __name__=='__main__':
#     logger = logging.getLogger('ner')
# 
#     main()
# =============================================================================
