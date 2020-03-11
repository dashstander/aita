import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import random
import torch
from torch.utils.data import Dataset, get_worker_info

class AitaDataset(Dataset):
    base_fp = 'data/pq'
    def __init__(self, tokenizer, max_length=512):
        super(AitaDataset).__init__()

        tbl = pa.concat_tables(
            [
                pq.read_table(
                    f'{self.base_fp}/{table}', 
                    columns = ['score', 'title', 'selftext']
                ) for table in os.listdir(self.base_fp)
            ]
        ).to_pandas(self_destruct=True)

        self.data = tbl.loc[tbl['selftext'] != '[removed]', :].dropna()

        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_data(self, submission):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        # For every sentence...
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_dict = self.tokenizer.encode_plus(
            submission,                      # Sentence to encode.
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # return_tensors = 'pt',     # Return pytorch tensors.
            max_length=self.max_length,
            pad_to_max_length=True,
            return_attention_mask=True
        ) 
        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        row = self.data.iloc[i, :]
        tokens, masks = self.tokenize_data(row['title']  + '\n' + row['selftext'])
        score = row['score']
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(masks, dtype=torch.float),
            torch.tensor(np.log(score), dtype=torch.float)
        )
        