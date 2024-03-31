# Import libary yang dibutuhkan
import pandas as pd
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from sklearn.metrics.pairwise import cosine_similarity

# Membuat class yang akan menjadi dasar dari segala macam operasi
class IndoBertModelRep(object):

    def __init__(self, data : list[str], model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1"),
                  tokinizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")):
        self.model = model
        self.tokinizer = tokinizer
        self.data = self.__process(data)

    def __repr__(self) -> str:
         return "[ Model experimental untuk mengubah sentence ke vector ] base -> IndoBert"

    def __process(self,data : list[str]) -> dict[str, list[torch.Tensor]] | dict[str, torch.Tensor]:
        max_lenght = 256
        tokens = {"input_ids":[],"attention_mask":[]}
        for sentence in tqdm(data,desc="transforming tokens"):
            text = self.tokinizer.encode_plus(
                sentence,
                max_length = max_lenght,
                padding = "max_length",
                truncation = True,
                return_tensors = "pt"
            )

            tokens["input_ids"].append(text["input_ids"][0])
            tokens["attention_mask"].append(text["attention_mask"][0])

        tokens["input_ids"] = torch.stack(tokens["input_ids"])
        tokens["attention_mask"] = torch.stack(tokens["attention_mask"])

        return tokens
    

    def __fittingData(self,dataToken : dict[str , torch.Tensor]) -> np.ndarray[int, float]:
            
            self.model.eval()
            output = self.model(**dataToken)
            embeddings = output.last_hidden_state

            attention = dataToken["attention_mask"]
            attention.unsqueeze(-1).shape

            val = attention.unsqueeze(-1)
            mask = val.expand(embeddings.shape).float()
            mask_embeddings = embeddings * mask

            summed = torch.sum(mask_embeddings,1)
            summed_mask = mask.sum(1)
            clamp_summed_mask = torch.clamp(summed_mask,min=1e-9)
            mean_pooling = summed / clamp_summed_mask

            return mean_pooling.detach().numpy()    


    def fit_transform(self, batch : int) -> pd.DataFrame: # type: ignore
        totalBatch = (len(self.data["input_ids"]) // batch) + 1
        baseBatch = 0
        baseEndBatch = batch
        limitBatch = len(self.data["input_ids"])
        self.arrayData = []

        for i in tqdm(range(totalBatch),desc="Training"):
            dataTrain = {"input_ids" : self.data["input_ids"][baseBatch:baseEndBatch if baseEndBatch < limitBatch else limitBatch],
                         "attention_mask":self.data["attention_mask"][baseBatch:baseEndBatch if baseEndBatch < limitBatch else limitBatch]}

            baseBatch += batch
            baseEndBatch += batch
            clean = self.__fittingData(dataTrain) # type: ignore
            self.arrayData.append(pd.DataFrame(clean))

            
        self.bankData = pd.concat(self.arrayData).reset_index(drop=True)
        self.transformSentence = self.__transformSentence
        return self.bankData
    


    def __transformSentence(self,sentence : str, returnPredict : bool = False) -> dict[str, np.ndarray[int, float]] | list[dict[str, np.ndarray[int, float]] | pd.DataFrame]: # type: ignore
        tokens = self.__process([sentence])
        arrayToken = self.__fittingData(tokens) # type: ignore
        

        #kalkulasi kemiripan
        cos = cosine_similarity(arrayToken,self.bankData.values)
        result = dict(index = cos[0],result = np.flip(cos.argsort())[0])

        if(returnPredict):
             return [result, pd.DataFrame(arrayToken)]
        
        return result
        