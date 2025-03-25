import torch
import torchaudio
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from typing import Dict, List, Optional, Union
import torchaudio.transforms as T

from inference import Inference

class VLSPDataset(Dataset):
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        path='/kaggle/input/vin-big-data-vlsp-2020-100h',
        path_csv='/kaggle/input/datacsv/VLSP_under10s_train.csv',
    ):
        super().__init__()
        df = pd.read_csv(path_csv)
        df.path = path + os.sep + df.path
        self.walker = df.to_dict("records")
        self.processor =processor 
    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        path, trans = self.walker[idx]['path'],self.walker[idx]['text']
        with self.processor.as_target_processor():
            trans=self.processor(trans.lower())['input_ids']
        wave, sr = torchaudio.load(path)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]

        return  {'input_values':specs, 'labels':trans}
    
class SemiSupervisedDataset(Dataset):
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        path_vlsp:str="/kaggle/input/vin-big-data-vlsp-2020-100h",
        path_vivos:str="/kaggle/input/vivos-vietnamese/vivos",
        path_commonv:str='/kaggle/input/commonvoice-vie/cv-corpus-15.0-2023-09-08',
        path_vtv24:str='/kaggle/input/audio-crawl-unlabel',
        
        path_csv_vlsp :str= "/kaggle/input/datacsv/VLSP_under10s_train.csv",
        path_csv_vios:str='/kaggle/input/datacsv/vivos_upder10s_train.csv',
        path_csv_commonv:str='/kaggle/input/datacsv/commonvoice_train.csv',
        path_csv_vtv24:str='/kaggle/input/audio-crawl-unlabel/train.csv',
        **kwargs,
    ):
        super().__init__()

       
        df_VLSP = pd.read_csv(path_csv_vlsp)
        df_VLSP.path = path_vlsp + os.sep + df_VLSP.path
        
   
        df_vivos=pd.read_csv(path_csv_vios)
        df_vivos.path=path_vivos+os.sep+df_vivos.path
        
        df_commonv=pd.read_csv(path_csv_commonv)
        df_commonv.path=path_commonv+os.sep+df_commonv.path
        

        df_vtv24 = pd.read_csv(path_csv_vtv24)
        df_vtv24['path'] = df_vtv24['path'].str.replace('\\', '/', regex=False)
        df_vtv24.path=path_vtv24+os.sep+df_vtv24.path
    
        walker_VLSP = df_VLSP.to_dict("records")
        
        walker_vivos=df_vivos['path'].to_list()
        walker_vtv24=df_vtv24['path'].to_list()
        walker_commonv=df_commonv['path'].to_list()
        
        self.walker=walker_vivos+walker_VLSP+walker_vtv24+walker_commonv
        random.shuffle(self.walker)
        self.processor=processor

         
    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]
        if type(item) == dict:
            return self.load_labeled_item(item)
        return self.load_unlabeled_item(item)

    def load_unlabeled_item(self, item):
        wave, sr = torchaudio.load(item)
        
        if sr >16000:
            resample_transform = T.Resample(orig_freq=sr, new_freq=16000)
            wave = resample_transform(wave)
        
#         wave=self.augment_waveform(wave)
        
        specs = self.processor(wave,text=None,sampling_rate=16000,output_tensor='pt').input_values[0][0]
        return  {'input_values':specs, 'labels':[]}       

    def load_labeled_item(self, item):
        path, trans = item['path'],item['text']
        with self.processor.as_target_processor():
            trans=self.processor(trans.lower())['input_ids']
        wave, sr = torchaudio.load(path)
     
        if sr >16000:
            resample_transform = T.Resample(orig_freq=sr, new_freq=16000)
            wave = resample_transform(wave)
        
       
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]
        return  {'input_values':specs, 'labels':trans}

class DatasetValidated(Dataset):
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        path='/kaggle/input/vivos-vietnamese/vivos',
        path_csv='/kaggle/input/datacsv/vivos_test.csv',

    ):
        super().__init__()
        df= pd.read_csv(path_csv)
        df.path=path+os.sep+df.path
        self.walker = df.to_dict("records")
        self.processor =processor 
    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        path, trans = self.walker[idx]['path'],self.walker[idx]['text']
        with self.processor.as_target_processor():
            trans=self.processor(trans.lower())['input_ids']
        wave, sr = torchaudio.load(path)
        if sr !=16000:
            resample_transform = T.Resample(orig_freq=sr, new_freq=16000)
            wave = resample_transform(wave)
        specs = self.processor(wave,sampling_rate=16000,output_tensor='pt').input_values[0][0]
        return  {'input_values':specs, 'labels':trans}
    



class DataCollatorTeacherCTCWithPadding:
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        teacher:Inference,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.processor=processor
        self.teacher=teacher
        self.padding=padding
        self.max_length=max_length
        self.max_length_labels=max_length_labels
        self.pad_to_multiple_of=pad_to_multiple_of
        self.pad_to_multiple_of_labels=pad_to_multiple_of_labels
    def __call__(self, batchs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": batch["input_values"]} for batch in batchs]
        label_features = [{"input_ids": batch["labels"]} for batch in batchs]
        mask_unlabel=[len(label_feature['input_ids'])==0 for label_feature in label_features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",

        )
        pre_values=batch["input_values"][torch.tensor(mask_unlabel)]
        if len(pre_values)>0:
            label_pre=self.teacher.bach_predict(pre_values,30)
            label_pre=self.processor(text=label_pre).input_ids
        for i,mask in  enumerate(mask_unlabel):
            if mask:
                label_features[i]["input_ids"]=label_pre.pop(0)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
    
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["input_values"]= batch["input_values"]
       
        return batch
    
class DataCollatorCTCWithPadding:
    def __init__(
        self,
        processor:Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.processor=processor
        self.padding=padding
        self.max_length=max_length
        self.max_length_labels=max_length_labels
        self.pad_to_multiple_of=pad_to_multiple_of
        self.pad_to_multiple_of_labels=pad_to_multiple_of_labels
    def __call__(self, batchs: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": batch["input_values"]} for batch in batchs]
        label_features = [{"input_ids": batch["labels"]} for batch in batchs]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",

        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
    
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["input_values"]= batch["input_values"]
        return batch
