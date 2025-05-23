import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
)

class Wav2Vec2WordpieceTokenizer(Wav2Vec2CTCTokenizer):
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )

    
        
    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        output_tokens = []
        for token_idx, token in enumerate(text.split()):
            end = len(token)
            sub_tokens = []
            while end > 0:
                start = 0
                cur_substr = None
                while start < end:
                    substr = token[start:end]
                    if substr in self.encoder:
                        cur_substr = substr
                        break
                    start += 1
                if cur_substr is None:
                    sub_tokens.insert(0, self.unk_token)
                    end = start - 1
                else:
                    sub_tokens.insert(0, cur_substr)
                    end = start
                
            
            if token_idx > 0:
                output_tokens.append(self.word_delimiter_token)
            output_tokens.extend(sub_tokens)
        return output_tokens
    def encode(self, text, **kwargs):
  
        if(type(text) ==str):  
            tokens = self._tokenize(text, **kwargs)
            token_ids = [self.encoder.get(token, self.unk_token_id) for token in tokens]
            return token_ids
        
        ids=[]
        for t in text:
            tokens = self._tokenize(t, **kwargs)
            token_ids = [self.encoder.get(token, self.unk_token_id) for token in tokens]
            ids.append(token_ids)
        return ids
    def __call__(self, text, return_tensors=None, **kwargs):
        token_ids = self.encode(text, **kwargs)

        if return_tensors == "pt":
            return torch.tensor([token_ids])

        return {"input_ids": token_ids}
    def decode_ids(
        self, 
        token_ids, 
        skip_special_tokens = False, 
        clean_up_tokenization_spaces = True,
        group_tokens: bool = True,
        spaces_between_special_tokens: bool = False,
    ) -> str:
        # For compatible with speechbrain interfaces
        return self.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            group_tokens=group_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens
        )