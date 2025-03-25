
import sentencepiece as spm


SENTENCEPIECE_MODEL_NAME = "sp_bpe_vietnamese"


def _prepare_tokenizer( vocab_size):
    """Prepare sentencepice tokenizer"""
    input_file = "spm_input.txt"
    model_type = "unigram"

    spm.SentencePieceTrainer.Train(
        f"--input={input_file} "
        f"--model_prefix={SENTENCEPIECE_MODEL_NAME} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--pad_id=1 --bos_id=0 --eos_id=2 --unk_id=3"
        f"--character_coverage=0.9995"
    )
    
class TokenizerSubWord():
    def __init__(self,path_model):
        self.sp= spm.SentencePieceProcessor()
        self.sp.Load(path_model)
        self.vocab_size=self.sp.get_piece_size()
        self.labels = [self.sp.IdToPiece(i) for i in range(self.vocab_size)]
    def __len__(self):
        return self.vocab_size
    def encode(self,trans):
        return self.sp.encode(trans)
    def decode(self,idxs):
        return self.sp.decode(idxs)
    def __call__(self, sentence):
        return self.encode(sentence)
    
