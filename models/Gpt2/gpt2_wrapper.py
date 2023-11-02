from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer


class GPT2:
    def __init__(self, model_dir):
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    def clean_output(self, string):
        start = string.find("<F>") + 3
        end = string.find("<T>", start)
        return string[start:end]

    def generate(self, input, max_length, preprocess_out=True):
        ids = self.tokenizer.encode(f'{input}', return_tensors='pt')
        final_outputs = self.model.generate(
            ids,
            do_sample=True,
            max_length=max_length,
            pad_token_id=self.model.config.eos_token_id,
            top_k=50,
            top_p=0.95,
        )

        out = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)

        if preprocess_out:
            out = self.clean_output(out)

        return out