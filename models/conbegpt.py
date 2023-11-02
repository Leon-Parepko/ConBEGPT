from models.Conbert.conbert_wrapper import Conbert
from models.Gpt2.gpt2_wrapper import GPT2
# from models.Estimator.estimator import Estimator


class Conbergpt():
    def __init__(self, device, conbert_dir, gpt2_dir):
        self.device = device
        self.conbert_dir = conbert_dir
        self.gpt2_dir = gpt2_dir

        print('Loading Conditional Bert model...')
        self.conbertr = Conbert(device, conbert_dir)

        print('Loading Estimator model...')
        self.estmator = None

        print('Loading GPT-2 model...')
        self.gpt2 = GPT2(gpt2_dir)

    def detoxicate(self, text, max_len, only_conbert=False):

        gpt2_prompt = f'<T>{text}'

        # Forward Conbert
        conbert_out = self.conbertr.detoxicate(text)
        gpt2_prompt += f'<NT>{conbert_out}'

        # Forward GPT-2
        if not only_conbert:
            gpt2_out = self.gpt2.generate(gpt2_prompt, max_length=max_len)

        return gpt2_out

