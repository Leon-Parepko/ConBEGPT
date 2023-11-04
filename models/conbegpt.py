from models.Conbert.conbert_wrapper import Conbert
from models.Gpt2.gpt2_wrapper import GPT2
from models.Estimator.estimator_wrapper import Estimator


class Conbergpt():
    """
    The main class that wraps the ConBERT, GPT-2 and Estimator models.
    !THE CON-BE-GPT MODEL!
    """
    def __init__(self, device, conbert_dir, estimator_dir, gpt2_dir):
        self.device = device
        self.conbert_dir = conbert_dir
        self.gpt2_dir = gpt2_dir
        self.estimator_dir = estimator_dir

        print('Loading Conditional Bert model...')
        self.conbertr = Conbert(device, conbert_dir)

        print('Loading Estimator model...')
        self.estmator = Estimator(model_dir=estimator_dir, device=device)

        print('Loading GPT-2 model...')
        self.gpt2 = GPT2(gpt2_dir)

    def detoxicate(self, text, max_len, only_conbert=False):
        """
        Detoxicates a given text using the ConBEGPT model.
        :param text: input text
        :param max_len: integer that specifies the maximum length of the generated text
        :param only_conbert: if True, only the ConBERT model is used
        :return: detoxicated text
        """

        gpt2_prompt = f'<T>{text}'

        # Forward Conbert
        out = self.conbertr.detoxicate(text)
        gpt2_prompt += f'<NT>{out}'

        # Forward GPT-2
        if not only_conbert:
            out = self.gpt2.generate(gpt2_prompt, max_length=max_len)

        return out

