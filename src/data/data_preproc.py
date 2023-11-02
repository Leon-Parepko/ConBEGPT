import pandas as pd
from tqdm import tqdm
from models.main_model import Conbergpt

def generate_estimator_dataset(data_file_path, save_path):
    df = pd.read_csv(data_file_path, index_col=0)
    toxic_text = df['toxic_text'].values
    de_toxic_text = df['de-toxic_text'].values
    init_toxisities = df['init_toxicity'].values
    detox_toxisities = df['detox_toxicity'].values

    # Create the dataframe wit 2 columns
    df = pd.DataFrame(columns=['text', 'toxicity'])


    print("Generating Estimator dataset ...")

    # fill with texts
    df['text'] = toxic_text
    df['toxicity'] = init_toxisities

    # append the detoxicated texts
    df = df.append(pd.DataFrame({'text': de_toxic_text, 'toxicity': detox_toxisities}), ignore_index=True)

    # save the dataframe
    df.to_csv(save_path)

    return None


def generate_gpt2_corpus(data_file_path, save_path, device='cpu', estimator_token=False):
    df = pd.read_csv(data_file_path, index_col=0)
    toxic_text = df['toxic_text'].values
    de_toxic_text = df['de-toxic_text'].values
    estimation = df['detox_toxicity'].values

    model = Conbergpt(device=device)
    model.construct()


    corpus = ""

    for tox_text, non_tox_text, est in tqdm(zip(toxic_text, de_toxic_text, estimation), total=len(toxic_text), desc="Generating corpus", colour="green"):

        # generate detoxicated text
        non_tox = model.detoxicate(tox_text)

        if estimator_token:
            string = f"<T>{tox_text}<NT>{non_tox}<E>{est}<F>{non_tox_text}\n"
        else:
            string = f"<T>{tox_text}<NT>{non_tox}<F>{non_tox_text}\n"

        corpus += string

    with open(save_path, 'w') as f:
        f.write(corpus)

    return None