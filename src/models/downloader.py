import zipfile
import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_and_extract_array(id, extract_path):
    out_file = f'{extract_path}temp_array.zip'
    download_file_from_google_drive(id, out_file)

    # Unzip the models
    with zipfile.ZipFile(out_file, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Delete the zip file
    os.remove(out_file)
    return


def check_files(directory_path, file_list):
    """
    Checks if a list of files exist in a directory path.
    """
    if not os.path.exists(directory_path):
        return False

    for file in file_list:
        file_path = os.path.join(directory_path, file)
        if not os.path.exists(file_path):
            return False
    return True


def download_data(download_gpt2=False, download_conbert=False, download_estimator=True, download_datasets=False):

    # Download fine-tuned GPT-2
    gpt_file_list = ['added_tokens.json',
                     'config.json',
                     'generation_config.json',
                     'merges.txt',
                     'pytorch_model.bin',
                     'special_tokens_map.json',
                     'tokenizer_config.json',
                     'vocab.json']

    if download_gpt2:
        if not check_files('models/Gpt2/', gpt_file_list):
            print('Downloading GPT-2 data...')
            download_and_extract_array('1UZ3qy_GmV2eafq8OrkIZAowMQVay3ydj', 'models/Gpt2/')

        else:
            print("GPT-2 data is already downloaded.")


    # Download ConBERT vocabs
    conbert_vocab_file_list = [ 'negative-words.txt',
                                'positive-words.txt',
                                'token_toxicities.txt',
                                'toxic_words.txt',
                                'word2coef.pkl']
    if download_conbert:
        if not check_files('models/Conbert/vocab/', conbert_vocab_file_list):
            print('Downloading Conditional BERT vocabs...')
            download_and_extract_array('1tEqOzcls_nAAyXNNNgAHo-4PIpCUbHzs', 'models/Conbert/')
        else:
            print("Conditional BERT vocabs are already downloaded.")


    # Download Estimator
    estimator_file_list = ['estimator_params.pt',
                           'vocab.pt']
    if download_estimator:
        if not check_files('models/Estimator/', estimator_file_list):
            print('Downloading Estimator...')
            download_and_extract_array('16H2AT_m3LLmL3CGHVXgsXezzNa9gUCXE', 'models/Estimator/')
        else:
            print("Estimator is already downloaded.")


    # Download all datasets (including intermediate)
    interm_datasets_list = ['estimator_dataset.csv',
                            'filtered_preprocessed.csv',
                            'gpt2_corpus.txt',
                            'train_normal_corpus.txt',
                            'train_toxic_corpus.txt']

    raw_datasets_list = ['filtered.tsv']

    if download_datasets:
        if not check_files('data/interm/', interm_datasets_list) or not check_files('data/raw/', raw_datasets_list):
            print('Downloading Datasets...')
            download_and_extract_array('1Hx6wyl4Q1JH-RHhnnySPk9_n5Rj2jRcA', 'data/')
        else:
            print("Datasets are already downloaded.")

