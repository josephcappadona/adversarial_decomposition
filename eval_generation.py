# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: venv
# ---

# +
import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from settings import EXPERIMENTS_DIR
from experiment import Experiment
from utils import to_device, load_weights, load_embeddings, create_embeddings_matrix
from vocab import Vocab
from train import create_model
from preprocess import load_dataset, create_dataset_reader
from sys import argv
# -

exp_id = f"train.{argv[1] if len(argv) > 1 else '0d1b9t6u'}"

# # Load everything

exp = Experiment.load(EXPERIMENTS_DIR, exp_id)

exp.config

preprocess_exp = Experiment.load(EXPERIMENTS_DIR, exp.config.preprocess_exp_id)
dataset_train, dataset_val, dataset_test, vocab, style_vocab, W_emb = load_dataset(preprocess_exp)

dataset_reader = create_dataset_reader(preprocess_exp.config)

model = create_model(exp.config, vocab, style_vocab, dataset_train.max_len, W_emb)

load_weights(model, exp.experiment_dir.joinpath('best.th'))

model = model.eval()


# ## Predict

def create_inputs(instances):
    if not isinstance(instances, list):
        instances = [instances,]
        
    if not isinstance(instances[0], dict):
        sentences = [
            dataset_reader.preprocess_sentence(dataset_reader.spacy( dataset_reader.clean_sentence(sent)))
            for sent in instances
        ]
        
        style = list(style_vocab.token2id.keys())[0]
        instances = [
            {
                'sentence': sent,
                'style': style,
            }
            for sent in sentences
        ]
        
        for inst in instances:
            inst_encoded = dataset_train.encode_instance(inst)
            inst.update(inst_encoded)            
    
    
    instances = [
        {
            'sentence': inst['sentence_enc'],
            'style': inst['style_enc'],
        } 
        for inst in instances
    ]
    
    instances = default_collate(instances)
    instances = to_device(instances)      
    
    return instances


def get_sentences(outputs):
    predicted_indices = outputs["predictions"]
    end_idx = vocab[Vocab.END_TOKEN]
    
    if not isinstance(predicted_indices, np.ndarray):
        predicted_indices = predicted_indices.detach().cpu().numpy()

    all_predicted_tokens = []
    for indices in predicted_indices:
        indices = list(indices)

        # Collect indices till the first end_symbol
        if end_idx in indices:
            indices = indices[:indices.index(end_idx)]

        predicted_tokens = [vocab.id2token[x] for x in indices]
        all_predicted_tokens.append(predicted_tokens)
        
    return all_predicted_tokens


sentence =  ' '.join(dataset_val.instances[1]['sentence'])

sentence

inputs = create_inputs(sentence)

outputs = model(inputs)

sentences = get_sentences(outputs)

' '.join(sentences[0])

# ### Swap style

possible_styles = list(style_vocab.token2id.keys()) #['negative', 'positive']

possible_styles

sentences0 = [s for s in dataset_val.instances if s['style'] == possible_styles[0]]
sentences1 = [s for s in dataset_val.instances if s['style'] == possible_styles[1]]

for i in np.random.choice(np.arange(len(sentences0)), 5):
    print(i, ' '.join(sentences0[i]['sentence']))

for i in np.random.choice(np.arange(len(sentences1)), 5):
    print(i, ' '.join(sentences1[i]['sentence']))

# #### Swap

target0 = np.random.choice(np.arange(len(sentences0)))
target1 = np.random.choice(np.arange(len(sentences0)))

print(' '.join(sentences0[target0]['sentence']))

print(' '.join(sentences1[target1]['sentence']))

inputs = create_inputs([
    sentences0[target0],
    sentences1[target1],
])

z_hidden = model(inputs)

z_hidden['style_hidden'].shape

z_hidden['meaning_hidden'].shape

original_decoded = model.decode(z_hidden)

original_sentences = get_sentences(original_decoded)

print(' '.join(original_sentences[0]))
print(' '.join(original_sentences[1]))

z_hidden_swapped = {
    'meaning_hidden': torch.stack([
        z_hidden['meaning_hidden'][0].clone(),
        z_hidden['meaning_hidden'][1].clone(),        
    ], dim=0),
    'style_hidden': torch.stack([
        z_hidden['style_hidden'][1].clone(),
        z_hidden['style_hidden'][0].clone(),        
    ], dim=0),
}

swaped_decoded = model.decode(z_hidden_swapped)

swaped_sentences = get_sentences(swaped_decoded)

print(' '.join(original_sentences[0]))
print(' '.join(original_sentences[1]))
print()
print(' '.join(swaped_sentences[0]))
print(' '.join(swaped_sentences[1]))




