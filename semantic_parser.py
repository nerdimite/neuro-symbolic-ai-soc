import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch import nn
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
from tqdm.notebook import tqdm

class Preprocessor():
    '''Preprocessor for preparing Queries and Programs for Seq2Seq'''
    def __init__(self, train_csv):
        self.spacy_en = spacy.load("en_core_web_sm")
        
        # Create fields
        self.que_f = Field(tokenize=self.tokenizer, use_vocab=True, init_token="<sos>", eos_token="<eos>", lower=True)
        self.prog_f = Field(tokenize=self.tokenizer, use_vocab=True, init_token="<sos>", eos_token="<eos>", lower=True)
        
        # Preprocess
        self.train_data = self.preprocess(train_csv)
        
    def tokenizer(self, text):
        tokens = [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

        updated_tokens = []
        for i, tok in enumerate(tokens):
            if tok == '<':
                updated_tokens.append(''.join(tokens[i:i+3]))
            elif tok in ['nxt', '>']:
                continue
            else:
                updated_tokens.append(tok)

        return updated_tokens
    
    def preprocess(self, train_csv):
        '''Returns the Dataset'''
        
        # Create the fields
        self.fields = {'query_text': ('query', self.que_f), 'program_text': ('program', self.prog_f)}
        
        # Create dataset object
        train_data = TabularDataset.splits(path="./", 
                                           train=train_csv, 
                                           format="csv", 
                                           fields=self.fields)[0]
        
        # Build vocabulary
        self.que_f.build_vocab(train_data, max_size=100, min_freq=1)
        self.prog_f.build_vocab(train_data, max_size=100, min_freq=1, specials=['<nxt>'])
        
        return train_data
    

class Seq2Seq(nn.Module):
    '''Sequence to Sequence Model using Transformers'''
    def __init__(self,
                 config,
                 device=None):
        super(Seq2Seq, self).__init__()
        '''Initialize the model'''
        
        # Create word embedding layers
        self.src_word_embedding = nn.Embedding(config['que_vocab_size'], config['embedding_dim'])
        self.trg_word_embedding = nn.Embedding(config['prog_vocab_size'], config['embedding_dim'])
        
        # Create positional embedding layers
        self.src_position_embedding = nn.Embedding(config['max_len'], config['embedding_dim'])
        self.trg_position_embedding = nn.Embedding(config['max_len'], config['embedding_dim'])
        
        # Create transformer
        self.transformer = nn.Transformer(config['embedding_dim'],
                                          config['num_heads'],
                                          config['num_encoder_layers'],
                                          config['num_decoder_layers'],
                                          config['forward_expansion'],
                                          config['dropout'])
        
        # Feedforward for Logits over vocabulary
        self.fc_out = nn.Linear(config['embedding_dim'], config['prog_vocab_size'])
        self.dropout = nn.Dropout(config['dropout'])
        
        self.src_pad_idx = config['que_pad_idx']
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Mount to device
        self.to(device)
        
    def make_src_mask(self, src):
        '''Create padding mask for src sequence'''
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask.to(self.device)
    
    def forward(self, src, trg):
        '''Forward pass'''
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape
        
        # Create positions
        src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)
        
        # Get embeddings
        src_embeds = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        trg_embeds = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )
        
        # Create masks
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        
        # Pass everything through the transformer
        out = self.transformer(src_embeds, 
                               trg_embeds, 
                               src_key_padding_mask=src_padding_mask,
                               tgt_mask=trg_mask)
        
        # Get logits over the vocabulary with a linear layer
        out = self.fc_out(out)
        # output shape (trg_seq_len, N, trg_vocab_size)
        return out
    
    def train_model(self, 
                    train_loader,
                    num_epochs,
                    num_steps,
                    lr=3e-4,
                    filename='semantic_parser.pth'):
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               factor=0.1, 
                                                               patience=10,
                                                               verbose=True)
        criterion = nn.CrossEntropyLoss(ignore_index=self.src_pad_idx)
        
        for epoch in range(num_epochs):
            pbar = tqdm(total=num_steps, desc='Epoch {}'.format(epoch+1))
            
            self.train()
            losses = []
            
            for i, batch in enumerate(train_loader):
                # Get the inputs and targets
                inp_seq, target = batch.query, batch.program

                # Forward pass
                output = self(inp_seq, target[:-1, :])

                # Reshape the output and targets for criterion
                output = output.reshape(-1, output.shape[2]) # (trg_seq_len * N, trg_vocab_size)
                target = target[1:].reshape(-1)

                optimizer.zero_grad()

                # Calculate Loss
                loss = criterion(output, target)
                losses.append(loss.item())

                # Backprop and Optimize
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
                
                # Metrics
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})

            mean_loss = sum(losses) / len(losses)
            scheduler.step(mean_loss)

            print(f'Epoch {epoch+1}: Mean Loss = {mean_loss}\n')
            pbar.close()
        
        # Save Model
        torch.save(self.state_dict(), filename)
    
    
class SemanticParser():
    '''Full Pipeline for Semantic Parsing from Query -> Program'''
    def __init__(self, preprocessor, config, filename='models/semantic_parser.pth', device=None, max_len=20):
        # Device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Fields (includes the vocabulary)
        self.que_f = preprocessor.que_f
        self.prog_f = preprocessor.prog_f
        self.preproc = preprocessor
        
        # Maximum length of program
        self.max_len = max_len
        
        # Load Model
        self.model = Seq2Seq(config, device)
        self.model.load_state_dict(torch.load(filename))
        
        
    def predict(self, query):
        '''Predicts the Program, given a query'''
        # Tokenize
        tokens = self.preproc.tokenizer(query)

        # Add <sos> and <eos> in beginning and end respectively
        tokens.insert(0, self.que_f.init_token)
        tokens.append(self.que_f.eos_token)

        # Convert the tokenized sequence into integers
        query_indices = [self.que_f.vocab.stoi[tok] for tok in tokens]

        # Convert to Tensor
        query_tensor = torch.LongTensor(query_indices).unsqueeze(1).to(self.device)
        
        # Init the program sequence with <sos>
        outputs = [self.prog_f.vocab.stoi["<sos>"]]
        
        # Generating the program
        for i in range(self.max_len):
            
            # Create program output tensor
            program_tensor = torch.LongTensor(outputs).unsqueeze(1).to(self.device)
            
            # Predict next token
            with torch.no_grad():
                output = self.model(query_tensor, program_tensor)

            # Get the word with highest probability
            word_idx = output.argmax(2)[-1, :].item()
            # Append to outputs
            outputs.append(word_idx)

            if word_idx == self.prog_f.vocab.stoi["<eos>"]:
                break

        # Decode to english
        program = [self.prog_f.vocab.itos[idx] for idx in outputs][1:-1]
        # Convert to list of instructions
        program_ = ' '.join(program).split(' <nxt> ')

        return program_