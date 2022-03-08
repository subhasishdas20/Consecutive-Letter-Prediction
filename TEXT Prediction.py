#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import string
import pickle
import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings


# In[28]:


text =['Bilbo Baggins lives a quiet, peaceful life in his comfortable hole at Bag End',
       'Bilbo lives in a hole because he is a hobbit—one of a race of small plump people about half the size of humans with furry toes and a great love of good food and drink',
       'Bilbo is quite content at Bag End, near the bustling hobbit village of Hobbiton but one day his comfort is shattered by the arrival of the old wizard Gandalf, who persuades Bilbo to set out on an adventure with a group of thirteen militant dwarves',
       'The dwarves are embarking on a great quest to reclaim their treasure from the marauding dragon Smaug and Bilbo is to act as their burglar', 
       'The dwarves are very skeptical about Gandalf’s choice for a burglar, and Bilbo is terrified to leave his comfortable life to seek adventure',
       'But Gandalf assures both Bilbo and the dwarves that there is more to the little hobbit than meets the eye']


# In[29]:


len(text)


# # Unique Words

# In[30]:


letters = set("". join(text))


# In[31]:


len(letters)


# In[32]:


ws = WordNetLemmatizer()


# In[33]:


def process_word(data):
    lemma_word = []
    for i in range(len(data)):
        data[i] = data[i].lower()
        word_token = word_tokenize(data[i])
        clean_data = [i for i in word_token if not i in stopwords.words() and i.isalnum()]
        
        a = []
        for i in clean_data:
            a.append(ws.lemmatize(i))
        lemma_word.append(a)
    return lemma_word     


# In[34]:


process_word(text)


# ## Encoding
# 

# In[35]:


int_to_words = dict(enumerate(letters))
words_to_int = {w : i for i,w in int_to_words.items()}


# In[37]:


int_to_words[33]


# In[41]:


words_to_int ['p']


# In[42]:


max_len_text= len(max(text, key=len))


# In[43]:


max_len_text


# ## Padding

# In[44]:


for i in range (len(text)):
    while len(text[i])< max_len_text:
        text[i]+= " "
    


# In[45]:


text                 #Post Padding 


# ## Creating Input & Output Sequence

# In[46]:


input_seq = []
target_seq = []

for i in range (len(text)):
    input_seq.append(text[i][:-1])       # Removing the last word from the input sequence 
    
    
    target_seq.append(text[i][1:])           # Removing the first word from the  target sequence
    
    print("Input Sequence: {}, \nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
    


# In[47]:


input_seq


# In[48]:


target_seq


# In[49]:


for i in range (len(text)):
    input_seq[i] = [words_to_int[char] for char in input_seq[i]]
    target_seq[i] = [words_to_int[char] for char in target_seq[i]]


# ## One hot encoding
# 

# In[50]:


dict_size = len(words_to_int)
seq_len = max_len_text-1
batch_size =len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    features = np.zeros((batch_size,seq_len, dict_size), dtype=np.float32)
    
    for i in range(batch_size):
        for x in range(seq_len):
            features[i,x, sequence[i][x]] = 1
    return features


# In[51]:


input_seq= one_hot_encode(input_seq,dict_size, seq_len, batch_size)   #converting input seq into one hot vector


# In[52]:


input_seq


# In[53]:


input_seq.shape


# In[54]:


input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)


# ## RNN MODEL

# In[55]:


class RNNModel(nn.Module):
    
    
    def __init__(self,input_size, output_size, hidden_dim, n_layers):
        super(RNNModel,self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        
        #Linear 
        self.fc = nn.Linear(hidden_dim,output_size)

    
    
    def forward(self,z):
        batch_size = z.size(0)
        
        #Initializing hidden state for first input using method below
        hidden = self.init_hidden(batch_size)
        # Passing the input & hidden state into the model & obtaining outputs
        out, hidden = self.rnn(z,hidden)
        
        # Reshapping the outputs so that it can fit into the  fc layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self,batch_size): 
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


# In[56]:


model = RNNModel(input_size=dict_size, output_size=dict_size, hidden_dim=15, n_layers=1)
loss_fn = nn.CrossEntropyLoss()                                                    #Define Loss, Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ## Training Model

# In[59]:


epochs = 2000

for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    output,hidden = model(input_seq)
    loss = loss_fn(output, target_seq.view(-1).long())
    loss.backward()
    optimizer.step()
    
    if epoch%50 == 0:
        print('Epoch:{}/{}.............'.format(epoch, epochs), end= ' ')
        print("Loss:  {:4f}".format(loss.item()))


# In[71]:


def predict(model,char):
    char =np.array([[words_to_int[i] for i in char]])
    char = one_hot_encode(char, dict_size, char.shape[1],1)
    char = torch.from_numpy(char)
    
    out, hidden = model(char)
    
    
    prob = nn.functional.softmax(out[-1], dim=0).data
    outcome = torch.max(prob, dim=0)[1].item()
    
    return int_to_words[outcome], hidden


# In[72]:


def sample(model, out_len, start='Bilbo'):
    model.eval()
    start=start.lower()
    chars = [i for i in start]
    size = out_len - len(chars)
    
    for i in range(size):
        char,h= predict(model,chars)
        chars.append(char)
    return ''.join(chars)


# In[79]:


sample(model, out_len=23, start='bustling')


# In[82]:


sample(model, out_len=20, start='peaceful')

