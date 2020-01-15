import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding

def text_clean(sent):
    sent = str(sent)
    sent = sent.lower()
    sent = re.sub(r'[\W\d]', " ", sent)
    tokens = sent.split(" ")
    cleaned_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(cleaned_words)

input_data = pd.read_excel("spam_msg.xlsx")
input_data.head()

input_data['Spam_label'] = input_data['Spam_label'].replace({'spam':1, 'non-spam':0})

#print(input_data.head())

stoplist = stopwords.words('english')

input_data['cleanText']=input_data['Text'].map(lambda sent : text_clean(sent))

#print(input_data.head(11))

corpus = input_data['cleanText'].tolist()

response = np.array(input_data['Spam_label'])
#print(response)

vocab_size=200
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)

#print(sequences)

max_length=20
padded_corpus = pad_sequences(sequences, maxlen=20, padding='post')

#print(padded_corpus)

embeddings_index = dict()
f = open('glove.6B.50d.txt',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#print('glove word vectors %s loaded' % len(embeddings_index))
#print(embeddings_index['happy'])
#print(embeddings_index['science'])

embedding_dim = 50
vocab_size = 200

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#print(embedding_matrix.shape)
#print(tokenizer.word_index.items())
#print(embedding_matrix[1])

model = Sequential()
model.add(Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False))  # 40 words with each dim of 5o dim vecotors (param = 2000)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_corpus, response, epochs=10, verbose=1)

#print(padded_corpus[0:1])

model.predict_classes(padded_corpus[0:1])
model.predict(padded_corpus[0:1])
model.predict_classes(padded_corpus[0:1])

#print(model.layers[0].get_weights()[0][1])
#print(model.layers[0].get_weights()[0][39])