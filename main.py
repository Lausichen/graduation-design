#import torch
import numpy as np
import sys
import time
import random
import datetime
from torch.autograd import Variable
from torch import nn
from model import Encoder, Decoder, Attention, Seq2Seq
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(datetime.datetime.now())

size_vocabulary = 20000
size_embedding = 100
size_layer = 512
num_layer = 4
batch_size = 50
embedding_dim = 100
hidden_dim = 100
batch_size = 256
clip = 5
data_dir = "./data"
train_dir = "./train"
pmi_path = "./data/PMI"

Special_Vocab = ["_PAD", "_EOS", "_SOS", "_UNK"]


def load_data(path, name):
    with open("{0}/{1}.post".format(path, name), encoding="utf-8") as file:
        post = [line.strip().split() for line in file.readlines()]
    with open("{0}/{1}.response".format(path, name), encoding="utf-8") as file:
        question = [line.strip().split() for line in file.readlines()]
    data = []
    for p, q in zip(post, question):
        data.append({'post': p, 'question': q})
    return data


def build_up_dictionary(vocab):
    word2index = {}
    index2word = []
    count = 0
    for word in vocab:
        if word not in word2index:
            word2index[word] = count
            index2word.append(word)
    return word2index, index2word


def build_vocab(path, data):
    file = open("question_words.txt", "r", encoding="utf-8")
    question_words = file.readline().strip().split('|')
    file.close()
    vocab = {}
    for pair in data:
        for word in pair["post"]+pair["question"]:
            try:
                vocab[word] += 1
            except:
                vocab[word] = 1
    vocab_list = Special_Vocab + question_words + sorted(vocab.items(), key=lambda vocab:vocab[1], reverse=True)
    '''if len(vocab_list) > size_vocabulary:
        vocab_list = vocab_list[:size_vocabulary]'''
    word2index, index2word = build_up_dictionary(vocab_list)
    vectors = {}
    with open("{0}/vector.txt".format(path), "r", encoding="utf-8") as file:
        for line in file:
            sline = line.strip()
            word = sline[:sline.find(" ")]
            vector = sline[sline.find(" ") + 1:]
            vectors[word] = vector
    embedding = []
    for word in vocab_list:
        if word in vectors:
            vector = map(float, vectors[word])
        else:
            vector = np.zeros((size_embedding), dtype=np.float32)
        embedding.append(vector)
    embedding = np.array(embedding, dtype=np.float32)
    return vocab_list, word2index, index2word, embedding, question_words


def load_PMI():
    keywords_list = []
    with open("{0}/all.txt".format(pmi_path), "r", encoding="utf-8") as file:
        for line in file:
            keywords_list.append(line.strip())
    keywords_index = {}
    for i in range(len(keywords_list)):
        keywords_index[keywords_list[i]] = i
    PMI = []
    with open("{0}/all_PMI.txt".format(pmi_path), "r", encoding="utf-8") as file:
        for line in file:
            linePMI = map(float, line.strip().split())
            PMI.append(linePMI)
    return keywords_list, keywords_index, PMI


def batch_data(data, question_words, keywords_index, key_to_vocab, word2index):
    def padding(sentense, length):
        return sentense + ["_EOS"] + ["_PAD"] * (length-len(sentense)-1)
    encoder_len = max([len(pair["post"]) for pair in data]) + 1
    decoder_len = max([len(pair["question"]) for pair in data]) + 1
    posts = []
    questions = []
    posts_length = []
    questions_length = []
    keywords_type = np.zeros((3, size_vocabulary))
    keyword_tensor = []
    word_type = []
    for i in range(4):
        keywords_type[1][i] = 1
    for i in range(4, 4 + len(question_words)):
        keywords_type[0][i] = 1
    for i in range(len(question_words) + 4, size_vocabulary):
        keywords_type[1][i] = 1
    for pair in data:
        for word in pair["question"]:
            if word in keywords_index.key(word):
                keywords_type[1][key_to_vocab[keywords_index[word]]] = 0
                keywords_type[2][key_to_vocab[keywords_index[word]]] = 1
                word_type.append(2)
            elif word in question_words:
                word_type.append(0)
            else:
                word_type.append(1)
        for i in range(decoder_len - len(pair["response"])):
            word_type.append(1)
        posts.append(padding(pair["post"], encoder_len))
        questions.append(padding(pair["questions"], decoder_len))
        posts_length.append(len(pair["post"])+1)
        questions_length.append(len(pair["response"])+1)
        for i in range(decoder_len):
            keyword_tensor.append(keywords_type)
    posts_index = []
    questions_index = []
    length_pq = len(posts)
    for post in length_pq:
        post_index = []
        for word in post:
            index = word2index[word]
            posts_index.append(index)
        posts_index.append(post_index)
    for post in length_pq:
        question_index = []
        for word in post:
            index = word2index[word]
            questions_index.append(index)
        posts_index.append(question_index)
    posts_index = torch.LongTensor(posts_index).transpose(0, 1).to(device)
    questions_index = torch.LongTensor(questions_index).transpose(0, 1).to(device)
    keyword_tensor = torch.LongTensor(keyword_tensor).to(device)
    word_type = torch.LongTensor(word_type).to(device)
    return posts_index, questions_index, keyword_tensor, word_type


def random_batch(batch_size, data):
    batched_data = []
    for i in random.choice(len(data), batch_size):
        batched_data.append(data[i])
    return batched_data


data = load_data(data_dir, "weibo_pair_train_Q_after")
vocab_list, word2index, index2word, embedding, question_words = build_vocab(data_dir, data)
keywords_list, keywords_index, PMI = load_PMI()
key_to_vocab = [0] * len(keywords_list)
for i in range(len(keywords_list)):
    if keywords_list[i] in word2index:
        key_to_vocab[i] = word2index[keywords_list[i]]

encoder = Encoder(len(word2index), embedding_dim, hidden_dim)
decoder = Decoder(len(word2index), embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.SGD(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

def train():
    model.train()
    for batch_id in range(0, 100):
        batched_data = random_batch(batch_size, data)
        posts_index, questions_index, keyword_tensor, word_type = batch_data(batched_data, question_words, keywords_index, key_to_vocab, word2index)
        optimizer.zero_grad()
        output = model(posts_index, questions_index)
        loss = criterion(output.view(-1, output.shape[2]), questions_index.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if batch_id % 100 == 0:
            print('current loss: {:.4f}'.format(loss))
    torch.save(model, '/model.pkl')


def words_to_index(words, word2index):
    indices = []
    for word in words:
        index = word2index[word]
        indices.append(index)
    return indices


def evaluate(model, input_seq, word2index, trg_ix2c):
    model.eval()
    with torch.no_grad():
        seq = torch.LongTensor(words_to_index(input_seq, word2index)).view(-1, 1).to(device)
        outputs, attn_weights = model.predict(seq, [seq.size(0)], max_trg_len)
        outputs = outputs.squeeze(1).cpu().numpy()
        attn_weights = attn_weights.squeeze(1).cpu().numpy()
        output_words = [trg_ix2c[np.argmax(word_prob)] for word_prob in outputs]
        show_attention(list('^' + text + '$'), output_words, attn_weights)'''