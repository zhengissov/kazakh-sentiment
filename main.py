import pandas as pd
import nltk
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import pandas as pd
data=pd.read_csv('kazakh_revs.csv')

stop = stopwords.words('kazakh')
wl = WordNetLemmatizer()

import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd

torch.backends.cudnn.deterministic = True


RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 200
LEARNING_RATE = 0.005
BATCH_SIZE = 128
NUM_EPOCHS = 30
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2


### Defining the feature processing

TEXT = torchtext.data.Field(
    tokenize='spacy', # default splits on whitespace
    tokenizer_language='xx_ent_wiki_sm',
    include_lengths=True # NEW
)

### Defining the label processing

LABEL = torchtext.data.LabelField(dtype=torch.long)

fields = [('review', TEXT), ('sentiment', LABEL)]

dataset = torchtext.data.TabularDataset(
    path='kazakh_revs.csv', format='csv',
    skip_header=True, fields=fields)

train_data, test_data = dataset.split(
    split_ratio=[0.8, 0.2],
    random_state=random.seed(RANDOM_SEED))

train_data, valid_data = train_data.split(
    split_ratio=[0.8, 0.12],
    random_state=random.seed(RANDOM_SEED))

TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
LABEL.build_vocab(train_data)

train_loader, valid_loader, test_loader = \
    torchtext.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True, # NEW. necessary for packed_padded_sequence
             sort_key=lambda x: len(x.review),
        device=DEVICE
)

class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, text, text_length):
        embedded = self.embedding(text)

        ## NEW
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed)

        hidden.squeeze_(0)

        output = self.fc(hidden)
        return output

torch.manual_seed(RANDOM_SEED)
model = RNN(input_dim=len(TEXT.vocab),
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES # could use 1 for binary classification
)

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for batch_idx, batch_data in enumerate(data_loader):

            # NEW
            features, text_length = batch_data.review
            targets = batch_data.sentiment.to(DEVICE)

            logits = model(features, text_length)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)

            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):

        # NEW
        features, text_length = batch_data.review
        labels = batch_data.sentiment.to(DEVICE)

        ### FORWARD AND BACK PROP
        logits = model(features, text_length)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()

        loss.backward()

        ### UPDATE MODEL PARAMETERS
        optimizer.step()

print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')

def clean_text(text):
    alphaPattern      = "[^\W\d_]"
    text = re.sub(r"[\W\d_]", " ", text)

    filtered_list = []
    stop_words = stopwords.words('kazakh')

    # my new custom stopwords
    my_extra = ['және', 'телефон', 'телефонды', 'оны', 'университет']
    # add the new custom stopwrds to my stopwords
    stop_words.extend(my_extra)
    # Tokenize the sentence
    words = word_tokenize(text)
    for w in words:
        if w.lower() not in stop_words:
            filtered_list.append(w)

    return ' '.join(filtered_list)

data_copy = data.copy()
data.head()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier

from flask import Flask, render_template
from flask import request
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    #splitting into train and test
    train, test= train_test_split(data, test_size=0.2, random_state=42)
    Xtrain, ytrain = train['review'], train['sentiment']
    Xtest, ytest = test['review'], test['sentiment']

    #Vectorizing data

    tfidf_vect = TfidfVectorizer() #tfidfVectorizer
    Xtrain_tfidf = tfidf_vect.fit_transform(Xtrain)
    Xtest_tfidf = tfidf_vect.transform(Xtest)


    count_vect = CountVectorizer() # CountVectorizer
    Xtrain_count = count_vect.fit_transform(Xtrain)
    Xtest_count = count_vect.transform(Xtest)


    # ### Logistic Regression

    # In[55]:


    lr = LogisticRegression()
    lr.fit(Xtrain_tfidf,ytrain)
    p1=lr.predict(Xtest_tfidf)
    s1=accuracy_score(ytest,p1)

    LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
    LRmodel.fit(Xtrain_tfidf, ytrain)
    text = request.form['text']



    mnb= MultinomialNB()
    mnb.fit(Xtrain_tfidf,ytrain)
    p2=mnb.predict(Xtest_tfidf)
    s2=accuracy_score(ytest,p2)



    linear_svc = LinearSVC(penalty='l2',loss = 'hinge')
    linear_svc.fit(Xtrain_tfidf,ytrain)
    p3=linear_svc.predict(Xtest_tfidf)
    s3=accuracy_score(ytest,p3)

    xgbo = XGBClassifier()
    xgbo.fit(Xtrain_tfidf,ytrain)
    p4=xgbo.predict(Xtest_tfidf)
    s4=accuracy_score(ytest,p4)

    def predict(vectoriser, model, text):
        # Predict the sentiment
        textdata = vectoriser.transform(text)
        sentiment = model.predict(textdata)

        # Make a list of text with sentiment.
        data = []
        for text, pred in zip(text, sentiment):
            data.append((text,pred))

        # Convert the list into a Pandas DataFrame.
        df = pd.DataFrame(data, columns = ['text','sentiment'])
        df = df.replace([0,1], ["Negative","Positive"])
        return df

    arr = [text]
    logDf = predict(tfidf_vect, lr, arr)
    mnbDf = predict(tfidf_vect, mnb, arr)
    linearSvcDf = predict(tfidf_vect, linear_svc, arr)
    xgboDf = predict(tfidf_vect, xgbo, arr)

    lrSentiment = logDf['sentiment'].values[0]
    mnbSentiment = mnbDf['sentiment'].values[0]
    linearSvcSentiment = linearSvcDf['sentiment'].values[0]
    xgboDfSentiment = xgboDf['sentiment'].values[0]


    import spacy

    nlp = spacy.load("xx_ent_wiki_sm")

    def predict2(model, sentence):

        model.eval()

        with torch.no_grad():
            tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
            indexed = [TEXT.vocab.stoi[t] for t in tokenized]
            length = [len(indexed)]
            tensor = torch.LongTensor(indexed).to(DEVICE)
            tensor = tensor.unsqueeze(1)
            length_tensor = torch.LongTensor(length)
            predict_probas = torch.nn.functional.softmax(model(tensor, length_tensor), dim=1)
            predicted_label_index = torch.argmax(predict_probas)
            predicted_label_proba = torch.max(predict_probas)
            return predicted_label_index.item(), predicted_label_proba.item()


    class_mapping = LABEL.vocab.stoi
    inverse_class_mapping = {v: k for k, v in class_mapping.items()}

    predicted_label_index, predicted_label_proba = \
        predict2(model, text)
    predicted_label = inverse_class_mapping[predicted_label_index]

    print(f'Predicted label index: {predicted_label_index}'
          f' | Predicted label: {predicted_label}'
          f' | Probability: {predicted_label_proba} ')

    return(render_template('index.html', lrSentiment=lrSentiment, mnbSentiment=mnbSentiment, linearSvcSentiment=linearSvcSentiment, xgboDfSentiment=xgboDfSentiment, predicted_label=predicted_label, predicted_label_proba=predicted_label_proba))

if __name__ == "__main__":
    app.run(debug=True)

