#!/usr/bin/env python
# coding: utf-8

# ### Dataset Information
# 
# Phone dataset contains 1400+ reviews for natural language processing. The dataset contains two columns - review and sentiment to perform the sentimental analysis.It was collected from "Kaspi Магазин", Amazon, Samsung phone reviews
# 
# ### Problem Statement
# For phone reviews, correctly categorize positive and negative reviews.
# 
# ### Overview
# Performed cleaning on the dataset. As a learning modals used Logistic Regression, Multinomial Naive Bayes, Linear SVM and XGBoost, LSTM and BERT. The maximum accuracy achieved using BERT is around 89%.

# # **1. Importing Libraries**

# In[1]:


print('qwe')


# In[4]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud,STOPWORDS
from bs4 import BeautifulSoup
import re,string,unicodedata

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from xgboost.sklearn import XGBClassifier

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import transformers
import tokenizers

#     "жақсы": 3,
#     "жақсылық": 3,
#     "жақсырақ": 2,
#     "жақтастары": 1,
#     "жақтаушылары": 2,
#     "жақтаушысы": 1,
#     "жақындастыруға": 2,
#     "жалған": -1,
#     "жалғыз": -2,
#     "жалқау": -1,
#     "жалтару": -2,
#     "жалықтырған": -3,
#     "жалықтыру": -2,
#     "жаман": -3,
#     "жаман сәттілік": -2,
#     "жанама әсері": -2,
#     "жанама әсерлері": -2,
#     "жанашыр": 2,
#     "жанданған": 3,
#     "жандандырады": 2,
#     "жанды": 3,
#     "жанжал": -3,
#     "жанжалдар": -3,
#     "жанжалды": -3,
#     "жанкештілік": -2,


# In[5]:


import tensorflow as tf
print('tf')


# # **2. Data Extraction and Cleaning**

# In[21]:


import pandas as pd
import nltk


# In[22]:


import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
nltk.download('punkt')


# In[23]:


import pandas as pd
data=pd.read_csv('kazakh_revs.csv')
data.head(10)


# In[24]:


data.info()


# In[25]:


data.describe() #descriptive statistics


# In[26]:


null_values = data.isnull().sum() #identifying missing values


# In[27]:


null_values.index[0]


# In[28]:


print('There are {} missing values for {} and {} missing values for {}.'.format(null_values[0],null_values.index[0],null_values[1],null_values.index[1]))


# In[29]:


num_duplicates = data.duplicated().sum() #identify duplicates
print('There are {} duplicate reviews present in the dataset'.format(num_duplicates))


# In[30]:


#view duplicate reviews
review = data['review']
duplicated_review = data[review.isin(review[review.duplicated()])].sort_values("review")
duplicated_review.head()


# In[31]:


#drop duplicate reviews
data.drop_duplicates(inplace = True)


# In[32]:


stop = stopwords.words('kazakh')
wl = WordNetLemmatizer()


# In[33]:


def clean_text(text):
    alphaPattern      = "[^\W\d_]"
    text = re.sub(r"[\W\d_]", " ", text)
        
    filtered_list = []
    stop_words = stopwords.words('kazakh')
    
    # my new custom stopwords
    my_extra = ['және', 'телефон', 'телефонды', 'оны']
    # add the new custom stopwrds to my stopwords
    stop_words.extend(my_extra)
    # Tokenize the sentence
    words = word_tokenize(text)
    for w in words:
        if w.lower() not in stop_words:
            filtered_list.append(w)

    return ' '.join(filtered_list)

clean_text('Телефон жақсы екен. Және ұнады')


# In[34]:


data_copy = data.copy()
data.head()


# In[35]:


stop_words = stopwords.words('kazakh')
print(stop_words)


# In[36]:


data['review']=data['review'].apply(clean_text)


# In[37]:


#converting target variable to numeric labels
# data.sentiment = [ 1 if each == "positive" else 0 for each in data.sentiment]


# In[38]:


#after converting labels
data.head()


# # **3. Exploratory data analysis** 

# In[39]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[40]:


### Count Plot
sns.set(style = "whitegrid" , font_scale = 1.2)
sns.countplot(data.sentiment,palette = ['green','red'],order = [1,0])
plt.xticks(ticks = np.arange(2),labels = ['positive','negative'])
plt.title('Target count for phone reviews')
plt.show()


# In[41]:


print('Positive reviews are', (round(data['sentiment'].value_counts()[1])),'i.e.', round(data['sentiment'].value_counts()[1]/len(data) * 100,2), '% of the dataset')
print('Negative reviews are', (round(data['sentiment'].value_counts()[0])),'i.e.',round(data['sentiment'].value_counts()[0]/len(data) * 100,2), '% of the dataset')


# In[ ]:


#word cloud for positive reviews
positive_data = data[data.sentiment == 1]['review']
positive_data_string = ' '.join(positive_data)
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width=1200, height=600,background_color="white").generate(positive_data_string)
plt.imshow(wc)
plt.axis('off')
plt.title('Word cloud for positive reviews',fontsize = 20)
plt.show()


# In[ ]:


#word cloud for negative reviews
negative_data = data[data.sentiment == 0]['review']
negative_data_string = ' '.join(negative_data)
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width=1200, height=600,background_color="white").generate(negative_data_string)
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.title('Word cloud for negative reviews',fontsize = 20)
plt.show()


# In[259]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,8))

text_len=positive_data.str.split().map(lambda x: len(x))
ax1.hist(text_len,color='green')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('Number of Words')
ax1.set_ylabel('Count')
text_len=negative_data.str.split().map(lambda x: len(x))
ax2.hist(text_len,color='red')
ax2.set_title('Negative Reviews')
ax2.set_xlabel('Number of Words')
ax2.set_ylabel('Count')
fig.suptitle('Number of words in texts')
plt.show()


# In[17]:


# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
# word = positive_data.str.split().apply(lambda x : len(x) )
# sns.distplot(word, ax=ax1,color='green')
# ax1.set_title('Positive Reviews')
# ax1.set_xlabel('Number of words per review')
# word = negative_data.str.split().apply(lambda x :len(x) )
# sns.distplot(word,ax=ax2,color='red')
# ax2.set_title('Negative Reviews')
# ax2.set_xlabel('Number of words per review')
# fig.suptitle('Distribution of number of words per reviews')
# plt.show()


# In[18]:


# fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,10))
# word = positive_data.str.split().apply(lambda x : [len(i) for i in x] )
# sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1,color='green')
# ax1.set_title('Positive Reviews')
# ax1.set_xlabel('Average word length per review')
# word = negative_data.str.split().apply(lambda x : [len(i) for i in x] )
# sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
# ax2.set_title('Negative Reviews')
# ax2.set_xlabel('Average word length per review')
# fig.suptitle('Distribution of average word length in each review')
# plt.show()


# In[ ]:


def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words
corpus = get_corpus(data.review)
corpus[:5]


# In[ ]:


from collections import Counter
counter = Counter(corpus)
most_common = counter.most_common(10)
most_common = pd.DataFrame(most_common,columns = ['corpus','countv'])
most_common


# In[46]:


most_common = most_common.sort_values('countv')


# In[47]:


plt.figure(figsize =(10,10))
plt.yticks(range(len(most_common)), list(most_common.corpus))
plt.barh(range(len(most_common)), list(most_common.countv),align='center',color = 'blue')
plt.title('Most common words in the dataset')
plt.show()


# In[48]:


def get_ngrams(review, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(review)
    bag_of_words = vec.transform(review) #sparse matrix of count_vectorizer
    sum_words = bag_of_words.sum(axis=0) #total number of words
    sum_words = np.array(sum_words)[0].tolist() #convert to list
    words_freq = [(word, sum_words[idx]) for word, idx in vec.vocabulary_.items()] #get word freqency for word location in count vec
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True) #key is used to perform sorting using word_freqency 
    return words_freq[:n]


# In[49]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(30,15))
uni_positive = get_ngrams(positive_data,20,1)
uni_positive = dict(uni_positive)
temp = pd.DataFrame(list(uni_positive.items()), columns = ["Common_words" , 'Count'])
sns.barplot(data = temp, x="Count", y="Common_words", orient='h',ax = ax1)
ax1.set_title('Positive reviews')
uni_negative = get_ngrams(negative_data,20,1)
uni_negative = dict(uni_negative)
temp = pd.DataFrame(list(uni_negative.items()), columns = ["Common_words" , 'Count'])
sns.barplot(data = temp, x="Count", y="Common_words", orient='h',ax = ax2)
ax2.set_title('Negative reviews')
fig.suptitle('Unigram analysis for positive and negative reviews')
plt.show()


# In[50]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(30,15))
bi_positive = get_ngrams(positive_data,20,2)
bi_positive = dict(bi_positive)
temp = pd.DataFrame(list(bi_positive.items()), columns = ["Common_words" , 'Count'])
sns.barplot(data = temp, x="Count", y="Common_words", orient='h',ax = ax1)
ax1.set_title('Positive reviews')
bi_negative = get_ngrams(negative_data,20,2)
bi_negative = dict(bi_negative)
temp = pd.DataFrame(list(bi_negative.items()), columns = ["Common_words" , 'Count'])
sns.barplot(data = temp, x="Count", y="Common_words", orient='h',ax = ax2)
ax2.set_title('Negative reviews')
fig.suptitle('Bigram analysis for positive and negative reviews')
plt.show()


# In[51]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(30,15))
tri_positive = get_ngrams(positive_data,20,3)
tri_positive = dict(tri_positive)
temp = pd.DataFrame(list(tri_positive.items()), columns = ["Common_words" , 'Count'])
sns.barplot(data = temp, x="Count", y="Common_words", orient='h',ax = ax1)
ax1.set_title('Positive reviews')
tri_negative = get_ngrams(negative_data,20,3)
tri_negative = dict(tri_negative)
temp = pd.DataFrame(list(tri_negative.items()), columns = ["Common_words" , 'Count'])
sns.barplot(data = temp, x="Count", y="Common_words", orient='h',ax = ax2)
ax2.set_title('Negative reviews')
fig.suptitle('Trigram analysis for positive and negative reviews')
plt.show()


# # **4. Predictive Modelling using Machine Learning** 

# In[52]:


# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from bs4 import BeautifulSoup
# import re,string,unicodedata

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
# from xgboost.sklearn import XGBClassifier

# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
# import transformers
# import tokenizers

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier


# In[53]:


#splitting into train and test
train, test= train_test_split(data, test_size=0.2, random_state=42)
Xtrain, ytrain = train['review'], train['sentiment']
Xtest, ytest = test['review'], test['sentiment']
print(train, test)


# In[54]:


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
print(classification_report(ytest, p1))
print("Logistic Regression Accuracy :", "{:.2f}%".format(100*s1))
plot_confusion_matrix(lr, Xtest_tfidf, ytest,cmap = 'Blues')
plt.grid(False)

LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(Xtrain_tfidf, ytrain)
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform((text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df


# In[56]:


text = ["Маған ұнамады, қатып қала береді",
        "Байланыс ұстамайды, нашар",
        "Маған ұнамады",
        "Камерасы онша емес, телефон соғылмайды"]

df = predict(tfidf_vect, lr, text)
print(df)

import pickle
pickle.dump(lr, open('model.pkl', 'wb'))


# In[36]:


text = ["Жақсы телефон",
        "Камерасы мықты, керемет",
        "Тамаша телефон, қолдануға оңай",
        "Өте тез жұмыс жасайды"]

df = predict(tfidf_vect, lr, text)
print(df)


# ### Multinomial Naive Bayes

# In[37]:


mnb= MultinomialNB()
mnb.fit(Xtrain_tfidf,ytrain)
p2=mnb.predict(Xtest_tfidf)
s2=accuracy_score(ytest,p2)
print(classification_report(ytest, p2))
print("Multinomial Naive Bayes Classifier Accuracy :", "{:.2f}%".format(100*s2))
plot_confusion_matrix(mnb, Xtest_tfidf, ytest,cmap = 'Blues')
plt.grid(False)


# In[38]:


text = ["Маған ұнамады, қатып қала береді",
        "Байланыс ұстамайды, нашар",
        "Маған ұнамады",
        "Камерасы онша емес, телефон соғылмайды"]

df = predict(tfidf_vect, mnb, text)
print(df)


# In[39]:


text = ["Жақсы телефон",
        "Камерасы мықты, керемет",
        "Тамаша телефон, қолдануға оңай",
        "Өте тез жұмыс жасайды"]

df = predict(tfidf_vect, mnb, text)
print(df)


# ### Linear SVM

# In[40]:


linear_svc = LinearSVC(penalty='l2',loss = 'hinge')
linear_svc.fit(Xtrain_tfidf,ytrain)
p3=linear_svc.predict(Xtest_tfidf)
s3=accuracy_score(ytest,p3)
print(classification_report(ytest, p3))
print("Linear Support Vector Classifier Accuracy :", "{:.2f}%".format(100*s3))
plot_confusion_matrix(linear_svc, Xtest_tfidf, ytest,cmap = 'Blues')
plt.grid(False)


# In[41]:


text = ["Маған ұнамады, қатып қала береді",
        "Байланыс ұстамайды, нашар",
        "Маған ұнамады",
        "Камерасы онша емес, телефон соғылмайды"]

df = predict(tfidf_vect, linear_svc, text)
print(df)


# In[42]:


text = ["Жақсы телефон",
        "Камерасы мықты, керемет",
        "Тамаша телефон, қолдануға оңай",
        "Өте тез жұмыс жасайды"]

df = predict(tfidf_vect, linear_svc, text)
print(df)


# ### XGboost 

# In[43]:


xgbo = XGBClassifier()
xgbo.fit(Xtrain_tfidf,ytrain)
p4=xgbo.predict(Xtest_tfidf)
s4=accuracy_score(ytest,p4)
print(classification_report(ytest, p4))
print("XGBoost Accuracy :", "{:.2f}%".format(100*s4))
plot_confusion_matrix(xgbo, Xtest_tfidf, ytest, cmap = 'Blues')
plt.grid(False)


# In[44]:


text = ["Маған ұнамады, қатып қала береді",
        "Байланыс ұстамайды, нашар",
        "Маған ұнамады",
        "Камерасы онша емес, телефон соғылмайды"]

df = predict(tfidf_vect, xgbo, text)
print(df)


# In[45]:


text = ["Жақсы телефон",
        "Камерасы мықты, керемет",
        "Тамаша телефон, қолдануға оңай",
        "Өте тез жұмыс жасайды"]

df = predict(tfidf_vect, xgbo, text)
print(df)


# # **5. Predictive Modelling using Deep Learning** 

# In[46]:


# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from bs4 import BeautifulSoup
# import re,string,unicodedata

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
# from xgboost.sklearn import XGBClassifier

# import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import plot_model
# import transformers
# import tokenizers


# In[47]:


def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  fig , ax = plt.subplots(1,2,figsize = (10,5))
  
  ax[0].plot(epochRange,history.history['accuracy'],label = 'Training Accuracy')
  ax[0].set_title('Training and Validation accuracy')
  ax[0].set_xlabel('Epoch')
  ax[0].set_ylabel('Accuracy')
  ax[0].legend()
  fig.tight_layout()
  plt.show()


# In[48]:


#splitting into train and test
# data_copy['review']=data_copy['review'].apply(clean_text,lemmatize = False)
#converting target variable to numerical value
# data_copy.sentiment = [ 1 if each == "positive" else 0 for each in data_copy.sentiment]
train, test= train_test_split(data_copy, test_size=0.2, random_state=42)
Xtrain, ytrain = train['review'], train['sentiment']
Xtest, ytest = test['review'], test['sentiment']


# ### LSTM

# In[49]:


#set up the tokenizer
MAX_VOCAB_SIZE = 10000
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE,oov_token="<oov>")
tokenizer.fit_on_texts(Xtrain)
word_index = tokenizer.word_index
#print(word_index)
V = len(word_index)
print("Vocabulary of the dataset is : ",V)


# In[50]:


##create sequences of reviews
seq_train = tokenizer.texts_to_sequences(Xtrain)
seq_test =  tokenizer.texts_to_sequences(Xtest)


# In[51]:


#choice of maximum length of sequences
seq_len_list = [len(i) for i in seq_train + seq_test]

#if we take the direct maximum then
max_len=max(seq_len_list)
print('Maximum length of sequence in the list: {}'.format(max_len))


# In[52]:


# when setting the maximum length of sequence, variability around the average is used.
max_seq_len = np.mean(seq_len_list) + 2 * np.std(seq_len_list)
max_seq_len = int(max_seq_len)
print('Maximum length of the sequence when considering data only two standard deviations from average: {}'.format(max_seq_len))


# In[53]:


perc_covered = np.sum(np.array(seq_len_list) < max_seq_len) / len(seq_len_list)*100
print('The above calculated number coveres approximately {} % of data'.format(np.round(perc_covered,2)))


# So we can use this number for our maxlen parameter.

# In[54]:


#create padded sequences
pad_train=pad_sequences(seq_train,truncating = 'post', padding = 'pre',maxlen=max_seq_len)
pad_test=pad_sequences(seq_test,truncating = 'post', padding = 'pre',maxlen=max_seq_len)


# In[55]:


#Splitting training set for validation purposes
Xtrain,Xval,ytrain,yval=train_test_split(pad_train,ytrain,
                                             test_size=0.2,random_state=10)


# In[56]:


def lstm_model(Xtrain,Xval,ytrain,yval,V,D,maxlen,epochs):
    print("----Building the model----")
    i = Input(shape=(maxlen,))
    x = Embedding(V + 1, D,input_length = maxlen)(i)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Conv1D(32,5,activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(128,return_sequences=True))(x)
    x = LSTM(64)(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    model.summary()

    #Training the LSTM
    print("----Training the network----")
    model.compile(optimizer= Adam(0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
#     #early_stop = EarlyStopping(monitor='val_accuracy', 
#                                mode='min', 
#                                patience = 2 )
#     #checkpoints= ModelCheckpoint(filepath='./',
#                             monitor="val_accuracy",
#                             verbose=0,
#                             save_best_only=True
#                            )
  #  callbacks = [checkpoints,early_stop]
    r = model.fit(Xtrain,ytrain, 
                  validation_data = (Xval,yval), 
                  epochs = epochs, 
                  verbose = 2,
                  batch_size = 32)
                  #callbacks = callbacks
    print("Train score:", model.evaluate(Xtrain,ytrain))
    print("Validation score:", model.evaluate(Xval,yval))
    n_epochs = len(r.history['loss'])
    
    return r,model,n_epochs 


# In[60]:


D = 64 #embedding dims
epochs = 10
r,model,n_epochs = lstm_model(Xtrain,Xval,ytrain,yval,V,D,max_seq_len,epochs)

print(n_epochs)
print(r)
print(model)


# In[61]:


#Plot accuracy and loss
plotLearningCurve(r,n_epochs)


# In[62]:


print("Evaluate Model Performance on Test set")
result = model.evaluate(pad_test,ytest)
print(dict(zip(model.metrics_names, result)))


# In[63]:


#Generate predictions for the test dataset
ypred = model.predict(pad_test)
ypred = ypred>0.5
#Get the confusion matrix
cf_matrix = confusion_matrix(ytest, ypred)
print(classification_report(ytest, ypred))
sns.heatmap(cf_matrix,annot = True,fmt ='g', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[326]:


text = ["Маған ұнамады, қатып қала береді",
        "Байланыс ұстамайды, нашар",
        "Маған ұнамады",
        "Камерасы онша емес, телефон соғылмайды"]

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform((text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df

df = predict(tfidf_vect, model, text)
print(df)


# ### BERT
# 

# In[64]:


train, test= train_test_split(data_copy, test_size=0.2, random_state=42)
Xtrain, ytrain = train['review'], train['sentiment']
Xtest, ytest = test['review'], test['sentiment']
#splitting the train set into train and validation
Xtrain,Xval,ytrain,yval=train_test_split(Xtrain,ytrain,
                                             test_size=0.2,random_state=10)


# In[65]:


#Perform tokenization
# automatically download the vocab used during pretraining or fine-tuning a given model,use from_pretrained() method
tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')


# In[66]:


#pass our texts to the tokenizer. 
Xtrain_enc = tokenizer(Xtrain.tolist(), max_length=max_seq_len, 
                         truncation=True, padding='max_length', 
                         add_special_tokens=True, return_tensors='np') #return numpy object
Xval_enc = tokenizer(Xval.tolist(), max_length=max_seq_len, 
                         truncation=True, padding='max_length', 
                         add_special_tokens=True, return_tensors='np') #return numpy object
Xtest_enc = tokenizer(Xtest.tolist(), max_length=max_seq_len, 
                         truncation=True, padding='max_length', 
                         add_special_tokens=True, return_tensors='np') #return numpy object




# In[67]:


#preparing our datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Xtrain_enc),
    ytrain
))
val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Xval_enc),
    yval
))
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(Xtest_enc),
    ytest
))


# In[68]:


def bert_model(train_dataset,val_dataset,transformer,max_len,epochs):
    print("----Building the model----")
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,),dtype=tf.int32,name = 'attention_mask') #attention mask
    sequence_output = transformer(input_ids,attention_mask)[0]
    cls_token = sequence_output[:, 0, :]
    x = Dense(512, activation='relu')(cls_token)
    x = Dropout(0.1)(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_ids,attention_mask], outputs=y)
    model.summary()
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    r = model.fit(train_dataset.batch(32),batch_size = 32,
                  validation_data = val_dataset.batch(32),epochs = epochs)
                  #callbacks = callbacks
    print("Train score:", model.evaluate(train_dataset.batch(32)))
    print("Validation score:", model.evaluate(val_dataset.batch(32)))
    n_epochs = len(r.history['loss'])
    
    return r,model,n_epochs 


# In[69]:


transformer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')


# In[333]:


epochs = 15
max_len = max_seq_len
r,model,n_epochs = bert_model(train_dataset,val_dataset,transformer,max_len,epochs)


# In[334]:


#Plot accuracy and loss
plotLearningCurve(r,n_epochs)


# In[335]:


print("Evaluate Model Performance on Test set")
result = model.evaluate(test_dataset.batch(32))
print(dict(zip(model.metrics_names, result)))


# In[336]:


#Generate predictions for the test dataset
ypred = model.predict(test_dataset.batch(32))
ypred = ypred>0.5
#Get the confusion matrix
cf_matrix = confusion_matrix(ytest, ypred)
print(classification_report(ytest, ypred))
sns.heatmap(cf_matrix,annot = True,fmt ='g', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# ### 

# In[157]:


get_ipython().system('pip install polyglot')


# In[1]:


print('qqwe')


# In[2]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[3]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# In[4]:


tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')


# In[5]:


result = model(tokens)


# In[6]:


result.logits


# In[7]:


int(torch.argmax(result.logits))+1


# In[ ]:




