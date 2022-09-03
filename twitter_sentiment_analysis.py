import nltk
import numpy as np
import pandas as pd 
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt

df=pd.read_csv('Data/Sentiment.csv')
df.head()

df=df[['text','sentiment']]

df.isna().sum()

data_pos=df[df['sentiment']=='Positive']
data_neg=df[df['sentiment']=='Negative']

from sklearn.utils import resample

data_pos_upsampled = resample(data_pos, 
                                 replace=True,    
                                 n_samples=len(data_neg),   
                                 random_state=42)
new_df_upsampled = pd.concat([data_pos_upsampled, data_neg])
new_df_upsampled['sentiment'].value_counts()

new_df_upsampled = new_df_upsampled.sample(frac = 1)

import re
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text=re.sub('rt','',text)
    text = [word for word in text.split(' ') if word not in stopword]
    text= " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text= " ".join(text)
    return text
new_df_upsampled["text"] = new_df_upsampled["text"].apply(clean)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

x = np.array(new_df_upsampled["text"])
y = np.array(new_df_upsampled["sentiment"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import classification_report
model=SVC()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

y_pred=model.predict(X_test)
print(classification_report(y_test,y_pred))

from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

import pickle
pickle.dump(cv,open('countvectorizer.pkl','wb'))
pickle.dump(model,open('SA_Model.pkl','wb'))