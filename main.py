import mailbox
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import *
import eml_parser, json, datetime
import email
import pandas as pd
import numpy as np
import os
import joblib
import time

# NLP with NLTK
import nltk
nltk.download("stopwords")
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# For visualization
from wordcloud import WordCloud
#import pyLDAvis
#import pyLDAvis.gensim

# Sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Keras
import tensorflow as tf
from tensorflow.keras import layers


#Data preprocessing
# Extracting the body text


def get_charsets(msg):
    charsets = set({})
    for c in msg.get_charsets():
        if c is not None:
            charsets.update([c])
    return charsets

def handle_error(errmsg, emailmsg,cs):
    '''print()
    print(errmsg)
    print("This error occurred while decoding with ",cs," charset.")
    print("These charsets were found in the one email.",get_charsets(emailmsg))
    print("This is the subject:",emailmsg['subject'])
    print("This is the sender:",emailmsg['From'])'''
    pass

def get_body_from_email(msg):
    body = None
    #Walk through the parts of the email to find the text body.    
    if msg.is_multipart():    
        for part in msg.walk():

            # If part is multipart, walk through the subparts.            
            if part.is_multipart(): 

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True) 
                        #charset = subpart.get_charset()

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)
                #charset = part.get_charset()

    # If this isn't a multi-part message then get the payload (i.e the message body)
    #elif msg.get_content_type() == 'text/plain':
        #body = msg.get_payload(decode=True) 
    
    # Uncomment this to also include html and other formats
    else:
        body = msg.get_payload(decode=True) 

    return body


# Phishing emails

phishingBoxFilenames = [
    "G:/codeAIPhishing/datasetEmail/monkey/phishing3.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing1.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing0.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2022.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2021.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2020.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2019.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2018.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2017.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2016.mbox",
    "G:/codeAIPhishing/datasetEmail/monkey/phishing2015.mbox"
]

phishingBoxes = [mailbox.mbox(f) for f in phishingBoxFilenames]
phishingMessages = [m[1] for phishingBox in phishingBoxes for m in phishingBox.items()]

phishingMessageBodies = []
for p in phishingMessages:
    body = get_body_from_email(p)
    if body is not None and len(body) > 0:
        phishingMessageBodies.append(body)
        
print(len(phishingMessages), len(phishingMessageBodies))

#Clair Fraud Email Database
added = []
with open(r"G:\codeAIPhishing\datasetEmail\clair\fradulent_emails.txt", 'r', errors="ignore") as f:
    body = ""
    inBody = False
    for line in f:
        if line.startswith("Status: O"):
            inBody = True
        
        elif line.startswith("From r") and len(body) > 0:
            inBody = False
            added.append(body)
            body = ""

        elif inBody:
            body += line

phishingMessageBodies = list(set(phishingMessageBodies + [a for a in added if len(a) > 0]))
print(len(phishingMessageBodies))

# SpamAssassin Spam (not exactly phishing, but NVIDIA article used it as phishing so attempting it)
# Khởi tạo trình phân tích email  
ep = eml_parser.EmlParser(include_raw_body=True)  

# Đường dẫn đến thư mục chứa email spam  
spamDir = "G:/codeAIPhishing/datasetEmail/spamassassin/spam_2/spam_2"
spamFilenames = [os.path.join(spamDir, f) for f in os.listdir(spamDir)]  

added = []  

# Vòng lặp qua từng file email  
for filename in spamFilenames:  
    try:  
        with open(filename, "rb") as f:  
            b = f.read()  

        # Phân tích email  
        m = ep.decode_email_bytes(b)  
        
        # Kiểm tra sự tồn tại của phần body trước khi truy cập  
        if "body" in m and len(m["body"]) > 0:  
            content = m["body"][0]["content"]  
            if content:  # Kiểm tra xem nội dung có khác None không  
                added.append(content)  
    
    except Exception as e:  
        # In ra lỗi nếu có vấn đề với file email hiện tại  
        print(f"Lỗi khi xử lý {filename}: {e}")  

# Gộp các nội dung email mới vào danh sách phishingMessageBodies  
# Giả định rằng phishingMessageBodies đã được định nghĩa trước đó  
phishingMessageBodies = list(set(phishingMessageBodies + added))  # Loại bỏ trùng lặp  
print(len(phishingMessageBodies))  # In ra tổng số nội dung duy nhất  


#Benign emails
ep = eml_parser.EmlParser(include_raw_body=True)

easyHamDir = "G:/codeAIPhishing/datasetEmail/spamassassin/easy_ham/easy_ham/"
hardHamDir = "G:/codeAIPhishing/datasetEmail/spamassassin/hard_ham/hard_ham/"

hamFilenames = [os.path.join(easyHamDir, f) for f in os.listdir(easyHamDir)] + [os.path.join(hardHamDir, f) for f in os.listdir(hardHamDir)]

benignMessageBodies = []
for filename in hamFilenames:
    with open(filename, "rb") as f:
        b = f.read()
    
    m = ep.decode_email_bytes(b)
    if len(m["body"]) >= 1:
        benignMessageBodies.append(m["body"][0]["content"])
    
print(len(benignMessageBodies))




stopWords = nltk.corpus.stopwords
stopWords = stopWords.words("english")
stopWords.extend(["nbsp", "font", "sans", "serif", "bold", "arial", "verdana", "helvetica", "http", "https", "www", "html", "enron", "margin", "spamassassin"])

def remove_custom_stopwords(p):
    return remove_stopwords(p, stopwords=stopWords)

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_custom_stopwords, remove_stopwords, strip_short, stem_text]


# Decode to utf-8 as needed

# Phishing
phishingDecoded = []
for b in phishingMessageBodies:
    try:
        p = b.decode("utf-8", errors="ignore")
    except AttributeError:
        p = b
    phishingDecoded.append(p)
phishingMessageBodies = phishingDecoded

# Benign
benignDecoded = []
for b in benignMessageBodies:
    try:
        p = b.decode("utf-8", errors="ignore")
    except AttributeError:
        p = b
    benignDecoded.append(p)
benignMessageBodies = benignDecoded

# Phishing emails
phishingPreprocessed = []
for b in phishingMessageBodies:
    p = preprocess_string(b, filters=CUSTOM_FILTERS)
    #p = gensim.parsing.preprocessing.remove_stopwords(p, stopwords=stopWords
    
    phishingPreprocessed.append(p)
print(len(phishingPreprocessed))

# Benign emails
benignPreprocessed = []
for b in benignMessageBodies:
    p = preprocess_string(b, filters=CUSTOM_FILTERS)
    benignPreprocessed.append(p)
print(len(benignPreprocessed))



# 1. Chuẩn bị dữ liệu
X = phishingPreprocessed + benignPreprocessed
y = [1] * len(phishingPreprocessed) + [0] * len(benignPreprocessed)

# 2. Chuyển các từ đã tiền xử lý về dạng văn bản để sử dụng với TF-IDF
X_text = [' '.join(tokens) for tokens in X]

# 3. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# 4. Tạo pipeline với TF-IDF và RandomForest
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# 5. Huấn luyện mô hình
print("Bắt đầu huấn luyện mô hình...")
start_time = time.time()
pipeline.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Thời gian huấn luyện: {training_time:.2f} giây")

# 6. Đánh giá mô hình
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Độ chính xác: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 7. Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
print("Ma trận nhầm lẫn:")
print(cm)

# 8. Lưu mô hình
joblib.dump(pipeline, 'phishing_detection_model.pkl')
print("Đã lưu mô hình vào 'phishing_detection_model.pkl'")

# 9. Thử nghiệm với một email mới
def predict_phishing(email_text):
    # Tiền xử lý email mới
    preprocessed = preprocess_string(email_text, filters=CUSTOM_FILTERS)
    text = ' '.join(preprocessed)
    
    # Dự đoán
    prediction = pipeline.predict([text])[0]
    probability = pipeline.predict_proba([text])[0][1]
    
    if prediction == 1:
        return f"Email này có khả năng là lừa đảo với xác suất {probability:.2f}"
    else:
        return f"Email này có vẻ an toàn với xác suất {1-probability:.2f}"

# Ví dụ kiểm tra
test_email = """
From exmh-workers-admin@redhat.com  Thu Aug 22 12:36:23 2002
Return-Path: <exmh-workers-admin@spamassassin.taint.org>
Delivered-To: zzzz@localhost.netnoteinc.com
Received: from localhost (localhost [127.0.0.1])
	by phobos.labs.netnoteinc.com (Postfix) with ESMTP id D03E543C36
	for <zzzz@localhost>; Thu, 22 Aug 2002 07:36:16 -0400 (EDT)
Received: from phobos [127.0.0.1]
	by localhost with IMAP (fetchmail-5.9.0)
	for zzzz@localhost (single-drop); Thu, 22 Aug 2002 12:36:16 +0100 (IST)
Received: from listman.spamassassin.taint.org (listman.spamassassin.taint.org [66.187.233.211]) by
    dogma.slashnull.org (8.11.6/8.11.6) with ESMTP id g7MBYrZ04811 for
    <zzzz-exmh@spamassassin.taint.org>; Thu, 22 Aug 2002 12:34:53 +0100
Received: from listman.spamassassin.taint.org (localhost.localdomain [127.0.0.1]) by
    listman.redhat.com (Postfix) with ESMTP id 8386540858; Thu, 22 Aug 2002
    07:35:02 -0400 (EDT)
Delivered-To: exmh-workers@listman.spamassassin.taint.org
Received: from int-mx1.corp.spamassassin.taint.org (int-mx1.corp.spamassassin.taint.org
    [172.16.52.254]) by listman.redhat.com (Postfix) with ESMTP id 10CF8406D7
    for <exmh-workers@listman.redhat.com>; Thu, 22 Aug 2002 07:34:10 -0400
    (EDT)
Received: (from mail@localhost) by int-mx1.corp.spamassassin.taint.org (8.11.6/8.11.6)
    id g7MBY7g11259 for exmh-workers@listman.redhat.com; Thu, 22 Aug 2002
    07:34:07 -0400
Received: from mx1.spamassassin.taint.org (mx1.spamassassin.taint.org [172.16.48.31]) by
    int-mx1.corp.redhat.com (8.11.6/8.11.6) with SMTP id g7MBY7Y11255 for
    <exmh-workers@redhat.com>; Thu, 22 Aug 2002 07:34:07 -0400
Received: from ratree.psu.ac.th ([202.28.97.6]) by mx1.spamassassin.taint.org
    (8.11.6/8.11.6) with SMTP id g7MBIhl25223 for <exmh-workers@redhat.com>;
    Thu, 22 Aug 2002 07:18:55 -0400
Received: from delta.cs.mu.OZ.AU (delta.coe.psu.ac.th [172.30.0.98]) by
    ratree.psu.ac.th (8.11.6/8.11.6) with ESMTP id g7MBWel29762;
    Thu, 22 Aug 2002 18:32:40 +0700 (ICT)
Received: from munnari.OZ.AU (localhost [127.0.0.1]) by delta.cs.mu.OZ.AU
    (8.11.6/8.11.6) with ESMTP id g7MBQPW13260; Thu, 22 Aug 2002 18:26:25
    +0700 (ICT)
From: Robert Elz <kre@munnari.OZ.AU>
To: Chris Garrigues <cwg-dated-1030377287.06fa6d@DeepEddy.Com>
Cc: exmh-workers@spamassassin.taint.org
Subject: Re: New Sequences Window
In-Reply-To: <1029945287.4797.TMDA@deepeddy.vircio.com>
References: <1029945287.4797.TMDA@deepeddy.vircio.com>
    <1029882468.3116.TMDA@deepeddy.vircio.com> <9627.1029933001@munnari.OZ.AU>
    <1029943066.26919.TMDA@deepeddy.vircio.com>
    <1029944441.398.TMDA@deepeddy.vircio.com>
MIME-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Message-Id: <13258.1030015585@munnari.OZ.AU>
X-Loop: exmh-workers@spamassassin.taint.org
Sender: exmh-workers-admin@spamassassin.taint.org
Errors-To: exmh-workers-admin@spamassassin.taint.org
X-Beenthere: exmh-workers@spamassassin.taint.org
X-Mailman-Version: 2.0.1
Precedence: bulk
List-Help: <mailto:exmh-workers-request@spamassassin.taint.org?subject=help>
List-Post: <mailto:exmh-workers@spamassassin.taint.org>
List-Subscribe: <https://listman.spamassassin.taint.org/mailman/listinfo/exmh-workers>,
    <mailto:exmh-workers-request@redhat.com?subject=subscribe>
List-Id: Discussion list for EXMH developers <exmh-workers.spamassassin.taint.org>
List-Unsubscribe: <https://listman.spamassassin.taint.org/mailman/listinfo/exmh-workers>,
    <mailto:exmh-workers-request@redhat.com?subject=unsubscribe>
List-Archive: <https://listman.spamassassin.taint.org/mailman/private/exmh-workers/>
Date: Thu, 22 Aug 2002 18:26:25 +0700

    Date:        Wed, 21 Aug 2002 10:54:46 -0500
    From:        Chris Garrigues <cwg-dated-1030377287.06fa6d@DeepEddy.Com>
    Message-ID:  <1029945287.4797.TMDA@deepeddy.vircio.com>


  | I can't reproduce this error.

For me it is very repeatable... (like every time, without fail).

This is the debug log of the pick happening ...

18:19:03 Pick_It {exec pick +inbox -list -lbrace -lbrace -subject ftp -rbrace -rbrace} {4852-4852 -sequence mercury}
18:19:03 exec pick +inbox -list -lbrace -lbrace -subject ftp -rbrace -rbrace 4852-4852 -sequence mercury
18:19:04 Ftoc_PickMsgs {{1 hit}}
18:19:04 Marking 1 hits
18:19:04 tkerror: syntax error in expression "int ...

Note, if I run the pick command by hand ...

delta$ pick +inbox -list -lbrace -lbrace -subject ftp -rbrace -rbrace  4852-4852 -sequence mercury
1 hit

That's where the "1 hit" comes from (obviously).  The version of nmh I'm
using is ...

delta$ pick -version
pick -- nmh-1.0.4 [compiled on fuchsia.cs.mu.OZ.AU at Sun Mar 17 14:55:56 ICT 2002]

And the relevant part of my .mh_profile ...

delta$ mhparam pick
-seq sel -list


Since the pick command works, the sequence (actually, both of them, the
one that's explicit on the command line, from the search popup, and the
one that comes from .mh_profile) do get created.

kre

ps: this is still using the version of the code form a day ago, I haven't
been able to reach the cvs repository today (local routing issue I think).



_______________________________________________
Exmh-workers mailing list
Exmh-workers@redhat.com
https://listman.redhat.com/mailman/listinfo/exmh-workers


"""

print("\nKiểm tra với email mẫu:")
print(predict_phishing(test_email))