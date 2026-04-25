<H3>NAME: Cynthia Mehul</H3>
<H3>REG. NO.: 212223240020</H3>
<H3>DATE: 25/04/2026</H3>
<H1 Align="center">Project Based Experiment<H1>
<H3>Objective:<H3>
To perform sentiment analysis using inbuilt Twitter(X) data and filtering the data that has only negative feedback for the code given in the following link.
<H3>Program:</H3>
    
```

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import twitter_samples

nltk.download('vader_lexicon')
nltk.download('twitter_samples')

texts = twitter_samples.strings('tweets.20150430-223406.json')
sid = SentimentIntensityAnalyzer()
negative_posts = []

for text in texts:
    scores = sid.polarity_scores(text)
    
    if scores['compound'] < 0:
        negative_posts.append((text, scores))

for post, score in negative_posts[:10]:
    print("\nPost:", post)
    print("Sentiment:", score)

```

<H3>Output:</H3>
<img width="1007" height="647" alt="image" src="https://github.com/user-attachments/assets/100d9a0b-eac8-4480-a2e3-55ef741e7b2b" />

<H3>Inference:</H3>
Use of NLTK and VADER to analyze text sentiments automatically.
Use of SentimentIntensityAnalyzer() to perform sentiment analysis.
Interpretation of polarity scores and filtering negative feedback from real-world social media data.

