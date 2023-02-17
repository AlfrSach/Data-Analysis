#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the necessary libraries
import pandas as pd
from bs4 import BeautifulSoup
import re
from newspaper import Article
import os
import nltk
from nltk import sent_tokenize, word_tokenize


# In[3]:


#Read the URLs from the Input.xlsx and store in a list
input_data = pd.read_excel('Input.xlsx')
url_ids = input_data['URL_ID'].tolist()
urls = input_data['URL'].tolist()

# Check if all URL_IDs and URLs are read successfully
print(len(url_ids))
print(len(urls))
print(url_ids[:5]) # Print first 5 URL_IDs to check if the list is correct
print(urls[:5]) # Print first 5 URLs to check if the list is correct


# In[3]:


def get_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        paragraphs = article.text.split('\n')
        article_text = '\n'.join([p.strip() for p in paragraphs if p.strip() != ''])
        article_title = article.title
        return article_title, article_text
    except Exception as e:
        print(f"Error in processing URL {url}: {e}")
        return None, None


# In[5]:


for url_id, url in zip(url_ids, urls):
    title, text = get_article_text(url)
    if title is None or text is None:
        continue
    filename = str(int(url_id))
    with open(filename + ".txt", "w", encoding="utf-8") as f:
        f.write(title + "\n" + text)


# In[4]:


# Create a list to store the rows of the dataframe
rows = []

# Iterate over the file names
for file_name in os.listdir():
    # Check if the file is a text file
    if file_name.endswith(".txt"):
        # Get the URL_ID from the file name by removing the '.txt' extension
        url_id = file_name[:-4]
        # Read the contents of the text file
        if url_id in str(url_ids):
            with open(file_name, "r", encoding="utf-8") as f:
                contents = f.read()
            
        # Split the contents into title and text
            title, text = contents.split("\n", 1)
        
        # Append the row to the list of rows
            rows.append([int(url_id), title, text])

# Sort the list of rows based on the URL_ID column
rows = sorted(rows, key=lambda x: x[0])


# In[5]:


# Create a dataframe from the list of rows
df = pd.DataFrame(rows, columns=['URL_ID', 'Title', 'Text'])


# In[6]:


df.to_excel("data.xlsx", index=False)


# In[159]:


df = pd.read_excel("data.xlsx")


# In[160]:


df


# In[161]:


df["Text"][0]


# ## Text Preprocessing

# #### Normalizing

# In[162]:


text = df["Text"][0]


# In[163]:


text = text.lower()


# In[164]:


re_special_char1 = r'\n'


# In[165]:


re_special_char2 = r"[^a-zA-Z|\s]"


# In[166]:


text = re.sub(re_special_char1, "|", text)


# In[167]:


text = re.sub(re_special_char2, "", text)


# In[168]:



text_list = text.split(" ")
text = " ".join(text_list[:-11])

print(text)


# #### Stop Word

# In[169]:


def read_stop_words(filename):
    with open(filename, "r") as file:
        stop_words = file.read().splitlines()
    return [word.lower() for word in stop_words]


# In[170]:


stop_words = []
for filename in ["StopWords_Auditor.txt", "StopWords_Currencies.txt", "StopWords_DatesandNumbers.txt", "StopWords_Generic.txt", "StopWords_GenericLong.txt", "StopWords_Geographic.txt", "StopWords_Names.txt"]:
    stop_words += read_stop_words(filename)


# In[171]:


stop_words


# In[172]:


text = text.split()
print(text)


# In[173]:


text = [word for word in text if word not in stop_words]


# In[174]:


def text_preprocessing(text):
    #Replace \n at end of sentence with |
    text = re.sub(re_special_char1, '|', text)
    #Normalizing the case
    text = text.lower()
    #Remove the special characters
    text = re.sub(re_special_char2, "", text)
    text_list = text.split(" ")
    text = " ".join(text_list[:-11])
    #Word Tokenization
    text = text.split()
    #Stopword removal 
    text = [word for word in text if word not in stop_words]
    #Joining text
    text = " ".join(text)
    return text


# In[175]:


df["Text"]= df["Text"].apply(text_preprocessing)


# In[176]:


df


# #### Master Dictionary

# In[177]:


#reading the two files that contain the positive and negative words, and create two lists to store the words
def read_words(filename):
    with open(filename, "r") as file:
        words = file.read().splitlines()
    return [word.lower() for word in words]


# In[178]:


positive_words = read_words("positive-words.txt")
negative_words = read_words("negative-words.txt")


# ### Extracting Derived Variables

# In[179]:


#compute the positive and negative scores of each text in your data
def compute_scores(text):
    positive_score = 0
    negative_score = 0
    for word in text.split():
        if word in positive_words:
            positive_score += 1
        if word in negative_words:
            negative_score += -1
    return {'positive_score': positive_score, 'negative_score': negative_score * -1}


# In[180]:


df[['positive_score', 'negative_score']] = df['Text'].apply(compute_scores).apply(pd.Series)


# In[181]:


df


# In[182]:


def compute_polarity_score(positive_score, negative_score):
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    return polarity_score


# In[183]:


df['polarity_score'] = df.apply(lambda x: compute_polarity_score(x['positive_score'], x['negative_score']), axis=1)


# In[184]:


def compute_subjectivity_score(text, positive_score, negative_score):
    total_words = len(text.split())
    subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)
    return subjectivity_score


# In[185]:


subjectivity_scores = df['Text'].apply(lambda x: compute_subjectivity_score(x, compute_scores(x)['positive_score'], compute_scores(x)['negative_score']))
df['subjectivity_score'] = subjectivity_scores


# In[186]:


df


# ### Complex Word Count & Analysis of Readability

# In[187]:


# nltk.download('punkt')


# In[188]:


def count_syllables(word):
    vowels = 'aeiouAEIOU'
    syllable_count = 0
    word = word.lower()
    if word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllable_count += 1
    if word.endswith('e'):
        syllable_count -= 1
    if syllable_count == 0:
        syllable_count += 1
    return syllable_count


# In[189]:


def compute_readability(text):
    avg_sentence_length = 0
    complex_words = 0
    sentences = text.split("|")
    total_words = len(nltk.word_tokenize(text))
    avg_sentence_length = total_words / len(sentences)
    words = nltk.word_tokenize(text)
    for word in words:
        syllable_count = count_syllables(word)
        if syllable_count > 2:
            complex_words += 1
    
    percentage_of_complex_words = complex_words / total_words
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)
    
    return {'Avg Sentence Length': avg_sentence_length, 'Percentage of Complex words': percentage_of_complex_words, 'Fog Index': fog_index}


# In[190]:


df[['Avg Sentence Length', 'Percentage of Complex words', 'Fog Index']] = df.apply(lambda x: compute_readability(x['Text']), axis=1, result_type="expand")


# ### Average Number of Words Per Sentence

# In[191]:


def avg_words_per_sentence(text):
    sentences = text.split("|")
    words = len(text.split())
    avg_words = words/len(sentences)
    return avg_words


# In[192]:


df['Avg Words Per Sentence'] = df['Text'].apply(lambda x: avg_words_per_sentence(x))


# ### Complex Words

# In[193]:


def count_complex_words(text):
    complex_words = 0
    words = nltk.word_tokenize(text)
    for word in words:
        syllable_count = count_syllables(word)
        if syllable_count > 2:
            complex_words += 1
    return complex_words


# In[194]:


df['Complex word count'] = df['Text'].apply(lambda x: count_complex_words(x))


# In[195]:


df


# ### Word Count & Syllable Count

# In[196]:


def word_count(text):
    words = word_tokenize(text)
    return len(words)


# In[197]:


df['Word Count'] = df['Text'].apply(word_count)


# In[198]:


def count_syllables_per_word(text):
    words = word_tokenize(text)
    syllables_per_word = [count_syllables(word) for word in words]
    return syllables_per_word


# In[199]:


df['Syllables per Word'] = df['Text'].apply(count_syllables_per_word)


# In[200]:


df


# ### Personal Pronouns

# In[201]:


def count_personal_pronouns(text):
    # Define the regular expression pattern to match personal pronouns
    pattern = re.compile(r"\b(I|we|my|ours|us)\b", re.IGNORECASE)
    
    # Use re.findall to search for all occurrences of the pattern in the text
    matches = re.findall(pattern, text)
    
    # Return the count of personal pronouns
    return len(matches)


# In[202]:


df["Personal Pronouns"] = df["Text"].apply(count_personal_pronouns)


# ### Average Word Length

# In[203]:


def avg_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)


# In[204]:


df["Average Word Length"] = df["Text"].apply(avg_word_length)


# In[205]:


df


# In[206]:


filtered_input_data = input_data[~input_data["URL_ID"].isin([44, 57, 144])]


# In[207]:


df.insert(2, "URL", filtered_input_data["URL"])


# In[208]:


df


# In[210]:


df = df.drop(columns=["Title", "Text"])


# In[211]:


df = df.rename(columns={
    "positive_score": "POSITIVE SCORE",
    "negative_score": "NEGATIVE SCORE",
    "polarity_score": "POLARITY SCORE",
    "subjectivity_score": "SUBJECTIVITY SCORE",
    "Avg Sentence Length": "AVG SENTENCE LENGTH",
    "Percentage of Complex words": "PERCENTAGE OF COMPLEX WORDS",
    "Fog Index": "FOG INDEX",
    "Avg Words Per Sentence": "AVG NUMBER OF WORDS PER SENTENCE",
    "Complex word count": "COMPLEX WORD COUNT",
    "Word Count": "WORD COUNT",
    "Syllables per Word": "SYLLABLE PER WORD",
    "Personal Pronouns": "PERSONAL PRONOUNS",
    "Average Word Length": "AVG WORD LENGTH"
})


# In[212]:


df


# In[213]:


df.to_excel("Output.xlsx", index=False)


# In[ ]:




