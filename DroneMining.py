#text mining LDA topic model
import pandas as pd
import re
import gensim
import spacy
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import webbrowser
import os
import glob

# Collect the file names of all .txt files in the directory.
# Use "*.txt" to match all text files.
directory = "/Users/peilokdorothymo/Library/CloudStorage/OneDrive-UniversityofBath/Bath/Year_One/Drone/Python_Text_Mining/Patent_Doc_txt"
file_names = glob.glob(os.path.join(directory, "*.txt"))

# Merge the text files into a single file titled 'all.txt'.
merged_file_path = os.path.join(directory, "all.txt")
with open(merged_file_path, 'w', encoding="utf-8") as out_file:
    for file_path in file_names:
        with open(file_path, 'r', encoding="utf-8") as in_file:
            out_file.write(in_file.read())
            out_file.write("\n")  # Add a newline separator between files

# Create a DataFrame from the merged txt file.
with open(merged_file_path, 'r', encoding="utf-8") as file:
    lines = file.read().splitlines()
data = pd.DataFrame({"text_column": lines})

# Preprocess the text data
def preprocess_text(text):
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = re.sub('\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub('\'', '', text)  # Remove apostrophes
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    return text

data['cleaned_text'] = data['text_column'].apply(preprocess_text)

# Full path to the file of the customised stopword list
stopword_file_path = "/Users/peilokdorothymo/Library/CloudStorage/OneDrive-UniversityofBath/Bath/Year_One/Drone/Python_Text_Mining/Extra_English.txt"
with open(stopword_file_path, 'r') as f:
    custom_stopwords = f.read().splitlines()

# Tokenize and remove stopwords
def tokenize(text):
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in custom_stopwords]
    return tokens

data['tokens'] = data['cleaned_text'].apply(tokenize)

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.max_length = 2000000

def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

data['lemmas'] = data['tokens'].apply(lemmatize)

# Create dictionary and corpus
id2word = corpora.Dictionary(data['lemmas'])
texts = data['lemmas']
corpus = [id2word.doc2bow(text) for text in texts]


# Define the directory where the CSV files will be saved
save_directory = "/Users/peilokdorothymo/Library/CloudStorage/OneDrive-UniversityofBath/Bath/Year_One/Drone/Python_Text_Mining/Processed_Files"
os.makedirs(save_directory, exist_ok=True)
# save lemmatized tokens to a CSV file
lemmas_csv_path = os.path.join(save_directory, "processed_lemmas.csv")
data[['lemmas']].to_csv(lemmas_csv_path, index=False)


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=5,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=5,
                                            alpha='auto',
                                            per_word_topics=True)

# Print the 10 keysword for each topic
topics = lda_model.print_topics(num_words=10)
for topic in topics:
     print(topic)
# The output shows 5 topics, each represented by a list of words with associated weights,
# indicating the importance of each word in that topic.

# Visualisation of the model and save it as a html file
# Open the html file
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'lda_visualization.html')
html_file_path = os.path.abspath("lda_visualization.html")
webbrowser.open("file://" + html_file_path)


save_