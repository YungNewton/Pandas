# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Download the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the stopwords to be excluded
stop_words = set(stopwords.words('english'))

# Open and read the file
with open('interview.txt', 'r') as file:
    text = file.read().lower() # convert all words to lowercase

# Tokenize the text
words = word_tokenize(text)

# Filter out the stopwords
filtered_words = [word for word in words if word not in stop_words and word.isalnum()]

# Count the frequency of each word
word_counts = Counter(filtered_words)

# Get the 5 most common words
top_five = word_counts.most_common(5)

# Print the top 5 most frequent words
for word, count in top_five:
    print(f'Word: {word}, Frequency: {count}')
