import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import ssl

# Attempt to create unverified HTTPS context; this step may be necessary depending on your Python environment.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Import and download the VADER lexicon for sentiment analysis.
import nltk
nltk.download('vader_lexicon')

# Step 1: Load the data and show the shape and first few rows.
df = pd.read_csv('stockdailyhlnews-1.csv')  # Replace with your csv file path.
print("Data shape:", df.shape)  # Should print (1989, 6).
print("First few rows of data:")
print(df.head())

# Add a column called sscore and fill it with the ‘compound’ sentiment analysis score based on the daily headline news for each day.
sid = SentimentIntensityAnalyzer()
df['sscore'] = df['news'].apply(lambda news: sid.polarity_scores(news)['compound'])

# Print the new dataframe with the 'sscore' column.
print("Dataframe after adding 'sscore' column:")
print(df.head())

# Calculate the average compound score and print it.
avgsscore = df['sscore'].mean()
print("Average compound score:", avgsscore)

# Step 2: Convert 'weekday' and 'president' columns to dummy variables and add them to the original dataframe.
df_dummy = pd.get_dummies(df, columns=['weekday', 'president'])

# Print the new dataframe with the dummy variables.
print("Dataframe after converting 'Weekday' and 'President' to dummy variables:")
print(df_dummy.head())

# Step 3: Conduct a linear regression to investigate the relationship between 'sscore', 'sp500', and 'ibm'.
result = ols(formula='ibm ~ sscore + sp500', data=df_dummy).fit()
print(result.summary())

# Interpretation: The regression summary above provides the coefficients, standard errors, and p-values among other statistics 
# for the intercept and the predictors ('sscore' and 'Sp500'). The p-value for each predictor tests the null hypothesis that 
# the predictor's coefficient is zero, assuming that all other predictors are in the model. If the p-value is less than 0.05,
# we can reject this null hypothesis and conclude that the predictor has a significant effect on the response ('ibm').

# Step 4: Predict whether the 'President' will be Republican or Democrat based on 'Sp500', 'Ibm', and 'sscore' using logistic regression.
X = df[['sp500', 'ibm', 'sscore']]
y = df['president']

# Convert the labels into numerical values.
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train a logistic regression model and print its score.
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
logmodel_score = logmodel.score(X_test, y_test)
print("Logistic regression model score:", logmodel_score)
