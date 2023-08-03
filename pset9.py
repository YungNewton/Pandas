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

# Store the adjusted R-squared in a variable named adj_rsquared
adj_rsquared = result.rsquared_adj
print("Adjusted R-squared:", adj_rsquared)

# Store the p-value of F-statistics in a variable named f_pvalue
f_pvalue = result.f_pvalue
print("p-value of F-statistics:", f_pvalue)

# Store the p-value of sscore in a variable named sscore_pvalue
sscore_pvalue = result.pvalues['sscore']
print("p-value of sscore:", sscore_pvalue)

# Store the p-value of sp500 in a variable named sp500_pvalue
sp500_pvalue = result.pvalues['sp500']
print("p-value of sp500:", sp500_pvalue)

# If a relationship exists between sscore and ibm stock price, then store a boolean value of
# True in a variable named sscore_rel; otherwise, sscore_rel should be set to False
sscore_rel = sscore_pvalue < 0.05  # We use 0.05 as the threshold for statistical significance
print("Is sscore related to IBM stock price?", sscore_rel)

# If a relationship exists between s&p 500 index and ibm stock price, then store a boolean value of True in a variable named sp500_rel; 
# otherwise, sp500_rel should be set to False
sp500_rel = sp500_pvalue < 0.05  # We use 0.05 as the threshold for statistical significance
print("Is S&P 500 index related to IBM stock price?", sp500_rel)

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