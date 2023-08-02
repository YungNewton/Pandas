# Import necessary libraries
import pandas as pd
from scipy import stats

# Load the dataset from a CSV file
data = pd.read_csv('all_stocks.csv', index_col='Date', parse_dates=True)

# Sort the data in ascending order by date to ensure calculations are accurate
data.sort_index(inplace=True)
# Check if there are missing dates in the data
required_dates = pd.date_range(start='2019-01-01', end='2022-04-18')
missing_dates = required_dates.difference(data.index)
# If there are missing dates, display a warning
if not missing_dates.empty:
    print(f"Warning: Missing dates in the data ")

# Compute daily return for each stock
returns = data.pct_change().fillna(0)

# Compute average daily return for each stock
ibm_rr = returns['ibm'].mean()
wmt_rr = returns['wmt'].mean()
msft_rr = returns['msft'].mean()
amzn_rr = returns['amzn'].mean()


#A t-test is a statistical hypothesis test used to determine whether there is a significant difference between the means of two groups.
#The test calculates a t-score, which expresses the size of the differences relative to the variation in the data. The larger the absolute value of the t-score, the smaller the chance that the differences occurred by coincidence, i.e., the more significant the difference.




#The p-value is a measure of the probability that an observed difference could have occurred just by random chance. 
# The smaller the p-value, the greater the statistical significance of the observed difference.

# Conduct a t-test to compare the rates of return of IBM and Walmart
ttest_result = stats.ttest_ind(returns['ibm'], returns['wmt'])
# Check if the difference is significant
iw_diff = 'YES' if ttest_result.pvalue < 0.05 else 'NO'
# Store the p-value of the t-test
iw_pvalue = ttest_result.pvalue

# Compute the weighted average return of the portfolio containing IBM and Walmart
iwp_rr = 0.4 * ibm_rr + 0.6 * wmt_rr
# Compute the weighted average return of the portfolio containing Microsoft and Amazon
map_rr = 0.4 * msft_rr + 0.6 * amzn_rr

# Determine which portfolio has a higher average return
best_portfolio = 1 if iwp_rr > map_rr else 2




# Compare the performance of Walmart's stock before and during Covid-19
pre_covid_wmt = returns.loc['2019-01-01':'2020-03-14', 'wmt']
during_covid_wmt = returns.loc['2020-03-15':'2021-05-25', 'wmt']
# Conduct a t-test to compare the rates of return before and during Covid-19
ttest_result_wmt = stats.ttest_ind(pre_covid_wmt, during_covid_wmt)
# Check if the performance was significantly better before Covid-19
better_w_covid = 'YES' if ttest_result_wmt.pvalue < 0.05 else 'NO'


#This line checks the p-value from the t-test. 
# If the p-value is less than 0.05, it means that there is a statistically significant difference between the two periods. 
# In this case, it's checking if the performance was significantly better during Covid-19. It assigns 'YES' if it is and 'NO' if it's not.

# Store the p-value of the t-test
better_w_pvalue = ttest_result_wmt.pvalue

# Print the results of the analysis
print(f'Difference in IBM and Walmart returns: {iw_diff}\np-value: {iw_pvalue}')
print(f'Average return for IBM/Walmart portfolio: {iwp_rr}')
print(f'Average return for Microsoft/Amazon portfolio: {map_rr}')
print(f'Portfolio with higher returns: {best_portfolio}')
print(f'Walmart performed better pre-Covid: {better_w_covid}\np-value: {better_w_pvalue}')
