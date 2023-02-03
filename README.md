# Data-Analysis---Pandas
I will be using Python Pandas & Python Matplotlib to analyze and answer some business questions about 12 months worth of sales data. The data contains hundreds of thousands of electronics store purchases broken down by month, product type, cost, purchase address, etc.
Firstly, I started by Concatenating multiple csvs together to create a new DataFrame (pd.concat)
Then I started cleaning the data: Drop NaN values from DataFrame,Removing rows based on a condition,Change the type of columns (to_numeric, to_datetime, astype)
Once I cleaned up the data a bit, the data exploration begins. In this I explore some high level business questions related to the data:
What was the best month for sales? How much was earned that month?
What city sold the most product?
What is the best time to display advertisemens to maximize the likelihood of customer’s buying product?
What products are most often sold together?
What product sold the most? Why do you think it sold the most?
To answer these questions I used different pandas & matplotlib methods. They include:
Adding columns
Parsing cells as strings to make new columns (.str)
Using the .apply() method
Using groupby to perform aggregate analysis
Plotting bar charts and lines graphs to visualize the results
Labeling the graphs
