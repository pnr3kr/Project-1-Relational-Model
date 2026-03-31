# Budget Dominates the Box Office: New Data Analysis Reveals Which Movie Factors Are Most Strongly Linked to Box Office Success

## Hook:
Every year, movie studios invest millions of dollars into producing and marketing films. However, predicting which movies will be financially successful is extremely difficult. A new data analysis of nearly 9,000 films finds that production budget alone accounts for nearly 73% of a model's ability to predict box office earnings — dwarfing the influence of genre, audience ratings, and runtime combined. Understanding what factors contribute to box office success could help studios make smarter decisions about production and marketing.

## Problem Statement:
The film industry generates large amounts of data about movies, including information about genre, ratings, runtime, and audience engagement. It is still challenging to determine which factors are most strongly associated with higher box office earnings. Studios often rely on experience or intuition when deciding what types of films to produce or promote. Understanding how different movie characteristics relate to financial success could provide valuable insights into what makes a film successful.

## Solution Description:
This project analyzes a combined IMDb and TMDB movie dataset to explore how different movie characteristics relate to box office earnings. 8,999 films with confirmed budget and revenue figures were examined, alongside variables including genre, IMDb ratings, runtime, release year, and original language. Five regression models were trained and compared (Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting) using 5-fold cross-validation and hyperparameter tuning on the top performers.

Gradient Boosting came out as the strongest model, achieving a cross-validation R² of 0.501 and a final test R² of 0.468. Linear models fell behind substantially (Ridge: R² 0.345, Lasso: R² 0.305), suggesting that revenue is shaped by nonlinear interactions that ensemble methods are better equipped to capture.

Feature importance analysis revealed a striking result: production budget alone contributed 72.7% of the model's explanatory power. Average IMDb rating (9.6%), release year (5.4%), and runtime (3.7%) were the next most influential features. Genre and language played relatively minor roles with Drama, Horror, and Crime being the most predictive genres. The analysis provides a data-driven foundation for understanding what makes a film financially successful, and highlights that no amount of critical acclaim or genre selection compensates for underinvestment in production.

## Chart:
![Results Chart](results/revenue_prediction_results.png)