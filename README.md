# Restaurant ML Projects

Three machine learning projects on a restaurant dataset (9,551 restaurants, 21 columns), built with Python, pandas, scikit-learn and matplotlib. The tasks come from the Cognifyz Machine Learning task list.

## Projects

### Task 1: Restaurant rating prediction (regression)
Predicts a restaurant's aggregate rating from votes, cost, price range, location, delivery options and number of cuisines.

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Random Forest | 0.340 | 0.253 | 0.627 |
| Decision Tree | 0.376 | 0.279 | 0.542 |
| Linear Regression | 0.431 | 0.340 | 0.400 |

Votes, longitude and latitude were the most important features.

### Task 2: Restaurant recommendation (content-based)
Recommends restaurants by cuisine, price range and rating using cosine similarity. Includes "recommend by preferences" and "find similar restaurants" functions.

### Task 3: Cuisine classification
Predicts a restaurant's primary cuisine among the 10 most common cuisines. Accuracy is about 0.45 (macro precision 0.32, macro recall 0.26). The confusion matrix shows the model leans towards the most common cuisines, because location, votes and cost say little about the food served.

## Key decisions
- Removed 2,148 unrated restaurants (rating 0.0), which are not real ratings.
- Dropped "Rating color" and "Rating text" to avoid data leakage.
- Dropped ID, name and address columns because they are unique per row.

## Limitations
- Votes is a strong predictor but is unavailable for brand-new restaurants.
- Latitude and longitude may act as a stand-in for city or country.
- Cuisine classification uses only the first listed cuisine.
- Recommendation quality is measured with a proxy metric because there is no real user feedback.

## How to run
1. Clone the repo and run `pip install -r requirements.txt`.
2. Place the dataset file `Dataset .csv` in the project root. It is not included in this repo.
3. Open a notebook in VS Code or Jupyter and run all cells.