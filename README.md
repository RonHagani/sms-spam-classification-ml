# SMS Spam Classification using SimpleKNN

This project was created as part of a Machine Learning course assignment.

The goal is to classify SMS messages as either regular messages (`ham`) or spam messages (`spam`).  
We used a simple KNN classifier that was implemented manually, together with several text representation methods.

---

## Authors

- Ron Hagani
- Stav Borodovski

---

## Dataset

The dataset was taken from Kaggle:

**SMS Spam Collection - A More Diverse Dataset**

Dataset link:  
https://www.kaggle.com/datasets/thedevastator/sms-spam-collection-a-more-diverse-dataset/data

The file used in this project is:

```text
train.csv
```

The dataset contains two columns:

- `sms` — the text message
- `label` — the message class

The labels are:

- `0` — regular message
- `1` — spam message

In the notebook, the labels were converted to:

```text
0 → ham
1 → spam
```

---

## What We Did

The main steps in the project were:

1. Loaded the dataset
2. Checked the class distribution
3. Cleaned the text
4. Split the data into train and test sets
5. Balanced the training data using undersampling
6. Converted the text into numerical vectors
7. Implemented a simple KNN classifier
8. Compared several vectorization methods
9. Used 5-Fold Cross Validation
10. Tested the final model on the test set

---

## Text Representation

We tested several ways to represent the SMS messages as numerical vectors:

- CountVectorizer
- TF-IDF with unigrams
- TF-IDF with unigrams and bigrams
- TF-IDF with SVD

The goal was to check which representation works best with the KNN model.

---

## Model

The model used in this project is a custom implementation of KNN called:

```text
SimpleKNN
```

The model uses cosine similarity to compare SMS messages.

For each test message, it finds the most similar messages in the training set and predicts the class based on the nearest neighbors.

---

## Handling Class Imbalance

The dataset contains more regular messages than spam messages.

To handle this, we used undersampling on the training set only.  
The test set was not changed, so the final evaluation stayed realistic.

---

## Model Selection

We used a manual Grid Search with 5-Fold Cross Validation.

We tested different vectorizers and different K values:

```text
K = 3, 5, 7, 9
```

The best model was selected according to the F1-score of the spam class.

---

## Evaluation

The final model was evaluated using:

- Classification Report
- F1-score for spam
- Confusion Matrix
- Example predictions

---

## Results

After running the notebook, the best model was:

```text
Best Vectorizer: TfidfVectorizer_Unigram
Best K: 5
F1-score for spam: 0.75483
```

---

## Files

```text
.
├── README.md
├── spam_classification.ipynb
└── train.csv
```

---

## How to Run

Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Then open:

```text
spam_classification.ipynb
```

Run the notebook cells from top to bottom.

---

## Conclusion

In this project, we built a simple spam classification model using KNN.

The project shows the basic machine learning process: loading data, cleaning text, extracting features, selecting a model, and evaluating the final results.

The results show that a simple KNN model can work reasonably well for SMS spam classification when the text is cleaned, represented properly, and the training data is balanced.