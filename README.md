# Resume Category Classification

This project aims to classify resumes into predefined categories using natural language processing (NLP) techniques and machine learning models.

## Methodology

1.  **Data Loading and Initial Exploration:**
    *   The project begins by loading resume data from a CSV file into a pandas DataFrame.
    *   Initial exploration included checking data dimensions (`df.shape`), identifying missing values (`df.isnull().sum()`), and detecting duplicate entries (`df.duplicated()`).
    *   The 'ID' column was dropped, and categories with fewer than 10 entries were filtered out to ensure sufficient data for training.

2.  **Text Preprocessing:**
    *   The raw resume text (`Resume_str` column) underwent several preprocessing steps to clean and standardize the data:
        *   **Lowercasing:** All text was converted to lowercase.
        *   **Punctuation Removal:** Punctuation marks were removed.
        *   **Extra Space Removal:** Multiple spaces were consolidated into single spaces.
        *   **Stopword Removal:** Common English stopwords (e.g., 'the', 'is', 'a') were removed to focus on more meaningful terms.
        *   **Lemmatization:** Words were reduced to their base or root form (e.g., 'running' to 'run') to minimize variations.

3.  **Feature Engineering:**
    *   **TF-IDF Vectorization:** The preprocessed text data was transformed into numerical feature vectors using `TfidfVectorizer`. TF-IDF (Term Frequency-Inverse Document Frequency) assigns weights to words based on their frequency in a document and across the entire corpus, reflecting their importance.

4.  **Target Encoding:**
    *   The categorical `Category` labels were converted into numerical representations using `LabelEncoder`, making them suitable for machine learning algorithms.

5.  **Data Splitting:**
    *   The dataset was split into training and testing sets (80% training, 20% testing) to evaluate model performance on unseen data.

6.  **Model Training and Evaluation:**
    *   **Support Vector Machine (SVM):** An SVM model (`SVC`) was trained on the TF-IDF features. Its performance was evaluated using accuracy score, classification report, and confusion matrix.
    *   **Multinomial Naive Bayes (MNB):** A Multinomial Naive Bayes model (`MultinomialNB`) was also trained and evaluated using the same metrics.

7.  **Hyperparameter Tuning (Started for SVM):**
    *   Hyperparameter tuning for the SVM model was initiated using `GridSearchCV` to find the optimal `C` and `kernel` parameters to potentially improve its accuracy.

## Accuracy and Performance

*   **Support Vector Machine (SVM):** Achieved an accuracy of approximately **58.35%**.
*   **Multinomial Naive Bayes (MNB):** Achieved an accuracy of approximately **52.72%**.

**Comparison:** The SVM model generally outperformed the Multinomial Naive Bayes model in initial evaluations.

**Observations:** Both models showed limitations, particularly struggling with certain categories where precision, recall, and f1-scores were very low (or zero). This indicates potential challenges with class imbalance or the distinctiveness of features for these categories.

**Single Resume Prediction:** A test conducted with a sample resume for 'CUSTOMER SERVICES' successfully predicted the category as 'BANKING'. While this single correct prediction is positive, the overall classification reports suggest significant room for improvement across all categories.

## Future Work

*   Complete and evaluate the results of the hyperparameter tuning for the SVM model.
*   Implement strategies to address the poor performance on specific categories, such as exploring more advanced text embeddings (e.g., Word2Vec, BERT), different model architectures, or techniques for handling imbalanced datasets.
*   Further refine text preprocessing steps.
