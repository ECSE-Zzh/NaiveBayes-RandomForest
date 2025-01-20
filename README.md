# Introduction/Background
This project evaluates the performance of Random Forest (RF), Naive Bayes, and large language models (LLMs)—DistilBERT,
BERT-base, and RoBERTa-base—on text classification tasks using single-labeled examples from the GoEmotions dataset.
The dataset comprises 58,000 human-annotated Reddit comments categorized into 27 emotion labels and a Neutral class,
with 83% of the examples being single-labeled. It is split into 43,410 training samples, 5,426 validation samples, and
5,427 test samples.

Traditional models, such as RF and Naive Bayes, performed reasonably well on majority
classes but struggled with the imbalanced distribution of minority classes. RF achieved a macro-average F1-score of 0.40,
while Naive Bayes scored 0.28. Techniques like SMOTE and undersampling were employed to address class imbalance,
leading to improved performance on minority classes but a decline in overall accuracy.

# Data preprocessing & exploratory analysis

To simplify the dataset, multi-labeled samples were excluded, retaining only single-labeled entries. The label column
was flattened, and unnecessary columns were dropped to reduce computational overhead. Text data was standardized by
converting all text to lowercase, removing non-alphanumeric characters, and eliminating stop words (e.g., ”is,” ”the”) that
do not contribute significantly to the content. For Random Forest (RF), the TfidfVectorizer was employed to transform
the text into numerical features. For the Naive Bayes model, CountVectorizer was used to create a sparse matrix of term
frequencies in the style of Bag-of-Words.

Class distribution analysis revealed significant imbalances, with ”neutral” being the dominant class and emotions like
”grief,” ”pride,” ”nervousness,” and ”relief” being underrepresented. This imbalance adversely affected the performance of
both the Random Forest (RF) and Naive Bayes models.

# Naive Bayes Implementation
MultinomialNB was developed from scratch to implement the Naive Bayes algorithm. Multinomial Naive Bayes was
chosen due to its effectiveness in handling feature counts, making it well-suited for text classification tasks.
Laplace smoothing is applied to handle zero probabilities, controlled by two hyperparameters: laplace smoothing, which
enables or disables smoothing, and alpha, the smoothing constant. The model’s parameters include priors, representing the
class frequencies, likelihoods, which store conditional probabilities of each feature given a class, and classes, which contains
the unique class labels.

The fit function trains the model using the feature matrix x and class labels y. It computes prior probabilities, filters the
training samples by class, and calculates the likelihoods, applying Laplace smoothing if enabled. The predict function takes
the feature matrix x as input, converts priors and likelihoods to logarithms to avoid numerical underflow, and calculates the
log posterior probabilities for each class. It returns the class with the highest log posterior for each sample. The model’s
performance is evaluated using the evaluate acc function, which computes the accuracy score by comparing the predicted
labels with the true labels.
