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
