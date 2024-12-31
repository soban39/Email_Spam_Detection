# Email_Spam_Detection
Spam Email Detection Project Documentation:
Project Concept:
The goal of this project is to develop a machine learning model that automatically classifies emails as spam or non-spam. The model is trained using a labeled dataset to recognize patterns commonly found in spam emails. By automating spam detection, organizations and individuals can reduce distractions and improve efficiency in handling email communications.
Problem Statement:
Spam emails are a persistent issue, causing productivity losses and security risks. This project addresses the problem by leveraging Artificial Intelligence to:
	Detect patterns in email text.
	Accurately classify emails as spam or ham using a trained model.
By using machine learning, the project automates spam detection and enhances email filtering systems.

Dataset:
	Name: spam.csv
	Source: Kaggle
	Description: Contains labeled email messages, where spam denotes unwanted emails and ham denotes legitimate emails.
Data Statistics:
	Total Records: 5572
	Spam Emails: 747
	Ham Emails: 4825
	No missing values or null entries were found.
	The dataset was critically evaluated for duplicates and missing values, ensuring a clean and reliable data source for the model.

Project Steps:
1. Data Acquisition
	Obtained the spam.csv dataset from Kaggle.
	Evaluated the dataset for missing values, duplicates, and class balance.
	Cleaned the data by removing unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4).
2. Exploratory Data Analysis (EDA)
	Visualizations:
o	Count plot showing the distribution of spam vs. ham emails.
o	Word clouds highlighting frequently used words in spam and ham emails.
	Insights:
o	Common words in spam emails include "win," "offer," and "free."
o	Ham emails often include terms like "project," "meeting," and "team."
3. Feature Engineering
	Techniques:
o	Cleaned text data by tokenizing and stemming using the Porter Stemmer.
o	Extracted numerical features using TfidfVectorizer.
o	For deep learning models, tokenized and padded sequences with:
	Vocabulary Size (input_dim): 8921
	Maximum Sequence Length (maxlen): 171
This step ensured compatibility with both traditional ML models and advanced architectures.
4. Model Training
        Logistic Regression:-
	Accuracy:
o	Training Data: 98.5%
o	Test Data: 96.2%
        Random Forest Classifier:-
	Accuracy:
o	Training Data: 98.7%
o	Test Data: 97.4%
        Deep Learning Model (LSTM):-
	Architecture:
o	Embedding Layer: input_dim=8921, output_dim=128, input_length=171
o	Two LSTM layers with 64 and 32 units respectively.
o	Dense Layer with Sigmoid activation.

	Accuracy:
o	Test Data: 95.8%
5. Model Evaluation
	Assessed model performance using:
o	Accuracy: Overall correctness of the predictions.
o	Precision: Proportion of correctly identified spam emails.
o	Recall: Ability to detect all actual spam emails.
o	Specificity: Correct classification of non-spam emails.
	Confusion Matrix:
o	Provided a visual comparison of predicted vs. actual classifications.
6. Bonus Task: Evaluation
	Explored potential improvements by:
o	Training additional models, including Random Forest and LSTM.
o	Experimenting with hyperparameter tuning for deep learning models.
7. Deep Dive
	Deep Dive 4: Feature Engineering
o	Focused on advanced feature extraction using TfidfVectorizer.
o	Preprocessed text data by tokenizing, stemming, and removing stop words.
	Deep Dive 5: Model Training
o	Built and trained a deep learning model (LSTM) for unstructured text data.
o	Experimented with a custom architecture to handle sequence data effectively.









Tools and Libraries:
	Programming Language:
o	Python 3.x
	Libraries
o	Data Handling: pandas, numpy
o	Visualization: matplotlib, seaborn
o	NLP: nltk, scikit-learn
o	Deep Learning: TensorFlow, Keras
o	Miscellaneous: WordCloud
These tools were chosen for their efficiency in handling text data, building models, and visualizing insights.

Results and Insights:
	Logistic Regression provided a robust baseline with high accuracy and precision.
	Random Forest improved test accuracy slightly by leveraging ensemble learning.
	The deep learning model performed comparably but is more suited for larger datasets or unstructured text.
Trade-offs:
	Logistic Regression: Simpler and faster, ideal for this dataset.
	LSTM: More computationally expensive, better for more complex data. Fine-tuning could improve performance in real-world applications.

Resources and References:
I.	Understanding Spam Filtering and Detection
II.	UCI SMS Spam Dataset
III.	Text Preprocessing in NLP
IV.	Logistic Regression in Machine Learning
V.	Introduction to LSTMs
VI.	Python Libraries Documentation

