import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_20newsgroups
import pickle
from sklearn.model_selection import train_test_split

# Load the 20 Newsgroups dataset

filename = 'clf_model.sav'
model = pickle.load(open(filename, 'rb'))


# Streamlit app
def main():
    newsgroups = fetch_20newsgroups(subset="all")
    categories = ['alt.atheism',
                  'comp.graphics',
                  'comp.os.ms-windows.misc',
                  'comp.sys.ibm.pc.hardware',
                  'comp.sys.mac.hardware',
                  'comp.windows.x',
                  'misc.forsale',
                  'rec.autos',
                  'rec.motorcycles',
                  'rec.sport.baseball',
                  'rec.sport.hockey',
                  'sci.crypt',
                  'sci.electronics',
                  'sci.med',
                  'sci.space',
                  'soc.religion.christian',
                  'talk.politics.guns',
                  'talk.politics.mideast',
                  'talk.politics.misc',
                  'talk.religion.misc']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.20,
                                                        random_state=42)

    # Vectorize the X_training data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_X_train_tfidf = vectorizer.fit_transform(X_train)

    st.title("20 Newsgroups Text Classification")
    st.write("This app classifies text into one of the 20 Newsgroups categories.")

    # Text input box
    text_input = st.text_area("Enter your text here:")

    # Classify button
    if st.button("Classify"):
        text_input_vec = vectorizer.transform([text_input])
        prediction = model.predict(text_input_vec)
        st.write("Predicted category:", categories[prediction[0]])


if __name__ == "__main__":
    main()