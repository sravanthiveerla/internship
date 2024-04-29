import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Function to load data
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

# Function to load trained models
def load_models():
    count_vectorizer = joblib.load(r"E:\Sravanthi Files\count_vectorizer.pkl")
    tfidf_transformer = joblib.load(r"E:\Sravanthi Files\tfidf_transformer.pkl")
    tfidf_matrix = joblib.load(r"E:\Sravanthi Files\tfidf_matrix.pkl")
    return count_vectorizer, tfidf_transformer, tfidf_matrix

# Function to preprocess query
def preprocess_query(query, count_vectorizer):
    query_vector = count_vectorizer.transform([query])
    return query_vector

# Function to retrieve similar documents
def retrieve_similar_documents(query, count_vectorizer, tfidf_transformer, tfidf_matrix, data):
    query_vector = preprocess_query(query, count_vectorizer)
    query_tfidf = tfidf_transformer.transform(query_vector)
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
    sorted_indices = similarity_scores.argsort()[0][::-1]
    top_indices = sorted_indices[:5]
    return data.iloc[top_indices]["file_content_chunks"].tolist()

# Main function
def main():
    st.title('Search Engine')

    data = load_data(r"E:\Sravanthi Files\Sub_Titles1.csv")

    count_vectorizer, tfidf_transformer, tfidf_matrix = load_models()

    query = st.text_input('Enter movie name:', '')

    if st.button('Explore here'):
        if query:
            retrieved_documents = retrieve_similar_documents(query, count_vectorizer, tfidf_transformer, tfidf_matrix, data)
            st.subheader('Top 5 documents related to the query:')
            for i, doc in enumerate(retrieved_documents, 1):
                st.write(f"Document {i}: {doc}")

if __name__ == '__main__':
    main()