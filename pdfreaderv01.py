import streamlit as st
import pandas as pd
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
##########################################################

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file) 
    text = ''
    num_pages = len(pdf_reader.pages)
    for page in range(num_pages):
        text += pdf_reader.pages[page].extract_text()
    return text

############################################################

nltk.download("stopwords")

############################################################
model_name = "deepset/roberta-base-squad2"

############################################################
#Function to calculate word frequencies 
def generate_word_freq(text):
    token=re.findall('\w+', text)
    token = [c for c in token if not c.isnumeric()] #remove numeric values
    words=[]
    for word in token:
        words.append(word.lower())
    
    sw=nltk.corpus.stopwords.words('english')
    
    # get the list without stop words
    words_ne=[]
    for word in words:
        if word not in sw:
            words_ne.append(word)
    sns.set_style('darkgrid')
    nlp_words=nltk.FreqDist(words_ne)
    nlp_words.plot(15);

############################################################

# Function to generate a summary using Hugging Face Transformers
def generate_summary(text):
    summarization_pipeline = pipeline("summarization") 
    summary = summarization_pipeline(text, max_length=50, min_length=30, do_sample=True, clean_up_tokenization_spaces=True)
    return summary[0]['summary_text']

# Function to classify text using Hugging Face Transformers
def generate_sentiment(text):
    classifier = pipeline("text-classification")
    outputs = classifier(text)
    sent_df = pd.DataFrame(outputs)
    return sent_df

# Function to answer questions using Hugging Face Transformers
def answer_questions(text, question):
    question_answering_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
    answer = question_answering_pipeline(question=question, context=text)
    return answer['answer']

# Streamlit app
def main():
    st.title("PDF Summary and Question Answering App")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.subheader("Sample Contents:")
        pdf_text = extract_text_from_pdf(uploaded_file)
        sample_text = pdf_text[0:100]
        st.text(sample_text)

        st.subheader("Most frequent words")
        st.pyplot(generate_word_freq(pdf_text))


        st.subheader("Sentiment:")
        sentiment = generate_sentiment(pdf_text)
        st.write(sentiment)

        st.subheader("Highlight:")
        summary = generate_summary(pdf_text)
        st.write(summary)


        st.subheader("Question Answering:")
        question = st.text_input("Ask a question about the document:")
        if st.button("Get Answer"):
            if question:
                answer = answer_questions(pdf_text, question)
                st.write("Answer:", answer)
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
