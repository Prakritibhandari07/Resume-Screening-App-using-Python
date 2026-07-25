import numpy
import re
import pickle
import streamlit as st
import nltk
from pypdf import PdfReader
import io

nltk.download('punkt')
nltk.download('stopwords')

# loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    cleanTxt = re.sub(r'http\S+\s', ' ', txt)
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
    cleanTxt = re.sub(r'#\S+\s', ' ', cleanTxt)
    cleanTxt = re.sub(r'RT|CC', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt)
    return cleanTxt.strip()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

category_mapping = {
    0: "HR", 1: "DESIGNER", 2: "INFORMATION-TECHNOLOGY", 3: "TEACHER",
    4: "ADVOCATE", 5: "BUSINESS-DEVELOPMENT", 6: "HEALTHCARE", 7: "FITNESS",
    8: "AGRICULTURE", 9: "BPO", 10: "SALES", 11: "CONSULTANT",
    12: "DIGITAL-MEDIA", 13: "AUTOMOBILE", 14: "CHEF", 15: "FINANCE",
    16: "APPAREL", 17: "ENGINEERING", 18: "ACCOUNTANT", 19: "CONSTRUCTION",
    20: "PUBLIC-RELATIONS", 21: "BANKING", 22: "ARTS", 23: "AVIATION",
}

def classify_text(resume_text):
    cleaned_resume = cleanResume(resume_text)
    vector = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(vector)[0]

    try:
        proba = clf.predict_proba(vector)[0]
        confidence = proba[prediction_id]
    except AttributeError:
        confidence = None

    category_name = category_mapping.get(prediction_id, "Unknown")
    return category_name, confidence

def main():
    st.title("📄 Resume Screening App")
    st.write("Upload a resume (or paste the text) to predict its job category.")

    tab1, tab2 = st.tabs([" Upload File", " Paste Text"])

    resume_text = None

    with tab1:
        uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            if file_ext == 'pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
            elif file_ext == 'txt':
                resume_bytes = uploaded_file.read()
                try:
                    resume_text = resume_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = resume_bytes.decode('latin-1')

    with tab2:
        pasted_text = st.text_area(
            "Paste resume text here",
            placeholder="Paste resume content directly if you don't have a file handy...",
            height=200
        )
        if st.button("Classify Pasted Text", type="primary"):
            if pasted_text.strip():
                resume_text = pasted_text
            else:
                st.warning("Please paste some resume text first.")

    if resume_text:
        category_name, confidence = classify_text(resume_text)
        st.divider()
        if confidence:
            st.success(f"**Predicted Category: {category_name}**  \nConfidence: {confidence:.1%}")
        else:
            st.success(f"**Predicted Category: {category_name}**")

if __name__ == "__main__":
    main()
