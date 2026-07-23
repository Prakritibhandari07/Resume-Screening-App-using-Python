import numpy
import re
import pickle
import streamlit as st
import nltk
from pypdf import PdfReader
import io

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

def cleanResume(txt):
    cleanTxt=re.sub(r'http\S+\s',' ',txt)
    cleanTxt=re.sub(r'@\S+',' ',cleanTxt)
    cleanTxt=re.sub(r'#\S+\s',' ',cleanTxt)
    cleanTxt=re.sub(r'RT|CC',' ',cleanTxt)
    cleanTxt=re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~"""),' ',cleanTxt)
    cleanTxt=re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt=re.sub(r'\s+',' ',cleanTxt)

    return cleanTxt.strip()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text   

#web app
def main():
    st.title("Resume Screening App")
    uploaded_file=st.file_uploader('Upload Resume',type=['txt','pdf','docs'])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()

        if file_ext == 'pdf':
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_ext == 'txt':
            resume_bytes=uploaded_file.read()
            try:
                resume_text=resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

        else:
            st.write("Unsupported file type. Please upload .pdf or .txt")
            return   

        cleaned_resume=cleanResume(resume_text)
        cleaned_resume=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

        
        category_mapping = {
            0: "HR",
            1: "DESIGNER", 
            2: "INFORMATION-TECHNOLOGY",
            3: "TEACHER",
            4: "ADVOCATE",
            5: "BUSINESS-DEVELOPMENT",
            6: "HEALTHCARE",
            7: "FITNESS",
            8: "AGRICULTURE",
            9: "BPO",
            10: "SALES",
            11: "CONSULTANT",
            12: "DIGITAL-MEDIA",
            13: "AUTOMOBILE",
            14: "CHEF",
            15: "FINANCE",
            16: "APPAREL",
            17: "ENGINEERING",
            18: "ACCOUNTANT",
            19: "CONSTRUCTION",
            20: "PUBLIC-RELATIONS",
            21: "BANKING",
            22: "ARTS",
            23: "AVIATION",
        }

        category_name=category_mapping.get(prediction_id,"unknown")
        st.write("Predicted Category:", category_name)



#python main

if __name__ == "__main__":
    main()