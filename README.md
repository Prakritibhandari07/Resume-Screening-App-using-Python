# Resume-Screening-App-using-Python

A simple web app built with **Streamlit** that classifies resumes into job categories using a pre-trained machine learning model (TF-IDF + classifier). Upload a resume in `.txt` or `.pdf` format, and the app predicts the most relevant job category (e.g., HR, Engineering, Finance, etc.).

🔗 **Live Demo:** [https://resume-screening-app-using-python-cbzasevgovm83jpro4s5fj.streamlit.app/]

---

## Features

- Upload resumes in `.txt` or `.pdf` format
- Cleans and preprocesses resume text (removes URLs, mentions, hashtags, special characters, etc.)
- Extracts text from PDF resumes using `pypdf`
- Predicts job category using a pre-trained TF-IDF vectorizer and classifier
- Maps numeric prediction to a human-readable category label

---

## Tech Stack

- [Streamlit](https://streamlit.io/) — web app framework
- [scikit-learn](https://scikit-learn.org/) — TF-IDF vectorizer and classification model
- [pypdf](https://pypi.org/project/pypdf/) — PDF text extraction
- [NLTK](https://www.nltk.org/) — text preprocessing utilities
- [NumPy](https://numpy.org/)

---

## Project Structure

```
Resume Screening using python/
│
├── app.py              # Main Streamlit application
├── clf.pkl              # Pre-trained classifier model
├── tfidf.pkl             # Pre-trained TF-IDF vectorizer
├── requirements.txt        # Python dependencies
└── README.md             # Project documentation
```

---

## Installation & Setup (Local)

1. **Clone the repository**
   ```bash
   git clone [Insert your GitHub repo link here]
   cd "Resume Screening using python"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate    # On Windows
   source venv/bin/activate # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`) in your browser.

---

## Deployment

This app can be deployed for free using **Streamlit Community Cloud**:

1. Push your project (including `app.py`, `clf.pkl`, `tfidf.pkl`, and `requirements.txt`) to a GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, branch, and `app.py` as the main file.
4. Click **Deploy**.

Once deployed, replace the placeholder at the top of this README with your live app link.

---

## Supported Job Categories

The model predicts one of the following categories:

HR, Designer, Information Technology, Teacher, Advocate, Business Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public Relations, Banking, Arts, Aviation

---

## Notes & Limitations

- PDF text extraction works best on text-based PDFs. Scanned/image-based PDFs (with no embedded text layer) will not extract properly and would require OCR (e.g., `pytesseract`) for support.
- `.docx` file support is not yet implemented; only `.txt` and `.pdf` are currently supported.
- Model accuracy depends entirely on the quality and scope of the training data used to build `clf.pkl` and `tfidf.pkl`.

---


## Author

[Insert your name / GitHub profile link here]
