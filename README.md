# Multilingual-Financial-Multiple-Choice-Question-Answering
Because there is so much financial information online now, like news articles, reports, and company announcements, it is getting harder to organize and understand financial text data by hand. It can be hard for computers to understand financial documents because they often have specialized terms, language specific to a certain field, and subtle meanings that depend on the context.
This project is based on Task 1 of the CLEF 2026 FinMMEval Lab. The goal of this task is to make models that can automatically sort financial text into the right domain categories. The job is to look at the content of financial texts and put them into one of several predefined financial domains, like finance, technology, taxes, or government regulation.
The main goal of this project is to classify financial domains, which means that the system needs to figure out what financial category a piece of text belongs in. This is an important job because putting financial data into structured categories makes it easier for analysts, investors, and regulatory bodies to find the information they need, make decisions, and keep track of what they know.

1. Project Overview

Automatically categorizes short financial texts (headlines, deal summaries) into 6 categories
Finance, Technology, Tax & Accounting, Business & Management, Government & Controls, and Industry. 
Designed for CLEF 2026 FinMMEval Lab Task 1. No translation supports Greek, Spanish and English.

----------------------------------------------------------------------

2. Dataset

- English  - FLARE-MultiFin (448 samples): https://huggingface.co/datasets/TheFinAI/flare-multifin-en
- Greek    - Plutus MultiFin (235 samples): https://huggingface.co/datasets/TheFinAI/plutus-multifin
- Spanish  - FLARE-ES MultiFin (203 samples): https://huggingface.co/datasets/TheFinAI/flare-es-multifin
- English .parquet file must be uploaded manually to /content/ in Colab before running.

----------------------------------------------------------------------
3. Requirements

- Platform: Google Colab

- Language: Python 3

- All packages install automatically when you run the first cell:
    pip install sentence-transformers datasets deep-translator transformers torch xgboost scikit-learn accelerate

- sentence-transformers  : SBERT multilingual embeddings

- transformers           : FinBERT tokenizer and model

- torch                  : deep learning backend

- datasets               : HuggingFace dataset loader

- xgboost                : gradient-boosted classifier

- scikit-learn           : Logistic Regression, TF-IDF, metrics, preprocessing

- deep-translator        : translation utilities

- pandas, numpy          : data handling

- matplotlib, seaborn    : visualizations

- accelerate             : HuggingFace training optimization

- GPU recommended for FinBERT: Runtime > Change runtime type > T4 GPU

----------------------------------------------------------------------


5. Key Methods Used

- TF-IDF baseline with Logistic Regression and XGBoost- SBERT (paraphrase-multilingual-mpnet-base-v2) for the multilingual semantic embeddings

- SBERT supports English, Greek and Spanish natively, no translation required

- Reduced and normalized SBERT features using StandardScaler and SelectKBest
Logistic Regression and XGBoost on SBERT embeddings for all three languages

- FinBERT (ProsusAI/finbert) separately fine-tuned for each language

- Training FinBERT with weighted cross-entropy loss to tackle class imbalance

----------------------------------------------------------------------

6. Output

Example Program Input and Output:

    Input:  "Apple acquires fintech startup for digital payments"
    Output: Technology

    Input:  "Government introduces new capital gains tax reform"
    Output: Tax & Accounting

    Input:  "Merger of regional banks increases market share"
    Output: Finance

    Input:  "EU regulatory framework tightens rules on hedge funds"
    Output: Government & Controls

    Input:  "Automotive supplier expands production in Southeast Asia"
    Output: Industry

    Input:  "Συγχωνευση τραπεζων για επεκταση στην αγορα"  (Greek)
    Output: Finance

    Input:  "Nueva regulacion fiscal para empresas tecnologicas"  (Spanish)
    Output: Tax & Accounting

Best Results (Logistic Regression + SBERT):

    English  - Accuracy: 0.789  |  F1 Macro: 0.704
    Greek    - Accuracy: 0.915  |  F1 Macro: 0.830
    Spanish  - Accuracy: 0.780  |  F1 Macro: 0.749

----------------------------------------------------------------------

7. Notes

- F1 Macro is the main metric. The dataset is imbalanced so accuracy alone can be misleading.
- Logistic Regression + SBERT gives the best performance for all three languages.- FinBERT works only well for English. It wasn’t trained on Greek or Spanish.
- Govt & Controls is the hardest class because of few samples and overlap with Finance.
- The English .parquet file needs to be in /content/ before running, or the notebook will fail.
