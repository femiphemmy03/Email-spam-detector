# Email Spam Detector
Internship assignment for Gateway English Learning Hub.

## Overview
This project generates 500 emails (250 genuine, 250 spam) and classifies them as spam or good using Python and Naive Bayes. It extracts URLs as a feature.

## Files
- `generate_emails.py`: Generates 500 mock .eml files.
- `spam_detector.py`: Classifies emails and extracts URLs.
- `email_analysis.csv`: Output with email data and predictions.
- `requirements.txt`: Dependencies.
- `emails/`: Contains generated emails (not included in submission due to size).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run `generate_emails.py` to create emails.
3. Run `spam_detector.py` to classify emails and generate `email_analysis.csv`.

## Results
Achieved [insert accuracy from spam_detector.py]% accuracy on the test set.

## Notes
- Emails were generated randomly due to insufficient data in the provided Google Drive.
- The spam detection model uses TF-IDF features and Naive Bayes.
