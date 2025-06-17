# MachineLearning1
# 🎬 Movie Recommendation System

This is my first machine learning project: **Movie Recommendation System** that predicts whether a user would like a movie and suggests similar movies based on genre.

## 🚀 Features

- **Random Forest Classifier** to predict if a user will like a movie
- **Content-based Filtering** using movie tags, genres, and release decades
- **TF-IDF Vectorization** on tags (500 features)
- **Genre Similarity Search** using FAISS (Facebook AI Similarity Search)
- **Interactive CLI**: Input a movie title or ID and get a like/dislike prediction with confidence
- **Visualization**: Confusion matrix and precision-recall curve for model evaluation

## 🧠 Data & Preprocessing

- Input: `cleaned_data.csv`
- Features used:
  - TF-IDF of `tag`
  - One-hot encoded `genres`
  - `normalized_rating`, `tag_count`
  - Release `decade` (one-hot encoded)

## 📊 Model Training

- **Algorithm**: `RandomForestClassifier`
- **Test Split**: 80/20
- **Metrics**: Confusion matrix, precision-recall curve, classification report

## 🛠 Tech Stack

`scikit-learn`, `faiss`, `pandas`, `matplotlib`, `seaborn`, `numpy`



