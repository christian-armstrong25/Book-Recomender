# Book Recommendation System

A personalized book recommendation system that uses matrix factorization to suggest books based on your reading history. The system can work with both the built-in dataset and your personal Goodreads library export.

## Features

- Matrix factorization-based collaborative filtering
- Support for Goodreads library exports
- Personalized recommendations based on your reading history
- Recommendations from your to-read list
- Diverse recommendations using tag-based similarity
- Handles both existing users and new users with their own reading history

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - torch
  - matplotlib (optional, for visualization)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Book-Recommender.git
cd Book-Recommender
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - The system uses the Goodreads 10k dataset
   - Place the dataset files in the `goodbooks-10k-1.0` directory
   - Required files:
     - books.csv
     - ratings.csv
     - book_tags.csv
     - tags.csv
     - to_read.csv

## Usage

### Training a New Model

To train a new recommendation model:
```bash
python book_recommender.py --train
```

This will:
- Load the dataset
- Train the model using matrix factorization
- Save the trained model to the `models` directory

### Getting Recommendations for a Random User

To get recommendations for a random user from the dataset:
```bash
python book_recommender.py
```

This will:
- Load the trained model
- Select a random user with both ratings and to-read entries
- Show two sets of recommendations:
  1. Top 10 books from their to-read list
  2. Top 10 books they might like

### Using Your Goodreads Data

To get personalized recommendations based on your Goodreads library:

1. Export your Goodreads library:
   - Go to your Goodreads profile
   - Click "Import and Export" in the left sidebar
   - Click "Export Library"
   - Save the CSV file

2. Run the recommender with your export:
```bash
python book_recommender.py --goodreads goodreads_library_export.csv
```

This will:
- Load your Goodreads data
- Map your book IDs to the dataset
- Create a personalized profile
- Show recommendations based on your reading history

## Output

The system provides two types of recommendations:

1. From your to-read list:
   - Books you've marked as "to-read" on Goodreads
   - Ordered by predicted rating
   - Includes title, author, and predicted rating

2. New recommendations:
   - Books you haven't rated or marked as to-read
   - Based on your reading preferences
   - Includes title, author, and predicted rating

## How It Works

The recommendation system uses matrix factorization to learn user preferences:
- Decomposes the user-item rating matrix into user and item factors
- Uses SVD initialization for better convergence
- Optimizes the factors using gradient descent
- Makes predictions through matrix multiplication
- Ensures diverse recommendations using tag-based similarity

## Troubleshooting

Common issues and solutions:

1. "Required file not found":
   - Ensure all dataset files are in the `goodbooks-10k-1.0` directory
   - Check file names match exactly

2. "User not found":
   - Make sure your Goodreads export is in the correct format
   - Check that book IDs are being mapped correctly

3. "No recommendations found":
   - Try training a new model with `--train`
   - Check that your Goodreads data contains enough ratings