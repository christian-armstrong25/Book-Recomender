"""
============================================================================
📚 Book Recommendation System
============================================================================
"""

# Standard library imports
import os
import argparse
from itertools import product

# Data processing and scientific computing
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim

"""
============================================================================
📊 1. Data Loading and Processing
============================================================================
"""

# Constants
DATA_DIR = 'goodbooks-10k-1.0'
MODEL_DIR = 'models'
REQUIRED_FILES = [
    f'{DATA_DIR}/books.csv',
    f'{DATA_DIR}/ratings.csv',
    f'{DATA_DIR}/book_tags.csv',
    f'{DATA_DIR}/tags.csv',
    f'{DATA_DIR}/to_read.csv'
]

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Load and validate all required datasets."""
    for file in REQUIRED_FILES:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file {file} not found")

    return tuple(pd.read_csv(file) for file in REQUIRED_FILES)

def create_rating_matrices(ratings_df, test_size=0.2, random_state=42):
    """Create train and test rating matrices."""
    train_ratings, test_ratings = train_test_split(
        ratings_df[['user_id', 'book_id', 'rating']],
        test_size=test_size,
        random_state=random_state
    )

    def create_matrix(ratings):
        return ratings.pivot(
            index='user_id',
            columns='book_id',
            values='rating'
        ).fillna(0)

    return create_matrix(train_ratings), create_matrix(test_ratings)

def get_book_details(books_df, book_id):
    """Get book title and author by book_id."""
    try:
        book = books_df[books_df['book_id'] == book_id]
        if len(book) == 0:
            return None
        return {
            'title': book['title'].values[0],
            'author': book['authors'].values[0]
        }
    except Exception as e:
        print(f"Error getting book details: {e}")
        return None

def load_goodreads_data(filepath):
    """
    Load and process Goodreads exported data.

    Args:
        filepath: Path to the Goodreads library export CSV file

    Returns:
        tuple: (ratings_df, to_read_df) containing processed ratings and to_read data
    """
    try:
        # Read Goodreads export
        goodreads_df = pd.read_csv(filepath)

        # Create a new user ID (using a high number to avoid conflicts)
        new_user_id = 1000000  # Keep as integer

        # Process ratings
        ratings_data = []
        for _, row in goodreads_df.iterrows():
            if pd.notna(row['My Rating']):
                ratings_data.append({
                    'user_id': new_user_id,
                    'book_id': int(row['Book Id']),  # Ensure integer
                    'rating': float(row['My Rating'])
                })

        # Process to_read (books marked as "to-read")
        to_read_data = []
        for _, row in goodreads_df.iterrows():
            if pd.notna(row['Exclusive Shelf']) and row['Exclusive Shelf'].lower() == 'to-read':
                to_read_data.append({
                    'user_id': new_user_id,
                    'book_id': int(row['Book Id'])  # Ensure integer
                })

        return pd.DataFrame(ratings_data), pd.DataFrame(to_read_data)

    except Exception as e:
        print(f"Error loading Goodreads data: {e}")
        return None, None

def map_goodreads_to_dataset_ids(goodreads_ratings_df, goodreads_to_read_df, books_df):
    """
    Map Goodreads book IDs to dataset book IDs.

    Args:
        goodreads_ratings_df: DataFrame with Goodreads ratings
        goodreads_to_read_df: DataFrame with Goodreads to_read entries
        books_df: DataFrame with dataset books

    Returns:
        tuple: (mapped_ratings_df, mapped_to_read_df, mapping_stats) with dataset book IDs
    """
    # Create mapping from Goodreads IDs to dataset IDs
    id_mapping = books_df.set_index('goodreads_book_id')['book_id'].to_dict()

    # Map ratings
    mapped_ratings = []
    total_ratings = len(goodreads_ratings_df)
    mapped_ratings_count = 0

    for _, row in goodreads_ratings_df.iterrows():
        if row['book_id'] in id_mapping:
            mapped_ratings.append({
                'user_id': int(row['user_id']),  # Ensure integer
                'book_id': int(id_mapping[row['book_id']]),  # Ensure integer
                'rating': float(row['rating'])
            })
            mapped_ratings_count += 1

    # Map to_read entries
    mapped_to_read = []
    total_to_read = len(goodreads_to_read_df)
    mapped_to_read_count = 0

    for _, row in goodreads_to_read_df.iterrows():
        if row['book_id'] in id_mapping:
            mapped_to_read.append({
                'user_id': int(row['user_id']),  # Ensure integer
                'book_id': int(id_mapping[row['book_id']])  # Ensure integer
            })
            mapped_to_read_count += 1

    # Create mapping statistics
    mapping_stats = {
        'total_ratings': total_ratings,
        'mapped_ratings': mapped_ratings_count,
        'total_to_read': total_to_read,
        'mapped_to_read': mapped_to_read_count
    }

    return pd.DataFrame(mapped_ratings), pd.DataFrame(mapped_to_read), mapping_stats

"""
============================================================================
🤖 2. Recommender System
============================================================================
"""

class Recommender:
    """
    A recommendation system that combines SVD initialization with gradient descent optimization.

    The model uses matrix factorization where:
    - R ≈ U × V^T, where R is the ratings matrix
    - U is the user factors matrix [num_users × n_components]
    - V is the item factors matrix [num_items × n_components]
    """

    def __init__(self, n_components=20, learning_rate=0.001, n_epochs=100):
        """Initialize the recommender system."""
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mean_rating = 0.0
        self.user_index = pd.Index([])
        self.item_index = pd.Index([])
        self.user_factors = None
        self.item_factors = None
        self.optimizer = None

    def _initialize_from_svd(self, ratings):
        """Initialize model parameters using SVD."""
        # Convert to sparse matrix and center the data
        ratings_sparse = csr_matrix(ratings.values)
        self.mean_rating = ratings_sparse.data.mean()
        ratings_sparse.data -= self.mean_rating

        # Perform SVD
        U, _, Vt = svds(ratings_sparse, k=self.n_components)

        # Convert to PyTorch tensors (making copies to avoid negative strides)
        self.user_factors = nn.Parameter(torch.FloatTensor(U.copy()))
        self.item_factors = nn.Parameter(torch.FloatTensor(Vt.T.copy()))

    def fit(self, ratings):
        """Fit the model using gradient descent with SVD initialization."""
        self.user_index = ratings.index
        self.item_index = ratings.columns

        # Initialize from SVD
        self._initialize_from_svd(ratings)

        # Setup optimizer
        self.optimizer = optim.Adam(
            [self.user_factors, self.item_factors],
            lr=self.learning_rate
        )

        # Convert ratings to PyTorch tensor
        ratings_tensor = torch.FloatTensor(ratings.values)
        mask = (ratings_tensor != 0)

        # Training loop
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()

            # Forward pass - ensure proper matrix multiplication
            predictions = torch.matmul(self.user_factors, self.item_factors.t())
            predictions = predictions + self.mean_rating

            # Compute loss only on non-zero ratings
            loss = torch.sum(((predictions - ratings_tensor) * mask) ** 2)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss.item():.4f}")

    def predict_all(self):
        """Get predictions for all user-item pairs."""
        with torch.no_grad():
            # Transpose item_factors for proper matrix multiplication
            predictions = torch.matmul(self.user_factors, self.item_factors.t())
            predictions = predictions + self.mean_rating
            return torch.clamp(predictions, 1.0, 5.0).numpy()

    def predict_user(self, user_id, user_ratings=None):
        """
        Get predictions for a single user.

        Args:
            user_id: The user to get predictions for
            user_ratings: Optional array of ratings for a new user
        """
        if user_ratings is not None:
            # For new users, use their ratings to make predictions
            with torch.no_grad():
                # Convert ratings to tensor and reshape to match item factors
                ratings_tensor = torch.FloatTensor(user_ratings).reshape(1, -1)  # Shape: [1, num_items]
                # Make predictions using item factors
                predictions = torch.matmul(ratings_tensor, self.item_factors)  # Shape: [1, n_components]
                predictions = torch.matmul(predictions, self.item_factors.t())  # Shape: [1, num_items]
                predictions = predictions + self.mean_rating
                return torch.clamp(predictions, 1.0, 5.0).numpy().flatten()
        else:
            # For existing users in the training data
            if user_id not in self.user_index:
                raise ValueError(f"User {user_id} not found")

            user_idx = self.user_index.get_loc(user_id)
            with torch.no_grad():
                predictions = torch.matmul(self.user_factors[user_idx], self.item_factors.t())
                predictions = predictions + self.mean_rating
                return torch.clamp(predictions, 1.0, 5.0).numpy()

    def recommend(self, user_id, n=10, exclude_rated=True, to_read_only=False, user_to_read=None, user_ratings=None, book_tags_df=None, train_matrix=None):
        """
        Get diverse top-N recommendations for a user.
        Uses tag similarity to break ties between high-rated books and ensure diversity.

        Args:
            user_id: The user to get recommendations for
            n: Number of recommendations to return
            exclude_rated: Whether to exclude books the user has already rated
            to_read_only: Whether to only recommend books from the to_read list
            user_to_read: Set of books the user has marked as to_read
            user_ratings: Optional array of ratings for a new user
            book_tags_df: DataFrame containing book tags for diversity calculation
            train_matrix: DataFrame containing the training ratings matrix
        """
        predictions = self.predict_user(user_id, user_ratings)
        recommendations = list(zip(self.item_index, predictions))

        if exclude_rated and user_ratings is not None:
            # For new users, exclude books they've rated
            rated_items = user_ratings != 0
            recommendations = [(item_id, pred) for (item_id, pred), rated
                             in zip(recommendations, rated_items) if not rated]
        elif exclude_rated and train_matrix is not None:
            # For existing users, exclude books from training data
            rated_items = train_matrix.loc[user_id] != 0
            recommendations = [(item_id, pred) for (item_id, pred), rated
                             in zip(recommendations, rated_items) if not rated]

        # Filter based on to_read status
        if user_to_read is not None:
            if to_read_only:
                # Only include books in user's to_read list
                recommendations = [(book_id, pred) for book_id, pred in recommendations
                                 if book_id in user_to_read]
            else:
                # Exclude books in user's to_read list
                recommendations = [(book_id, pred) for book_id, pred in recommendations
                                 if book_id not in user_to_read]

        # Sort by predicted rating
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Get all 5-star predictions (4.95 or higher)
        five_star_books = [(book_id, pred) for book_id, pred in recommendations if pred >= 4.95]

        if len(five_star_books) > n and book_tags_df is not None:
            # Calculate tag similarity matrix for 5-star books (optimized)
            book_ids = [book_id for book_id, _ in five_star_books]

            # Pre-compute tag sets for all books
            book_tags_dict = {}
            for book_id in book_ids:
                # Use the correct column names from book_tags_df
                book_tags = book_tags_df[book_tags_df['goodreads_book_id'] == book_id]['tag_id']
                book_tags_dict[book_id] = set(book_tags)

            # Calculate similarities only for books with tags
            tag_similarities = np.zeros((len(book_ids), len(book_ids)))
            for i, book_id1 in enumerate(book_ids):
                tags1 = book_tags_dict[book_id1]
                if not tags1:
                    continue

                for j, book_id2 in enumerate(book_ids):
                    if i != j:
                        tags2 = book_tags_dict[book_id2]
                        if tags2:
                            intersection = len(tags1.intersection(tags2))
                            union = len(tags1.union(tags2))
                            tag_similarities[i, j] = intersection / union if union > 0 else 0

            # Select diverse set of books
            selected_indices = [0]  # Start with the highest rated book
            remaining_indices = list(range(1, len(book_ids)))

            while len(selected_indices) < n and remaining_indices:
                # Find the book that is least similar to already selected books
                min_similarity = float('inf')
                best_idx = None

                for idx in remaining_indices:
                    # Calculate average similarity to selected books
                    avg_similarity = np.mean([tag_similarities[idx, sel_idx] for sel_idx in selected_indices])
                    if avg_similarity < min_similarity:
                        min_similarity = avg_similarity
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)

            # Return the selected diverse set of books
            return [five_star_books[i] for i in selected_indices]

        # If we don't have enough 5-star books or no tag data, return top N by rating
        return recommendations[:n]

    @staticmethod
    def evaluate(true_ratings, pred_ratings):
        """Compute RMSE and MAE for predictions."""
        mask = true_ratings.values != 0
        rmse = np.sqrt(mean_squared_error(true_ratings.values[mask], pred_ratings[mask]))
        mae = mean_absolute_error(true_ratings.values[mask], pred_ratings[mask])
        return rmse, mae

    def save_model(self, filepath):
        """Save the trained model to a file."""
        try:
            state_dict = {
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'mean_rating': self.mean_rating,
                'user_index': self.user_index,
                'item_index': self.item_index
            }
            torch.save(state_dict, filepath)
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        try:
            state_dict = torch.load(filepath)
            model = cls()
            model.user_factors = state_dict['user_factors']
            model.item_factors = state_dict['item_factors']
            model.mean_rating = state_dict['mean_rating']
            model.user_index = state_dict['user_index']
            model.item_index = state_dict['item_index']
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

"""
============================================================================
🚀 3. Main Execution
============================================================================
"""

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Book Recommendation System')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--model-name', type=str, default='recommender_model.pt',
                      help='Name of the model file (default: recommender_model.pt)')
    parser.add_argument('--goodreads', type=str, help='Path to Goodreads library export CSV file')
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, args.model_name)
    model = None

    print("\n📚 Book Recommendation System")
    print("=" * 50)

    # Load and prepare data
    try:
        print("\n📥 Loading data...")
        books_df, ratings_df, book_tags_df, tags_df, to_read_df = load_data()
        print("✅ Data loaded successfully")

        # Debug information
        print(f"\n📊 Data Statistics:")
        print(f"  • Number of users: {len(ratings_df['user_id'].unique())}")
        print(f"  • Number of books: {len(books_df)}")
        print(f"  • Number of to_read entries: {len(to_read_df)}")
        print(f"  • Number of users with to_read lists: {len(to_read_df['user_id'].unique())}")

        # Load Goodreads data if provided
        user_ratings_df = None
        user_to_read_df = None
        sample_user = None
        mapping_stats = None

        if args.goodreads:
            print("\n📥 Loading Goodreads data...")
            goodreads_ratings_df, goodreads_to_read_df = load_goodreads_data(args.goodreads)
            if goodreads_ratings_df is not None and goodreads_to_read_df is not None:
                print("✅ Goodreads data loaded successfully")

                # Map Goodreads IDs to dataset IDs
                print("\n🔄 Mapping Goodreads IDs to dataset IDs...")
                mapped_ratings_df, mapped_to_read_df, mapping_stats = map_goodreads_to_dataset_ids(
                    goodreads_ratings_df, goodreads_to_read_df, books_df
                )

                # Store user data separately
                user_ratings_df = mapped_ratings_df
                user_to_read_df = mapped_to_read_df
                sample_user = int(mapped_ratings_df['user_id'].iloc[0])

                print(f"  • Found {mapping_stats['total_ratings']} ratings in your profile")
                print(f"    - {mapping_stats['mapped_ratings']} books matched in our database")
                print(f"  • Found {mapping_stats['total_to_read']} books in your to_read list")
                print(f"    - {mapping_stats['mapped_to_read']} books matched in our database")
            else:
                print("❌ Failed to load Goodreads data")
                return
        else:
            # Find users with both ratings and to_read entries
            users_with_to_read = set(to_read_df['user_id'])
            users_with_ratings = set(ratings_df['user_id'])
            valid_users = list(users_with_to_read.intersection(users_with_ratings))

            if not valid_users:
                print("❌ No users found with both ratings and to_read entries")
                return

            # Select a random user from valid users
            sample_user = int(np.random.choice(valid_users))  # Ensure integer
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Create rating matrices for training
    print("\n🔄 Creating rating matrices...")
    train_matrix, test_matrix = create_rating_matrices(ratings_df)
    print("✅ Matrices created successfully")

    if args.train or not os.path.exists(model_path):
        # Define hyperparameters
        hyperparams = {
            'n_components': [100],
            'learning_rate': [0.003],
            'n_epochs': [60]
        }

        # Train and evaluate models
        print("\n🤖 Training model...")
        for n_components, lr, epochs in product(
            hyperparams['n_components'],
            hyperparams['learning_rate'],
            hyperparams['n_epochs']
        ):
            print(f"\nModel configuration:")
            print(f"  • Components: {n_components}")
            print(f"  • Learning rate: {lr}")
            print(f"  • Epochs: {epochs}")

            # Initialize and train model
            model = Recommender(
                n_components=n_components,
                learning_rate=lr,
                n_epochs=epochs
            )
            model.fit(train_matrix)

            # Evaluate
            predictions = model.predict_all()
            test_aligned = test_matrix.reindex(
                index=model.user_index,
                columns=model.item_index,
                fill_value=0
            )
            rmse, mae = Recommender.evaluate(test_aligned, predictions)

            print(f"\nModel performance:")
            print(f"  • RMSE: {rmse:.3f}")
            print(f"  • MAE: {mae:.3f}")

            # Save the trained model
            print(f"\n💾 Saving model to {model_path}...")
            model.save_model(model_path)
            print("✅ Model saved successfully")
    else:
        # Load existing model
        print(f"\n📂 Loading model from {model_path}...")
        model = Recommender.load_model(model_path)
        if model is None:
            print("❌ Error loading model. Please train a new model with --train")
            return
        print("✅ Model loaded successfully")

    # If using Goodreads data, create a user profile matrix
    if user_ratings_df is not None:
        print("\n🔄 Creating user profile...")
        # Create a user profile matrix with the same structure as train_matrix
        user_matrix = pd.DataFrame(0,
                                 index=[sample_user],
                                 columns=train_matrix.columns)

        # Fill in the user's ratings
        for _, row in user_ratings_df.iterrows():
            if row['book_id'] in user_matrix.columns:
                user_matrix.loc[sample_user, row['book_id']] = row['rating']

        # Get user's to_read list
        user_to_read = set(user_to_read_df['book_id'])

        # Get user's ratings array
        user_ratings = user_matrix.loc[sample_user].values
    else:
        # Get user's to_read list from the dataset
        user_to_read = set(to_read_df[to_read_df['user_id'] == sample_user]['book_id'])
        user_matrix = train_matrix
        user_ratings = None

    # Get user statistics
    num_rated = user_matrix.loc[sample_user].astype(bool).sum()

    print(f"\n📊 Selected user {sample_user}:")
    print(f"  • Number of books rated: {num_rated}")
    print(f"  • Number of books in to_read list: {len(user_to_read)}")

    # Get recommendations for to_read books
    print("\n📚 Top 10 recommended books from your to-read list:")
    print("   (This may take a moment...)")
    recommendations = model.recommend(sample_user, n=10, to_read_only=True,
                                   user_to_read=user_to_read, user_ratings=user_ratings,
                                   book_tags_df=book_tags_df, train_matrix=train_matrix)
    for i, (book_id, pred_rating) in enumerate(recommendations, 1):
        book_info = get_book_details(books_df, book_id)
        if book_info:
            print(f"  {i}. {book_info['title']}")
            print(f"     by {book_info['author']}")
            print(f"     Predicted rating: {pred_rating:.2f}/5.00")
            print()

    # Get recommendations for other books
    print("\n📚 Top 10 recommended books you might like:")
    print("   (This may take a moment...)")
    recommendations = model.recommend(sample_user, n=10, to_read_only=False,
                                   user_to_read=user_to_read, user_ratings=user_ratings,
                                   book_tags_df=book_tags_df, train_matrix=train_matrix)
    for i, (book_id, pred_rating) in enumerate(recommendations, 1):
        book_info = get_book_details(books_df, book_id)
        if book_info:
            print(f"  {i}. {book_info['title']}")
            print(f"     by {book_info['author']}")
            print(f"     Predicted rating: {pred_rating:.2f}/5.00")
            print()

if __name__ == "__main__":
    main()