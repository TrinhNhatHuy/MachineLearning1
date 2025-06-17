import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the cleaned data
df = pd.read_csv('cleaned_data.csv')

# Only use a subset of the movies (e.g., 5000 movies)
df_subset = df.sample(n=5000, random_state=42)

# Extract the genres columns (one-hot encoded)
genre_columns = [col for col in df_subset.columns if col not in ['userId', 'movieId', 'title', 'rating', 'timestamp', 'normalized_rating', 'implicit_ratings', 'tag']]

# Extract the genre features
genre_vectors = df_subset[genre_columns]

# Convert genre vectors into a sparse matrix
genre_vectors_sparse = csr_matrix(genre_vectors)

# Initialize the Nearest Neighbors model for approximate search
knn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine', n_jobs=-1)

# Fit the model with the genre vectors
knn.fit(genre_vectors_sparse)

# Example movie ID to query
movie_id = 474

# Print first few rows to see movieId and titles
print(df_subset[['movieId', 'title']].head())

# Check if movie_id exists in the dataset
if movie_id not in df_subset['movieId'].values:
    print(f"Movie ID {movie_id} not found in the dataset. Selecting a random movie.")
    movie_id = df_subset['movieId'].sample(n=1).values[0]

# Get the index of the movie in the subset
movie_index = df_subset[df_subset['movieId'] == movie_id].index[0]

# Query the model for similar movies
distances, indices = knn.kneighbors(genre_vectors_sparse[movie_index], n_neighbors=10)

# Retrieve the recommended movies based on indices
# Ensure indices correspond to the correct movies in the subset
recommended_movies = df_subset.iloc[indices[0]].copy()

# Add the distance values to the recommended movies dataframe
recommended_movies['distance'] = distances[0]

# Display the recommended movies with the title, movieId, and distance
print(recommended_movies[['title', 'movieId', 'distance']])
