import pandas as pd #data manipulation and analysis
#load files, merge datasets, handle missing values, group data
import re #text processing using regular expressions
#extract release years from titles and clean titles
from sklearn.preprocessing import MultiLabelBinarizer #converts multi-label categorical data into binary columns
#split "Action | Comedy" into lists, create one-hot encoded columns

#1. Data Loading & Initial Cleaning

#Load movie datasets
movies = pd.read_csv("movies.csv") 
#moviesID, title, genres

ratings = pd.read_csv("ratings.csv")
#userId,movieId,rating,timestamp

tags = pd.read_csv("tags.csv")
#userId,movieId,tag,timestamp

links = pd.read_csv("links.csv")
#movieId,imdbId,tmdbId

tags.dropna(subset=["tag"], inplace=True)  # Remove rows where 'tag' is missing
links.dropna(subset=["tmdbId"], inplace=True) #Remove rows where 'tmdbId' is missing

#2. Data Type Conversion
ratings["userId"] = ratings["userId"].astype("int32")
ratings["movieId"] = ratings["movieId"].astype("int32")
ratings["rating"] = ratings["rating"].astype("float32")
movies["movieId"] = movies["movieId"].astype("int32")

#tags["tag"].fillna("", inplace=True) | Fill missing tags with an empty string
#links["tmdbId"].fillna(0, inplace=True) |  Assign 0 to missing tmdbId values

#3. Title Processing
#function to extract years from the titles
def extract_year(title):
  match_year = re.search(r"\((\d{4})\)",title) #(\d{4})-> capture exactly 4 digits as a group "1995"
  return int(match_year.group(1)) if match_year else None
  #.group(1) return only the captured part 1995

movies["year"] = movies["title"].apply(extract_year)

#remove years from titles
movies["title"] = movies["title"].apply(lambda x: re.sub(r"\(\d{4}\)","", x).strip())
#.strip() remove the blank space after removing year

#4. Rating Normalization(USING MEAN_CENTERING METHOD)
#calculate mean rating for each user
user_avg_ratings = ratings.groupby("userId")["rating"].transform("mean")
user_avg_ratings
#normalize ratings = rating - user's avg rating
ratings["normalized_rating"] = ratings["rating"] - user_avg_ratings

#5. Implicit Ratings (Like/Dislike)
#create implicit ratings(like/dislike)
ratings["implicit_ratings"]= ratings["rating"].apply(lambda x: 1 if x >=3.5 else 0)

# Compute average rating per movie
movie_avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()

# Rename column
movie_avg_ratings.rename(columns={"rating": "avg_movie_rating"}, inplace=True)

# Merge movies and ratings
df = ratings.merge(movies[['movieId', 'title','genres']], on="movieId", how="left")

#6. Genre One-Hot Encoding
#split genres into lists
df["genres"] = df["genres"].fillna("Unknown").apply(lambda x: x.split("|"))#split the genre string into a list of genres

if "genres" in df.columns:
#apply one-hot encoding
  mlb = MultiLabelBinarizer()
  genres_encoded = pd.DataFrame(mlb.fit_transform(df["genres"]), columns=mlb.classes_)
  #fit_transform()-> mlb examines all genre lists to identify every unique genre and create an internal maping of genre-to-column-index, store these in mlb.classes_
  #Concatenate with movies DataFrame
  df = pd.concat([df.drop("genres", axis=1), genres_encoded], axis=1)
  
#7. Tag Processing
#Remove rows where tags is missing
tags.dropna(subset=["tag"],inplace = True)

#Convert tags to lowercase
tags["tag"]=tags["tag"].str.lower()

#Group tags by movieID
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
#.apply(lambda x: " ".join(x)) takes all associated tags and joins them into a single string seperated by spaces

# Merge tags with your main dataset (df)
df = df.merge(tags_grouped, on="movieId", how="left")

# Fill missing tags with an empty string 
df["tag"] = df["tag"].fillna("")

#8. Final DataFrame Assembly
columns_order = ["userId", "movieId", "title", "rating", "timestamp", "normalized_rating", "implicit_ratings","tag"] + list(genres_encoded.columns)
#Pandas DataFrame.columns returns an Index object, but concatenation (+) requires a list.
df = df[columns_order]
#pd.set_option('display.max_columns', None)->display all column of dataset
df.to_csv('cleaned_data.csv', index=False)
print(df.head())





