import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
movies_data = pd.read_csv('movies.csv')
print("==================MOVIES DATASET==========================")
movies_data=pd.DataFrame(movies_data) 
print(movies_data)
movies_data['year'] = movies_data.title.str.extract('(\(\d\d\d\d\))',expand=False) #Finding a year stored between parentheses
movies_data['year'] = movies_data.year.str.extract('(\d\d\d\d)',expand=False) #Removing the parentheses from the year
movies_data['title'] = movies_data.title.str.replace('(\(\d\d\d\d\))', '') #Removing the years from the 'title' column
movies_data['title'] = movies_data['title'].apply(lambda x: x.strip()) #Ending whitespace characters
movies_data['genres'] = movies_data.genres.str.split('|')  #Removing | from genres

# We dont need the genre information in our first case so coping into a new dataframe.
movies_Genres = movies_data.copy()
#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_data.iterrows():
    for genre in row['genres']:
        movies_Genres.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
movies_Genres = movies_Genres.fillna(0)
print("=====================GENRES==========================")
print(movies_Genres.head())

user_input= [{'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]

input_movies = pd.DataFrame(user_input)
print("==================User Input Movies==========================")
print(input_movies.head())

#Filtering out the movies by title
inputId = movies_data[movies_data['title'].isin(input_movies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
input_movies = pd.merge(inputId, input_movies)
#Dropping information we won't use from the input dataframe
input_movies = input_movies.drop('genres', 1).drop('year', 1)
#Filtering out the movies from the input
userMovies = movies_Genres[movies_Genres['movieId'].isin(input_movies['movieId'].tolist())]
#print(userMovies)
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print("==================User Genre Table==========================")
print(userGenreTable)
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(input_movies['rating'])
#The user profile
print("==================USER PROFILE==========================")
print(userProfile)
#Now let's get the genres of every movie in our original dataframe
genreTable = movies_Genres.set_index(movies_Genres['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
#print(genreTable)#.head()
genreTable.shape

#Multiply the genres by the weights and then take the weighted average
#print("CALCULATIONS")
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
#print(recommendationTable_df)#.head()

#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
#print(recommendationTable_df)#.head()

ratings_df=pd.read_csv('ratings.csv',chunksize=100000)
chunk_list=[]
for chunk in ratings_df:
    chunk.drop('timestamp',axis=1,inplace=True)
    chunk_list.append(chunk)
    
ratings_df = pd.concat(chunk_list)
ratings_df=ratings_df.head(1000000)
print("==================Ratings DATASET==========================")
print(ratings_df.head())
#print(ratings_df)

movies_data = movies_data.drop('genres', 1)
#print(movies_data.head())#.head()

userSubset = ratings_df[ratings_df['movieId'].isin(input_movies['movieId'].tolist())]
userSubsetGroup = userSubset.groupby(['userId'])
#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup[0:3]
userSubsetGroup = userSubsetGroup[0:100]

#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    input_movies = input_movies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = input_movies[input_movies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
pearsonCorrelationDict.items()
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
#print(pearsonDF)#.head()
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

##print(topUsers.head(10))#.head()
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
##print("users rating")
##print(topUsersRating)
#print(topUsersRating)#.head()
#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
print("==================TOP USERS RATING==========================")
print(topUsersRating.head())
#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
#print(tempTopUsersRating)#.head()
#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
#print(recommendation_df)#.head()
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
#print(recommendation_df)#.head(10)
a2=movies_data.loc[movies_data['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
#a2 = a2[["title"]].head(1000)
##print("A2")
##print(a2)
print("==================RECOMMENDED MOVIES=========================")
print(a2.loc[:,['title']].head(10))
