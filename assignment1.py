#!/usr/bin/env python
# coding: utf-8

# In[268]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model


# In[269]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[270]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[271]:


# Some data structures that will be useful


# In[272]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)


# In[273]:


##################################################
# Play prediction                                #
##################################################


# In[274]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]


# In[275]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
games = []
for d in hoursTrain:
    if d[1] not in games:
        games.append(d[1])
    u,i = d[0],d[1]
    reviewsPerUser[u].append(i)
    reviewsPerItem[i].append(u)


# In[276]:


# determine a random game for a user that they haven't played
def randomgame(user):
    gamesPlayed = reviewsPerUser[user]
    gamesNotPlayed = [element for element in games if element not in gamesPlayed]
    
    random_game = random.sample(gamesNotPlayed, 1)
    return random_game[0]


# In[277]:


# Augmented validation set should be our "ground truth label"
augmentedValidationSet = []
for d in hoursValid:
    augmentedValidationSet.append([d[0],d[1],'1']) # Played game
    augmentedValidationSet.append([d[0],randomgame(d[0]),'0']) # Have not played game


# In[280]:


# Improved strategy
return2 = set()
count2 = 0
threshold = 1.7
for ic, i in mostPopular:
    count2 += ic
    return2.add(i)
    if count2 > totalPlayed/threshold: break


# In[283]:


gamesPerUser = defaultdict(list) # shows all the games played by each individual user
usersPerGame = defaultdict(list) # shows all the users that have played an individual game
for d in hoursTrain:
    u,i = d[0],d[1]
    gamesPerUser[u].append(i)
    usersPerGame[i].append(u)


# In[284]:


def Jaccard(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[287]:


def CosineSet(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    numer = len(s1.intersection(s2))
    denom = math.sqrt(len(s1)) * math.sqrt(len(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[288]:


predictions4 = []
for d in augmentedValidationSet:
    user = d[0]
    game = d[1]
    GPU = gamesPerUser[user]
    UPG = usersPerGame[game] # this is u_g
    jaccardMax = 0
    cosMax = 0
    for g in GPU: # for each game in g'
        game_prime = usersPerGame[g] # this is u_g'
        jaccard = Jaccard(UPG, game_prime)
        cosine = CosineSet(UPG, game_prime)
        if jaccard > jaccardMax:
            jaccardMax = jaccard
        if cosine > cosMax:
            cosMax = cosine
    # check similarity functions
    if cosMax > 0.0665 and jaccardMax > 0.052 or len(usersPerGame[g]) > 60:
        predictions4.append([d[0], d[1], '1'])
    else:
        predictions4.append([d[0], d[1], '0'])


# In[289]:


correct4 = 0
for pred, actual in zip(predictions4, augmentedValidationSet):
    if pred == actual:
        correct4 += 1


# In[290]:


accuracy4 = correct4 / len(predictions4)
accuracy4


# In[330]:


def ties(user_list):
    num = 0
    zeros = []
    ones = []
    
    global userGamePred
    user = userGamePred[user_list]
    for game in user:
        if game[1] == 0:
            zeros.append(game)
        else:
            ones.append(game)
    
    user_length = len(user)
    zero_length = len(zeros)
    one_length = len(ones)
    if one_length > zero_length:
        sorted_ones = sorted(ones, key=lambda x: x[2])
        num_updates = int(one_length - user_length/2)
        for i in range(num_updates):
            updated_game = sorted_ones[i][0]
            game_index = 0
            for game in user:
                if game[0] == updated_game:
                    new_pred = (game[0], 0, game[2])
                    userGamePred[user_list][game_index] = new_pred
                    break
                game_index += 1
    elif zero_length > one_length:
        sorted_zeros = sorted(zeros, key=lambda x: x[2], reverse=True)
        num_updates = int(zero_length - user_length/2)
        for i in range(num_updates):
            updated_game = sorted_zeros[i][0]
            game_index = 0
            for game in user:
                if game[0] == updated_game:
                    new_pred = (game[0], 1, game[2])
                    userGamePred[user_list][game_index] = new_pred
                    break
                game_index += 1


# In[332]:


userGamePred = defaultdict(list)
predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')

    GPU = gamesPerUser[u]
    UPG = usersPerGame[g] # this is u_g
    jaccardMax = 0
    cosMax = 0
    for game in GPU: # for each game in g'
        game_prime = usersPerGame[game] # this is u_g'
        jaccard = Jaccard(UPG, game_prime)
        cosine = CosineSet(UPG, game_prime)
        if jaccard > jaccardMax:
            jaccardMax = jaccard
        if cosine > cosMax:
            cosMax = cosine
            
    pred = None
    # check similarity functions
    if cosMax > 0.0665 and jaccardMax > 0.052 or len(usersPerGame[g]) > 60:
        pred = 1
    else:
        pred = 0
    userGamePred[u].append((g, pred, ((cosMax+jaccardMax))*len(usersPerGame[g])))

# check if number of 1's and 0's per user are equal, if not update
for user_list in userGamePred:
    ties(user_list)

# write out to .csv value for submission
for user_list in userGamePred:
    for game in userGamePred[user_list]:
        _ = predictions.write(user_list + ',' + game[0] + ',' + str(game[1])  + '\n')
predictions.close()


# In[293]:


##################################################
# Hours played prediction                        #
##################################################


# In[309]:


trainHours = [r[2]['hours_transformed'] for r in allHours]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[333]:


hoursPerUser = {}
hoursPerItem = {}
trainData = {}

# train on entire dataset
for user, item, features in allHours:
    # Update trainData
    trainData[(user, item)] = features['hours_transformed']

    # Update hoursPerUser
    if user not in hoursPerUser:
        hoursPerUser[user] = {item: features['hours_transformed']}
    else:
        hoursPerUser[user][item] = features['hours_transformed']

    # Update hoursPerItem
    if item not in hoursPerItem:
        hoursPerItem[item] = {user: features['hours_transformed']}
    else:
        hoursPerItem[item][user] = features['hours_transformed']

betaU = {}
betaI = {}

# hoursPerUser maps each user to a list of games and their associated playtimes
for u in hoursPerUser:
    betaU[u] = 0

# hoursPerItem maps each game to a list of users and the corresponding playtimes they spent on that game.
for g in hoursPerItem:
    betaI[g] = 0

alpha = globalAverage 


# In[334]:


def iterate2(lamb):
    global alpha
        
    # calculate betaI
    for item in betaI:
        betaI_sum = 0
        for user in hoursPerItem[item]:
            betaI_sum += (trainData[(user, item)] - (alpha + betaU[user]))
        betaI[item] = betaI_sum / (lamb + len(hoursPerItem[item]))
        
    # calculate betaU
    for user in betaU:
        betaU_sum = 0
        for item in hoursPerUser[user]:
            betaU_sum += (trainData[(user, item)] - (alpha + betaI[item]))
        betaU[user] = betaU_sum / (lamb + len(hoursPerUser[user]))  
    
    # calculate alpha
    alpha_sum = 0
    for user, item in trainData:
        alpha_sum += (trainData[(user, item)] - (betaU[user] + betaI[item]))
    alpha = alpha_sum / len(trainData)


# In[335]:


# running iterate for 3 iterations and lambda = 4.4
for x in range(3):
    iterate2(4.4)


# In[336]:


# Calculate MSE on the validation set
valid_data2 = [(user, game, features['hours_transformed']) for user, game, features in hoursValid]
predictions2 = [alpha + betaU[user] + betaI[item] for user, item, _ in valid_data2]
actual_values2 = [hours_transformed for _, _, hours_transformed in valid_data2]

validMSE2 = numpy.mean((numpy.array(predictions2) - numpy.array(actual_values2))**2)
print(f"Mean Squared Error on Validation Set: {validMSE2}")


# In[337]:


# Write the prediction hour values to .csv for submission
predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    alpha = alpha
    bu = betaU[u]
    bi = betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()


# In[ ]:




