# #This code was written without consulting code written by anyone else. 
# # I used the documentation provided by the sklearn and the matplotlib library
# # - Isac Simkin
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import sys
import copy
import random

outFile = open(sys.argv[3], "a+") #output file

with open(sys.argv[1]) as f:
    first_line = f.readline()

splitArr = first_line.split(",")
if (len(splitArr) > 1): # if the separator is a comma
    data = pd.read_csv(sys.argv[1], sep = ",") 
    dataLength = len(data.columns) # number of columns 
    names = []
    for i in range (dataLength):
        names.append("Attribute " + str(i))
    data = pd.read_csv(sys.argv[1], sep = ",", names = names) #assign names to the columns 
    data = data.select_dtypes(include = 'number') #extracts only the columns that have float64 data types 
else: #separator is white space
    data = pd.read_csv(sys.argv[1], delim_whitespace = True) 
    dataLength = len(data.columns) # number of columns 
    names = []
    for i in range (dataLength):
        names.append("Attribute " + str(i))
    data = pd.read_csv(sys.argv[1], delim_whitespace = True, names = names) #assign names to the columns 
    data = data.select_dtypes(include = 'number') #extracts only the columns that have float64 data types 


#Scale it to a 0 - 1 range 
x = data.values
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)

#Chooses 2 columns at random
namesArr = data.columns
choice1 = random.choice(namesArr)
choice2 = random.choice(namesArr)
if (choice1 == choice2): #if the columns happen to be the same, make another choice
	choice2 = random.choice(namesArr)

#obtains the max value of the data set and the input value for 'k'
m = data.values[:, 0 : dataLength].max()
k = int(float(sys.argv[2]))

# obtains the silhouette score. Sklearn ONLY used for the silhouette score 
kmeans_model = KMeans(n_clusters=k, random_state=1).fit(data)
labels = kmeans_model.labels_
silhouetteScore = metrics.silhouette_score(data, labels, metric='euclidean')

#defines the initial position of the centroids at random within 0 and the maximum value of the entire dataset
for i in range(k):
    centroids = {
        i+1: [np.random.randint(0, m), np.random.randint(0, m)]       
    }

# plot = plt.figure()
# plt.scatter(data[choice1], data[choice2], color = 'k')

#colors to be used for the centroids and then for assigning to each data point.
colors = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm', 6: 'y'}

# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colors[i])
# plt.xlim(0, m)
# plt.ylim(0, m)
# plt.xlabel(choice1)
# plt.ylabel(choice2)

def assign(data, centroids):
    for i in centroids.keys():
        data['distance_from_{}'.format(i)] = (np.sqrt((data[choice1] - centroids[i][0]) ** 2+ (data[choice2] - centroids[i][1]) ** 2))
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    data['closest'] = data.loc[:, centroid_distance_cols].idxmin(axis=1)
    data['minDistance'] = data.loc[:, centroid_distance_cols].min(axis = 1) # distance to closest centroid
    data['closest'] = data['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    data['color'] = data['closest'].map(lambda x: colors[x])
    return data

data = assign(data, centroids)

# plot2 = plt.figure()
# plt.scatter(data[choice1], data[choice2], color=data['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colors[i])
# plt.xlim(0, m)
# plt.ylim(0, m)
# plt.xlabel(choice1)
# plt.ylabel(choice2)
# plt.show()

old_centroids = copy.deepcopy(centroids)

#updates the position of the centroids
def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(data[data['closest'] == i][choice1])
        centroids[i][1] = np.mean(data[data['closest'] == i][choice2])
    return k

centroids = update(centroids)
    
# plot3 = plt.figure()
# ax = plt.axes()
# plt.scatter(data[choice1], data[choice2], color=data['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colors[i])
# plt.xlim(0, m)
# plt.ylim(0, m)
# plt.xlabel(choice1)
# plt.ylabel(choice2)
# for i in old_centroids.keys():
#     old_x = old_centroids[i][0]
#     old_y = old_centroids[i][1]
#     dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
#     dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
#     ax.arrow(old_x, old_y, dx, dy, head_width=.2, head_length=.3, fc=colors[i], ec=colors[i])


# updates the centroids until they are as close as they can be to the median
while True:
	closest_centroids = data['closest'].copy(deep=True)
	centroids = update(centroids)
	data = assign(data, centroids)
	if closest_centroids.equals(data['closest']):
	    break

#plot for the final result 
# plot4 = plt.figure()
# plt.scatter(data[choice1], data[choice2], color=data['color'], alpha=0.5, edgecolor='k')
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colors[i])

for i in data['color']:
	for v,k in colors.items():
		if (i == k):
			outFile.write(str(v) + "- " + k + "\n")

# plt.xlim(0, m)
# plt.ylim(0, m)
# plt.xlabel(choice1)
# plt.ylabel(choice2)

#obtains the SSE value from the min distances obtained in the assign function
def sse(column):
    data[column] = np.square(data[column])
    sse = sum(data[column]) #SSE Value 
    return sse

outFile.write("SSE = " + str(sse('minDistance')) + "\n")
outFile.write("Silhouette Coefficient = " + str(silhouetteScore))
outFile.close()
#plt.show() #uncomment to check out the progression of the k-means algorithm in plots 
