import csv
import numpy as np
import matplotlib.pyplot as plt

"""
Takes in a string with a path to a CSV file formatted as in the link above, and returns the first 20 data points
(without the Generation and Legendary columns but retaining all other columns) in a single structure.
"""
def load_data(filepath):
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        dict_list = []
        for row in reader:
            row.pop('Generation')
            row.pop('Legendary')
            cols_to_int = ['#', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            for col in cols_to_int:
                row[col] = int(row[col])
            dict_list.append(row)
            if len(dict_list) == 20:
                return dict_list


"""
Takes in one row from the data loaded from the previous function, calculates the corresponding x, y values for that
Pokemon as specified above, and returns them in a single structure.
"""
def calculate_x_y(stats):
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return x, y


"""
Performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y) feature representation,
and returns a data structure representing the clustering.
"""
def hac(dataset):
    for i in range(len(dataset)):  # check for invalid inputs, and pop entry if invalid
        if isinstance(dataset[i], list):
            dataset[i] = tuple(dataset[i])
        for entry in dataset[i]:
            if not np.isfinite(entry):
                dataset.pop(i)
                continue

    length = len(dataset)
    hac_arr = [[None for _ in range(4)] for _ in range(length-1)]
    dist_arr = np.array([[None for _ in range(length)] for _ in range(length)])
    selected = []

    for i in range(length):  # fill out dist_arr
        for j in range(length):
            point1 = dataset[i]
            point2 = dataset[j]
            dist = ((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)**0.5
            dist_arr[i][j] = dist

    def reset():  # helper method to fill in 0s in dist_arr when overwritten
        for cluster in selected:
            for i in range(len(dist_arr)):
                dist_arr[cluster][i] = 0
                dist_arr[i][cluster] = 0

    def find_next():  # helper method to find the next two points/clusters to merge
        l = len(dist_arr)
        min_dist = 9999999
        for a in range(l):
            for b in range(l):
                if dist_arr[a][b] <= min_dist and dist_arr[a][b] != 0:
                    if dist_arr[a][b] == min_dist:
                        continue
                    else:
                        min_dist = dist_arr[a][b]
                        indices = (a, b)
        for a in range(l):
            dist_arr[indices[0]][a] = 0
            dist_arr[a][indices[0]] = 0
            dist_arr[indices[1]][a] = 0
            dist_arr[a][indices[1]] = 0
        selected.append(indices[0])
        selected.append(indices[1])
        return indices

    def dist_between_clusters(a, b):
        if isinstance(a, tuple):
            a = [a]
        if isinstance(b, tuple):
            b = [b]
        dist = 9999999
        for i in range(len(a)):
            for j in range(len(b)):
                point1 = a[i]
                point2 = b[j]
                temp = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
                dist = min(dist, temp)
        return dist

    for entry in range(length-1):
        next_two = find_next()  # find next clusters/points to merge
        new_cluster = [dataset[next_two[0]], dataset[next_two[1]]]
        new_cluster = [z for y in (x if isinstance(x[0], tuple) else [x] for x in new_cluster) for z in y]  # flatten list of tuples
        dataset.append(new_cluster)  # add new cluster to dataset

        new_row = []
        for i in range(len(dist_arr)):  # construct the row/column in dist_arr for the new cluster
            new_row.append(dist_between_clusters(dataset[i], dataset[len(dataset)-1]))
        dist_arr = np.append(dist_arr, [new_row], axis=0)  # add as row to dist_arr
        new_row.append(0)
        new_row = [[item] for item in new_row]
        dist_arr = np.append(dist_arr, new_row, axis=1)  # add as column to dist_arr

        reset()  # recompute 0'd out rows and columns

        dist = dist_between_clusters(dataset[next_two[0]], dataset[next_two[1]])  # compute dist between clusters
        cluster1_size = len(dataset[next_two[0]]) if isinstance(dataset[next_two[0]], list) else 1  # construct cluster 1 size
        cluster2_size = len(dataset[next_two[1]]) if isinstance(dataset[next_two[1]], list) else 1  # construct cluster 2 size
        num_clusters = cluster1_size + cluster2_size  # construct new cluster size
        hac_arr[entry] = [next_two[0], next_two[1], dist, num_clusters]  # add constructed row to hac_arr

    return np.matrix(hac_arr)


"""
Takes in the number of samples we want to randomly generate, and returns these samples in a single structure.
"""
def random_x_y(m):
    rand_list = []
    for _ in range(m):
        x = np.random.randint(1, 360)
        y = np.random.randint(1, 360)
        rand_list.append([x, y])
    return rand_list


"""
Performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y) feature representation,
and imshow the clustering process. Reuses most of the code from hac() and adds plotting functionality.
"""
def imshow_hac(dataset):
    for i in range(len(dataset)):  # check for invalid inputs, and pop entry if invalid
        if isinstance(dataset[i], list):
            dataset[i] = tuple(dataset[i])
        for entry in dataset[i]:
            if not np.isfinite(entry):
                dataset.pop(i)
                continue

    points = list(map(list, zip(*dataset)))
    for x, y in zip(points[0], points[1]):
        plt.scatter(x, y)  # plot given point

    length = len(dataset)
    hac_arr = [[None for _ in range(4)] for _ in range(length - 1)]
    dist_arr = np.array([[None for _ in range(length)] for _ in range(length)])
    selected = []

    for i in range(length):  # fill out dist_arr
        for j in range(length):
            point1 = dataset[i]
            point2 = dataset[j]
            dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
            dist_arr[i][j] = dist

    def reset():  # helper method to fill in 0s in dist_arr when overwritten
        for cluster in selected:
            for i in range(len(dist_arr)):
                dist_arr[cluster][i] = 0
                dist_arr[i][cluster] = 0

    def find_next():  # helper method to find the next two points/clusters to merge
        l = len(dist_arr)
        min_dist = 9999999
        for a in range(l):
            for b in range(l):
                if dist_arr[a][b] <= min_dist and dist_arr[a][b] != 0:
                    if dist_arr[a][b] == min_dist:
                        continue
                    else:
                        min_dist = dist_arr[a][b]
                        indices = (a, b)
        for a in range(l):
            dist_arr[indices[0]][a] = 0
            dist_arr[a][indices[0]] = 0
            dist_arr[indices[1]][a] = 0
            dist_arr[a][indices[1]] = 0
        selected.append(indices[0])
        selected.append(indices[1])
        return indices

    def dist_between_clusters(a, b):
        if isinstance(a, tuple):
            a = [a]
        if isinstance(b, tuple):
            b = [b]
        dist = 9999999
        for i in range(len(a)):
            for j in range(len(b)):
                point1 = a[i]
                point2 = b[j]
                temp = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
                # dist = min(dist, temp)
                if temp < dist:
                    dist = temp
                    closest_points = (point1, point2)
        return dist, closest_points

    for entry in range(length - 1):
        next_two = find_next()  # find next clusters/points to merge
        new_cluster = [dataset[next_two[0]], dataset[next_two[1]]]
        new_cluster = [z for y in (x if isinstance(x[0], tuple) else [x] for x in new_cluster) for z in y]  # flatten list of tuples
        dataset.append(new_cluster)  # add new cluster to dataset

        new_row = []
        for i in range(len(dist_arr)):  # construct the row/column in dist_arr for the new cluster
            new_row.append(dist_between_clusters(dataset[i], dataset[len(dataset) - 1])[0])
        dist_arr = np.append(dist_arr, [new_row], axis=0)  # add as row to dist_arr
        new_row.append(0)
        new_row = [[item] for item in new_row]
        dist_arr = np.append(dist_arr, new_row, axis=1)  # add as column to dist_arr

        reset()  # recompute 0'd out rows and columns

        dist, closest_points = dist_between_clusters(dataset[next_two[0]], dataset[next_two[1]])  # compute dist and closest points between clusters
        cluster1_size = len(dataset[next_two[0]]) if isinstance(dataset[next_two[0]], list) else 1  # construct cluster 1 size
        cluster2_size = len(dataset[next_two[1]]) if isinstance(dataset[next_two[1]], list) else 1  # construct cluster 2 size
        num_clusters = cluster1_size + cluster2_size  # construct new cluster size
        hac_arr[entry] = [next_two[0], next_two[1], dist, num_clusters]  # add constructed row to hac_arr

        line_segment = list(map(list, zip(*closest_points)))  # form coordinates for next line segment
        plt.plot(line_segment[0], line_segment[1])  # plot given line segment
        plt.pause(0.1)  # wait 0.1 seconds

    plt.show()  # finally show the plot
