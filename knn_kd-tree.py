"""
Student: Vishnu Rampersaud 
Course: CSCI 49900 
Instructor: Dey 

This is an implementation of knn using a kd-tree. 

**Note: Before using the code, please make sure your data is going to be accurately parsed
    - in the load_file() function, you can omit rows & columns as necessary 
    - set the class_col global variable so the program knows the column to omit when manipulating data.
      This column should only be set to either -1 for the last column, or 0 for the first column.
    - in the find_xmin_xmax(), normalize(), __euclidean_distance() functions omit the classification
      column from being tampered with. Setting the class_col to either -1 or 0 solves this. 
    - set num_of_attributes in main function to the number of attributes in the data - minus the class column
    - There is also a weighted and unweighted version of knn. By default, this program uses the weighted knn. 
      To change to unweighted knn: 
        - in the find_best_k() function, find the line that says "prediction = weighted_classification(arguments)"
        - change weighted_classification() to unweighted_classification(); keep the same parameters
        - do the same for the run_knn_for_k() function
"""

from operator import itemgetter
from math import sqrt

# the classification column can only be either, '-1' which is the last column, or '0' which is the first column
class_col = -1

# Node class that holds the value of the node, and the lest and right children
class Node(): 

    def __init__(self, root): 
        self.left = None
        self.right = None
        self.value = root

# Kd-tree class; Use to construct a kd-tree 
class kd_tree():

    # private variables
    # @num_of_attributes: takes in the number of attributes/ dimensions the tree will have 
    def __init__ (self, num_of_attributes):
        self.root = None
        self.num_of_attributes = num_of_attributes
        pass

    # insert node into the tree based on the specified attribute 
    def insert (self, new_node): 
        
        if self.root == None: 
            self.root = Node(new_node)
        else: 
            self.__insert(new_node, self.root, attribute=-1)

    # internal insert function
    def __insert (self, new_node, root, attribute): 

        attribute += 1
        if (self.num_of_attributes == attribute): 
            attribute = 0

        if new_node[attribute] >= root.value[attribute]: 
            if root.right == None:
                root.right = Node(new_node)
            else:
                self.__insert(new_node, root.right, attribute)
        elif new_node[attribute] < root.value[attribute]:
            if root.left == None: 
                root.left = Node(new_node)
            else: 
                self.__insert(new_node, root.left, attribute)

        pass

    # checks to see if node is present in the tree 
    def contains(self, query_node): 

        if (self.root != None): 
            return self.__contains(query_node, self.root, attribute=-1)

    # internal contains function
    def __contains(self, query_node, root, attribute): 

        attribute += 1
        if (self.num_of_attributes == attribute): 
            attribute = 0

        if query_node == root.value: 
            return True
        elif query_node[attribute] >= root.value[attribute]: 
            if root.right != None: 
                return self.__contains(query_node, root.right, attribute)
            else: 
                return False

        elif query_node[attribute] < root.value[attribute]: 
            if root.left != None: 
                return self.__contains(query_node, root.left, attribute)
            else: 
                return False

    # computes the nearest neighbors of the specified query point
    # return the best distance, and also populates the inputted list with all neighbor distances
    def nearest(self, query_node, neighbor_distances): 

        if (self.root != None): 
            return self.__nearest(query_node, self.root, neighbor_distances, best_dist=999, attribute=-1)

    # internal nearest neighbor function
    def __nearest(self, query_node, root, neighbor_distances, best_dist, attribute): 

        # compare nodes based on this attribute 
        attribute += 1
        if (self.num_of_attributes == attribute): 
            attribute = 0

        # if the query node's attribute is greater than the current node's, then traverse to the right side / good side 
        if query_node[attribute] >= root.value[attribute]: 
            dist = self.__euclidean_distance(query_node, root.value)
            neighbor_distances.append([dist, root.value])
            if (dist < best_dist): 
                best_dist = dist

            if root.right != None: 
                best_dist =  self.__nearest(query_node, root.right, neighbor_distances, best_dist, attribute)
                # check to see whether it is worth traversing the other child of this node / bad side 
                # if it is, then traverse the other child 
                if ( abs( query_node[attribute] - root.value[attribute] ) < best_dist ): 
                    if root.left != None: 
                        best_dist =  self.__nearest(query_node, root.left, neighbor_distances, best_dist, attribute)
            
            # check to see whether it is worth traversing the other child of this node / bad side
            # if it is, then traverse the other child 
            elif root.right == None: 
                if ( abs( query_node[attribute] - root.value[attribute] ) < best_dist ): 
                    if root.left != None: 
                        best_dist =  self.__nearest(query_node, root.left, neighbor_distances, best_dist, attribute)

            return best_dist

        # if the query node's attribute is less than the current node's, then traverse to the left side / good side 
        elif query_node[attribute] < root.value[attribute]: 
            dist = self.__euclidean_distance(query_node, root.value)
            neighbor_distances.append([dist, root.value])
            if (dist < best_dist): 
                best_dist = dist

            if root.left != None: 
                best_dist = self.__nearest(query_node, root.left, neighbor_distances, best_dist, attribute)
                # check to see whether it is worth traversing the other child of this node / bad side
                # if it is, then traverse the other child 
                if ( abs( query_node[attribute] - root.value[attribute] ) < best_dist ):
                    if root.right != None:
                        best_dist =  self.__nearest(query_node, root.right, neighbor_distances, best_dist, attribute)

            # check to see whether it is worth traversing the other child of this node 
            # if it is, then traverse the other child 
            elif root.left == None: 
                if ( abs( query_node[attribute] - root.value[attribute] ) < best_dist ): 
                    if root.right != None: 
                        best_dist =  self.__nearest(query_node, root.right, neighbor_distances, best_dist, attribute)

            return best_dist

    # computes euclidean distance 
    def __euclidean_distance(self, query_node, training_row): 

        distance = 0.0

        if class_col < 0:
            skip_class_col = abs(class_col)
            for i in range(len(query_node)-skip_class_col): 
                distance += ((query_node[i] - training_row[i])**2)

        else: 
            skip_class_col = class_col + 1
            for i in range(skip_class_col, len(query_node)): 
                distance += ((query_node[i] - training_row[i])**2)
    
        eucl_distance = sqrt(distance)
        return eucl_distance 

    # prints the tree 
    def printTree(self): 

        if self.root != None: 
            self.__printTree(self.root)

    # internal printTree function
    def __printTree(self, root): 

        if root != None: 
            self.__printTree(root.left)
            print(root.value)
            self.__printTree(root.right)


# load file into a list data structure
def load_file (filename, dataset): 

    # open file and store into data_file object 
    data_file = open(filename, 'r')

    # separate each row by commas and store each row into a list, and then all rows into a list of lists
    # remove unwanted data and only keep attributes and classification that will be used
    for line in data_file: 
        if not line: 
            continue
        data_row = line.split(",")
        data_row = data_row[1:-1]

        for i in range(len(data_row)): 
            data_row[i] = int(data_row[i])

        dataset.append(data_row)
   
    pass 

# find the minimum and maximum attribute in the dataset 
def find_xmin_xmax(training_dataset): 

    xmin = 999
    xmax = 0

    if class_col < 0:
        skip_class_col = abs(class_col)
        for train_row in training_dataset: 
        # omit the classification column
            for i in range(len(train_row)-skip_class_col): 
                if train_row[i] < xmin: 
                    xmin = train_row[i]
                elif train_row[i] > xmax: 
                    xmax = train_row[i]
    else: 
        skip_class_col = class_col + 1
        for train_row in training_dataset: 
            # omit the classification column
            for i in range(skip_class_col, len(train_row)): 
                if train_row[i] < xmin: 
                    xmin = train_row[i]
                elif train_row[i] > xmax: 
                    xmax = train_row[i]

    return [xmin, xmax]


# normalize the data so that the attributes are measured evenly
def normalize(dataset, x_min_max):
    if class_col < 0:
        skip_class_col = abs(class_col)
        for i in range(len(dataset)):
            # omit the classification column 
            for j in range(len(dataset[i])-skip_class_col):  
                dataset[i][j] = (dataset[i][j] - x_min_max[0]) / (x_min_max[1] - x_min_max[0])
    else: 
        skip_class_col = class_col + 1
        for i in range(len(dataset)):
            # omit the classification column 
            for j in range(skip_class_col, len(dataset[i])):  
                dataset[i][j] = (dataset[i][j] - x_min_max[0]) / (x_min_max[1] - x_min_max[0])

    return dataset

# Sort the data by the specified attributes 
def sort_data_by_attribute(training_dataset, attribute_col, left, right):

    # sort data from left index to right index 
    sorted_data = sorted(training_dataset[left:right+1], key = itemgetter(attribute_col) )

    # place the sorted data into the correct indices of the original list 
    count = 0
    for i in range (left, right+1): 
        training_dataset[i] = sorted_data[count]
        count += 1

    return training_dataset

# find the median of the specified datset 
def find_median(training_dataset, left, right): 

    med_index = int( ((right-left)+1) / 2 )

    return med_index

# Fill the training dataset into the tree 
def fill_tree (training_dataset, num_of_attributes, attribute, left, right, tree): 

    if (left <= right ):
        if (num_of_attributes == attribute): 
            attribute = 0
        sort_data_by_attribute(training_dataset, attribute, left, right)
        med_index = left + find_median(training_dataset, left, right)
        tree.insert(training_dataset[med_index]) 
        fill_tree(training_dataset, num_of_attributes, attribute+1, left, med_index-1, tree)
        fill_tree(training_dataset, num_of_attributes, attribute+1, med_index+1, right, tree)

# Weighted knn prediction 
def weighted_classification(neighbor_distances, majority_class, k): 

    # find all present classificatons in the data and store them into a list 
    classifications = []
    for i in range(k): 
        # stop looking for classifications if reach end of list, and the number of neighbors is less than k 
        if (i >= len(neighbor_distances)): 
            break
        if neighbor_distances[i][1][class_col] not in classifications: 
            classifications.append(neighbor_distances[i][1][class_col])


    # sums up the weighted value for each classification in k-nearest neighbors
    total_classification_count = []
    for classes in classifications: 
        count = 0
        for i in range(k): 
            # stop looking for classifications if reach end of list, and the number of neighbors is less than k 
            if (i >= len(neighbor_distances)): 
                break
            if (classes == neighbor_distances[i][1][class_col]): 
                count += 1/(neighbor_distances[i][0]+1)
        total_classification_count.append([count, classes])

    # Identifies the classificaton with the highest weighted sum in k-nearest neighbors
    # if there is a tie, then the majority class is selected as the tie breaker
    max = 0
    max_class = majority_class
    for i in range(len(total_classification_count)): 
        if (total_classification_count[i][0] > max): 
            max = total_classification_count[i][0]
            max_class = total_classification_count[i][1]
        elif (total_classification_count[i][0] == max): # and max != 0): 
            max_class = majority_class

    # return the class with the highest weighted sum / majority class
    return max_class

# Unweighted knn prediction 
def unweighted_classification(neighbor_distances, majority_class, k): 

    # find all present classificatons in the data and store them into a list 
    classifications = []
    for i in range(k):
        # stop looking for classifications if reach end of list, and the number of neighbors is less than k 
        if (i >= len(neighbor_distances)): 
            break
        if neighbor_distances[i][1][class_col] not in classifications: 
            classifications.append(neighbor_distances[i][1][class_col])

    # sums up the unweighted value for each classification in k-nearest neighbors
    total_classification_count = []
    for classes in classifications: 
        count = 0
        for i in range(k): 
            # stop looking for classifications if reach end of list, and the number of neighbors is less than k 
            if (i >= len(neighbor_distances)): 
                break
            if (classes == neighbor_distances[i][1][class_col]): 
                count += 1
        total_classification_count.append([count, classes])

    # Identifies the classificaton with the highest weighted sum in k-nearest neighbors
    # if there is a tie, then the majority class is selected as the tie breaker
    max = 0
    max_class = majority_class
    for i in range(len(total_classification_count)): 
        if (total_classification_count[i][0] > max): 
            max = total_classification_count[i][0]
            max_class = total_classification_count[i][1]
        elif (total_classification_count[i][0] == max): # and max != 0): 
            max_class = majority_class

    # return the class with the highest weighted sum / majority class
    return max_class

# Identifies the majority class in the data 
def find_majority_class(neighbor_distances): 

    # find all present classificatons in the data and store them into a list 
    classifications = []
    for i in range(len(neighbor_distances)):
        if neighbor_distances[i][1][class_col] not in classifications: 
            classifications.append(neighbor_distances[i][1][class_col])

    # counts the total number of each class
    total_classification_count = []
    for classes in classifications: 
        count = 0
        for i in range(len(neighbor_distances)): 
            if (classes == neighbor_distances[i][1][class_col]): 
                count += 1
        total_classification_count.append([count, classes])

    # finds the majority class 
    max = 0
    max_class = None
    for i in range(len(total_classification_count)): 
        if (total_classification_count[i][0] > max): 
            max = total_classification_count[i][0]
            max_class = total_classification_count[i][1]
  
    return max_class

# tries all k-values possible to find the most accurate k 
def find_best_k(len_of_dataset, val_dataset, tree): 

    my_results=open("my_results.txt","w")

    majority_class = None
    accuracy = []

    # Test all k-values 
    for k in range (0, len_of_dataset): 
        success_count_for_k = 0

        # predict classification for each row in the validation dataset
        for i in range(len(val_dataset)): 

            # find the distance from the row to all other points in the tree
            distance_of_all_neighbors = []
            tree.nearest(val_dataset[i], distance_of_all_neighbors) 
            distance_of_all_neighbors.sort(key = lambda x: x[0])

            # find the majority class
            if (k == 0 and i == 0): 
                majority_class = find_majority_class(distance_of_all_neighbors)

            # use weighted knn to predict the class of the row 
            prediction = weighted_classification(distance_of_all_neighbors, majority_class, k)
            if (prediction == val_dataset[i][class_col]): 
                success_count_for_k += 1

        # determine accuracy for k 
        acc = ( success_count_for_k / len(val_dataset) )
        #print(k, acc)
        my_results.write(str(acc) + '\n')
        accuracy.append([k, acc])

    # sort the accuracy to find the best 
    accuracy.sort(key=lambda x: x[1])
    best_accuracy = accuracy[-1][1]
    
    # Check to see if more than one k has the same accuracy
    acc_counter = 0
    best_accuracy_list = []
    for j in range (len(accuracy)-1, -1, -1): 
        if accuracy[j][1] == best_accuracy: 
            acc_counter += 1
            best_accuracy_list.append(accuracy[j])
        elif (accuracy[j][1] != best_accuracy): 
            break

    # if there is more than one k with the best accuracy, find the median k for the best accuracy 
    best_accuracy_list.sort(key=lambda x: x[0])
    if (len(best_accuracy_list) > 1): 
        average_best_k = best_accuracy_list[int(len(best_accuracy_list)/2)][0]
    else: 
        average_best_k = best_accuracy_list[0][0]
    
    return average_best_k
  
# run the knn algorithm on selected k 
def run_knn_for_k(test_dataset, tree, k): 

    success = 0
    # predict classification for each row in the validation dataset
    for i in range(len(test_dataset)): 

        # find the distance from the row to all other points in the tree
        distance_of_all_neighbors = []
        tree.nearest(test_dataset[i], distance_of_all_neighbors) 
        distance_of_all_neighbors.sort(key = lambda x: x[0])

        # set the majority class
        if (i == 0): 
            majority_class = find_majority_class(distance_of_all_neighbors)

        # use weighted knn to predict the class of the row 
        prediction = weighted_classification(distance_of_all_neighbors, majority_class, k)
        if (prediction == test_dataset[i][class_col]): 
            success += 1

    # determine accuracy for k 
    accuracy = ( success / len(test_dataset) )
    
    return accuracy
    

if __name__ == '__main__': 

    # Load training data in from file 
    training_dataset = []
    load_file("train.csv", training_dataset)

    # find the min and max of all the attributes in the dataset 
    x_min_max = find_xmin_xmax(training_dataset)

    # normalize the data 
    normalize(training_dataset, x_min_max)

    # Load validation data in from file and normalize the data
    val_dataset = [] 
    load_file("val.csv", val_dataset)
    normalize(val_dataset, x_min_max)

    # Load test data in from file and normalize the data
    test_dataset = []
    load_file("test.csv", test_dataset)
    normalize(test_dataset, x_min_max)

    # set the number of attributes in the data 
    num_of_attributes = len(training_dataset[0])-1

    # set the end index of the data
    end_index = len(training_dataset)-1

    # set the length of the dataset/ how many rows it has
    len_of_dataset = len(training_dataset)

    # Construct a kd-tree to hold data
    tree = kd_tree(num_of_attributes) 
    # Fill in tree with training data
    fill_tree(training_dataset, num_of_attributes, 0, 0, end_index, tree)

    # run knn and find the best k value
    best_k = find_best_k(len_of_dataset, val_dataset, tree)

    # using the best k value, run knn on the test data set to find the accuracy  
    accuracy_for_best_k = run_knn_for_k(test_dataset, tree, best_k)

    #print the best k value found, and the accuracy on the test data 
    print("Best k: ", best_k)
    print("Accuracy on test data: ", accuracy_for_best_k)




