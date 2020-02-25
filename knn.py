"""
Student: Vishnu Rampersaud 
Course: CSCI 49900 
Instructor: Dey 

This is an implementation of knn. 

**Note: Before using the code, please make sure your data is going to be accurately parsed
    - in the load_file() function, you can omit rows & columns as necessary 
    - set the class_col global variable so the program knows the column to omit when manipulating data.
      This column should only be set to either -1 for the last column, or 0 for the first column.
    - in the find_xmin_xmax(), normalize(), euclidean_distance() functions omit the classification
      column from being tampered with. This is solved by setting the class_col variable
    - There is also a weighted and unweighted version of knn. By default, this program uses the weighted knn. 
      To change to unweighted knn: 
        - in the find_best_k() function, find the line that says "prediction = weighted_classification(arguments)"
        - change weighted_classification() to unweighted_classification(); keep the same parameters
        - do the same for the run_knn_for_k() function
"""
from math import sqrt 

# the classification column can only be either, '-1' which is the last column, or '0' which is the first column
class_col = -1

# load file into a list data structure
def load_file (filename, dataset): 

    # open file and store into data_file object 
    data_file = open(filename, 'r')

    # separate each row by commas and store the values into a list 
    # remove unwanted data and only keep attributes and classification that will be used
    for line in data_file: 
        if not line: 
            continue
        line = line.strip()
        data_row = line.split(",")
        data_row = data_row[1:-1]

        for i in range(len(data_row)): 
            data_row[i] = int(data_row[i])

        dataset.append(data_row)
   
    pass 

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
        skip_class_col = class_col
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
        skip_class_col = class_col
        for i in range(len(dataset)):
            # omit the classification column 
            for j in range(skip_class_col, len(dataset[i])):  
                dataset[i][j] = (dataset[i][j] - x_min_max[0]) / (x_min_max[1] - x_min_max[0])

    return dataset

# Find the euclidean distance between two rows
def euclidean_distance(test_row, training_row): 

    distance = 0.0

    if class_col < 0:
        skip_class_col = abs(class_col)
        for i in range(len(test_row)-skip_class_col): 
            distance += ((test_row[i] - training_row[i])**2)

    else: 
        skip_class_col = class_col
        for i in range(skip_class_col, len(test_row)): 
            distance += ((test_row[i] - training_row[i])**2)
    
    eucl_distance = sqrt(distance)
    return eucl_distance 

# compute the distance between one row and all the other rows in the training set
# store the results into a list 
def compute_all_neighbors(test_row, training_dataset): 

    neighbor_distances = []
    for train_row in training_dataset: 
        distance = euclidean_distance(test_row, train_row)
        neighbor_distances.append([distance, train_row])

    neighbor_distances.sort(key=lambda x: x[0])

    return neighbor_distances

# Weighted knn prediction 
def weighted_classification(neighbor_distances, majority_class, k): 

    # find all present classificatons in the data and store them into a list 
    classifications = []
    for i in range(k):
        if neighbor_distances[i][1][class_col] not in classifications: 
            classifications.append(neighbor_distances[i][1][class_col])

    # sums up the weighted value for each classification in k-nearest neighbors
    total_classification_count = []
    for classes in classifications: 
        count = 0
        for i in range(k): 
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

def unweighted_classification(neighbor_distances, majority_class, k): 

    # find all present classificatons in the data and store them into a list 
    classifications = []
    for i in range(k):
        if neighbor_distances[i][1][class_col] not in classifications: 
            classifications.append(neighbor_distances[i][1][class_col])

    # sums up the unweighted value for each classification in k-nearest neighbors
    total_classification_count = []
    for classes in classifications: 
        count = 0
        for i in range(k): 
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
def find_best_k(training_datset, val_dataset): 

    my_results=open("my_results.txt","w")

    majority_class = None
    accuracy = []

    # Test all k-values 
    for k in range (0, len(training_dataset)): 
        success_count_for_k = 0

        # predict classification for each row in the validation dataset
        for i in range(len(val_dataset)):
            
            # find the distance from the row to all other points in the tree
            distance_of_all_neighbors = compute_all_neighbors(val_dataset[i], training_dataset)
            
            # find the majority class 
            if (k == 0 and i == 0): 
                majority_class = find_majority_class(distance_of_all_neighbors)
            
            # use weighted knn to predict the class of the row     
            prediction = unweighted_classification(distance_of_all_neighbors, majority_class, k)
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
def run_knn_for_k(training_dataset, test_dataset, k): 

    success = 0
    # predict classification for each row in the validation dataset
    for i in range(len(test_dataset)): 

        # find the distance from the row to all other points in the tree
        distance_of_all_neighbors = compute_all_neighbors(test_dataset[i], training_dataset)
        
        # set the majority class
        if (i == 0): 
            majority_class = find_majority_class(distance_of_all_neighbors)
        
        # use weighted knn to predict the class of the row    
        prediction = unweighted_classification(distance_of_all_neighbors, majority_class, k)
        if (prediction == test_dataset[i][class_col]): 
            success += 1

    # determine accuracy for k
    accuracy = ( success / len(test_dataset) )
    
    return accuracy
    
    pass


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

    # run knn and find the best k value
    best_k = find_best_k(training_dataset, val_dataset)

    # using the best k value, run knn on the test data set to find the accuracy  
    accuracy_for_best_k = run_knn_for_k(training_dataset, test_dataset, best_k)

    #print the best k value found, and the accuracy on the test data 
    print("Best k: ", best_k)
    print("Accuracy on test data: ", accuracy_for_best_k)
