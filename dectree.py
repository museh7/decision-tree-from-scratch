import pandas as pd
import os

#gini impurity function calculates gini in a given list
y=[1,0,0,1,0,1,0,0,0,0,0]
def gini(y):
    numdict = {}
    total = len(y)
    sum_squared_probs = 0
    for item in y:
        if item not in numdict:
            numdict[item] = 1
        else:
            numdict[item] += 1
    for count in numdict.values():
        sum_squared_probs += (count/total)**2
    return(1 - sum_squared_probs)
x= gini(y)

df1 = pd.read_csv("churn_data.csv")


# Test every feature and threshold, split churn labels left/right,
def best_split(df1):
    #remove el churn column
    df2 = df1.loc[:, df1.columns != 'churn']
    best_gini = 1 
    best_feature = "" 
    best_threshold = 0
    for column in df2.columns:
        # loop over each value in each column and compare to fixed threshold
        for threshold in df1[column].unique():
            right_labels = []
            left_labels = []
            for index, row in df1.iterrows():
                if row[column] < threshold:
                    left_labels.append(row['churn'])
                else:
                    right_labels.append(row['churn'])
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            left_gini = gini(left_labels)
            right_gini = gini(right_labels)
            #weighted gini for this feature and threshold  and keep the split with the lowest score.
            weighted_gini = (left_gini * (len(left_labels) / (len(left_labels) + len(right_labels)))+ right_gini * (len(right_labels) / (len(right_labels) + len(left_labels))))
            if weighted_gini < best_gini:
                best_gini = weighted_gini 
                best_feature = column 
                best_threshold = threshold
    return best_feature,best_threshold,best_gini
# Creating a tree node class
class TreeNode:
    #decision nodes and leaf nodes, decisions won't have a prediction leaf's will
    def __init__(self, feature, threshold, child_left, child_right, prediction):
        self.feature = feature
        self.threshold = threshold
        self.child_left = child_left
        self.child_right = child_right
        self.prediction = prediction

#build_tree function using recusrsion
def build_tree(df1):
    #check for stopping conditions
    churn_list = df1['churn']
    #subset already pure?
    if gini(churn_list) == 0:
        return TreeNode(None, None, None, None, churn_list.iloc[0]) #returns prediction as first element of subset if pure alr
    x = best_split(df1)
    # No valid split found
    if x[0] == "":
        #predictes label with higher count
        prediction = churn_list.value_counts().idxmax()
        return TreeNode(None, None, None, None, prediction)
    #find weighted gini and compare to gini(churn_list)
    split_gini = x[2]
    #second stopping condition are splites improving gini purity?
    if split_gini >= gini(churn_list):
        prediction = churn_list.value_counts().idxmax()
        return TreeNode(None, None, None, None, prediction)
    #make the dataframe smaller by using best thresholds 
    left_subset = df1[df1[x[0]] < x[1]]
    right_subset = df1[df1[x[0]] >= x[1]]
    #recursive step keep calling build_tree until tree is done
    child_left = build_tree(left_subset)
    child_right = build_tree(right_subset)
    return TreeNode(x[0], x[1], child_left, child_right, None)
    
# STEP 8: WRITE predict_one()

# STEP 9: WRITE predict() FOR MANY ROWS

# STEP 10: TEST AGAINST YOUR CHURN DATASET