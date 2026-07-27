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



       

# Test every feature and threshold, split churn labels left/right,
def best_split():
    df1 = pd.read_csv("churn_data.csv")
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
            weighted_gini = (left_gini*(len(left_labels)/(len(left_labels)+len(right_labels)))+ right_gini*(len(right_labels)/(len(right_labels)+len(left_labels))))
            if weighted_gini < best_gini :
                best_gini = weighted_gini 
                best_feature = column 
                best_threshold = threshold





# STEP 6: CREATE A TREE NODE CLASS

# STEP 7: WRITE build_tree() USING RECURSION

# STEP 8: WRITE predict_one()

# STEP 9: WRITE predict() FOR MANY ROWS

# STEP 10: TEST AGAINST YOUR CHURN DATASET