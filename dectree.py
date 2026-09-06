import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
#splitting into training and test
df1 = pd.read_csv("churn_data.csv")

train_df, test_df = train_test_split(
    df1,
    test_size=0.2,
    random_state=42,
    stratify=df1['churn']
)

#gini impurity function calculates gini in a given list
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
#prediction for one custmer
def predict_one(row, tree):
    #check if already a node or decision 
    if tree.prediction is not None:
        return tree.prediction
    else:
# Follow the left or right branch based on this node's feature and threshold
        if row[tree.feature] < tree.threshold:
            return predict_one(row, tree.child_left)
        else:
            return predict_one(row, tree.child_right)

# predicttion for many customers
def predict(rows,tree):
    pre_dict = {}
    for index, row in rows.iterrows():
        prediction = predict_one(row, tree)
        pre_dict.update({index: prediction})
    return pre_dict
def main():
    tree = build_tree(train_df)
    predictions = predict(test_df, tree)
    #have predictions and actuals in a csv
    results = test_df[['churn']].copy()
    results['predicted_churn'] = results.index.map(predictions)
    results.to_csv("predictions.csv")

    #calculate accuracy 
    correct = 0
    for index, row in results.iterrows():
        if row['predicted_churn'] == row['churn']:
            correct += 1

    x = (correct / len(results)) * 100
    print(f"accuracy is {x}%")
        





if __name__ == "__main__":
    main()



