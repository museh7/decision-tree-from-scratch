import pandas as pd
import os

# STEP 1: WRITE A GINI IMPURITY FUNCTION
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
#print(x)


       

    
# STEP 2: TEST IT ON SIMPLE LABEL LISTS
#done in test_gini.py

# STEP 3: WRITE A FUNCTION THAT SPLITS DATA BASED ON ONE FEATURE AND THRESHOLD
def best_split():
    df = pd.read_csv("churn_data.csv")
    right_set = []
    left_set = []
    gini_left = 0
    gini_right = 0
    for feature in df.columns:
        a = df[feature].unique().tolist()
        for a in range(len(a)):
            for row in df[feature]:
                if row < a:
                    left_set.append(row)
                else:
                    right_set.append(row)
            if gini(right_set) < gini_right and gini(left_set) < gini_left():
                gini_left = gini(left_set)
                gini_right = gini(right_set)
    
    print(gini_left)
    print(gini_right)

best_split()


# STEP 4: LOOP THROUGH ALL POSSIBLE FEATURES AND THRESHOLDS

# STEP 5: PICK THE SPLIT WITH THE LOWEST WEIGHTED GINI

# STEP 6: CREATE A TREE NODE CLASS

# STEP 7: WRITE build_tree() USING RECURSION

# STEP 8: WRITE predict_one()

# STEP 9: WRITE predict() FOR MANY ROWS

# STEP 10: TEST AGAINST YOUR CHURN DATASET