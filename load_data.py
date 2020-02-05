import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def load_heart_data(filename):
    """
    Loads heart Classification Data Set
    :return: X (data) and y (labels)

    """

    data = pd.read_csv(filename) 
    
    y = data.target  # 0 - no heart disease 1 - Heart disease     
    to_drop = ['target']
    X = data.drop(to_drop, axis=1)  
    
    return X, y

def load_breast_cancer_data(filename):
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) Data Set
    :return: X (data) and y (labels)
    """

    data = pd.read_csv(filename)

    # y includes our labels and x includes our features
    y = data.diagnosis  # malignant (M) or benign (B)
    to_drop = ['Unnamed: 32', 'id', 'diagnosis']
    X = data.drop(to_drop, axis=1)

    # Convert string labels to numerical values
    y = y.values
    y[y == 'M'] = 1
    y[y == 'B'] = 0
    y = y.astype(int)

    return X, y


def load_bank_data(filename):
    """
    Loads bank Classification Data Set
    return: X (data) and y (labels)
    """

    data = pd.read_csv(filename)  
    
    y = data.Personal_Loan  # 0 - no personal loan, 1 - personal loan 
    to_drop = ['Personal_Loan', 'ZIP_Code', 'ID']      
    X = data.drop(to_drop, axis=1)  
        
    return X, y




