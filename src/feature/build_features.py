import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    """Preprocess the data."""
    try:
        data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
        data = data.drop(['Serial_No'], axis=1)
        data = pd.get_dummies(data, columns=['University_Rating', 'Research'])
        return data
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def split_data(data):
    """Split the data into training and testing sets."""
    try:
        x = data.drop(['Admit_Chance'], axis=1)
        y = data['Admit_Chance']
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
        scaler = MinMaxScaler()
        scaler.fit(xtrain)
        Xtrain = scaler.transform(xtrain)
        Xtest = scaler.transform(xtest)
        return Xtrain, Xtest, ytrain, ytest
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise