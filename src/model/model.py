from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

def train_model(Xtrain, ytrain):
    """Train the MLPClassifier model with updated parameters."""
    try:
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Increased complexity
            max_iter=1000,  # Increased max iterations
            alpha=0.0001,  # L2 regularization
            learning_rate='adaptive',  # Adaptive learning rate
            random_state=123
        )
        model.fit(Xtrain, ytrain)
        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model, Xtest, ytest):
    """Evaluate the model."""
    try:
        ypred = model.predict(Xtest)
        conf_matrix = confusion_matrix(ytest, ypred)
        accuracy = accuracy_score(ytest, ypred)
        return conf_matrix, accuracy
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def perform_grid_search(model, x, y):
    """Perform grid search for hyperparameter tuning."""
    try:
        params = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500, 1000, 1500]
        }
        grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(x, y)
        return grid.best_params_, grid.best_score_
    except Exception as e:
        logger.error(f"Error performing grid search: {str(e)}")
        raise