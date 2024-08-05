import logging
import warnings
from sklearn.exceptions import ConvergenceWarning
from src.data.data_loader import load_data
from src.feature.build_features import preprocess_data, split_data
from src.model.model import train_model, evaluate_model, perform_grid_search
from src.visualization.visualization import plot_loss_curve, plot_scatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='admission_prediction.log'
)
logger = logging.getLogger(__name__)

# Filter out ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    try:
        logger.info("Starting admission prediction process")
        
        # Load data
        try:
            data = load_data('Admission.csv')
            logger.info("Data loaded successfully")
            logger.info(f"First few rows of the dataframe:\n{data.head()}")
            logger.info(f"Dataframe information:\n{data.info()}")
        except FileNotFoundError:
            logger.error("Admission.csv file not found")
            return
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return

        # Preprocess data
        try:
            data = preprocess_data(data)
            logger.info("Data preprocessed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return

        # Split data
        try:
            Xtrain, Xtest, ytrain, ytest = split_data(data)
            logger.info("Data split successfully")
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return

        # Train model
        try:
            model = train_model(Xtrain, ytrain)
            logger.info("Model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return

        # Evaluate model
        try:
            conf_matrix, accuracy = evaluate_model(model, Xtest, ytest)
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            logger.info(f"Accuracy: {accuracy}")
            
            print("\n--- Model Evaluation Results ---")
            print(f"Confusion Matrix:\n{conf_matrix}")
            print(f"Accuracy: {accuracy:.2f}")
            print("--------------------------------\n")
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return
        
        # Plot loss curve
        try:
            plot_loss_curve(model)
            logger.info("Loss curve plotted successfully")
        except Exception as e:
            logger.warning(f"Error plotting loss curve: {str(e)}")

        # Perform grid search
        try:
            best_params, best_score = perform_grid_search(model, Xtrain, ytrain)
            logger.info(f"Best Parameters: {best_params}")
            logger.info(f"Best Score: {best_score}")
        except Exception as e:
            logger.error(f"Error performing grid search: {str(e)}")
            return

        # Plot scatter plot
        try:
            plot_scatter(data)
            logger.info("Scatter plot created successfully")
        except Exception as e:
            logger.warning(f"Error creating scatter plot: {str(e)}")

        logger.info("Admission prediction process completed successfully")

    except Exception as e:
        logger.critical(f"Unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()