import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_curve(model):
    """Plot the loss curve."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Loss', color='blue')
        plt.title('Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logger.warning(f"Error plotting loss curve: {str(e)}")

def plot_scatter(data):
    """Plot a scatter plot of GRE Score vs TOEFL Score."""
    try:
        plt.figure(figsize=(15, 8))
        sns.scatterplot(data=data, x='GRE_Score', y='TOEFL_Score', hue='Admit_Chance')
        plt.show()
    except Exception as e:
        logger.warning(f"Error creating scatter plot: {str(e)}")