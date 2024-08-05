Neural Networks Solution 

Project Stucture
├── data/
│   └── Admission.csv               # The dataset file
├── src/
│   ├── __init__.py                 # Makes src a package
│   ├── data/
│   │   ├── __init__.py             # Makes data a package
│   │   └── data_loader.py          # Module for loading data
│   ├── feature_engineering/
│   │   ├── __init__.py             # Makes feature_engineering a package
│   │   └── feature_engineering.py  # Module for feature engineering
│   ├── model/
│   │   ├── __init__.py             # Makes model a package
│   │   └── model.py                # Module for model operations
│   ├── visualization/
│   │   ├── __init__.py             # Makes visualization a package
│   │   └── visualization.py        # Module for data visualization
├── tests/
│   ├── __init__.py                 # Makes tests a package
│   ├── test_data_loader.py         # Unit tests for data_loader.py
│   ├── test_feature_engineering.py # Unit tests for feature_engineering.py
│   ├── test_model.py               # Unit tests for model.py
│   └── test_visualization.py       # Unit tests for visualization.py
├── main.py                         # Main script to run the project
├── README.md                       # Project description and instructions
├── requirements.txt                # List of dependencies
└── .gitignore                      # Git ignore file


Steps to Push code from VS code to Github.
First authenticate your githib account and integrate with VS code. Click on the source control icon and complete the setup.
1. Click terminal and open new terminal
2. git config --global user.name "Swapnilin"
3. git config --global user.email swapnilforcat@gmail.com
4. git init
5. git add .
6. git commit -m "Your commit message"