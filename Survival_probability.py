import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def calculate_survival_probability(Class, Sex, Age, Family, Price, Embarked):
    # Load the dataset
    df = pd.read_csv("data2.csv", delimiter= ";")

    # Drop irrelevant columns
    df.drop(['Id', 'Name', 'Surname', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['Embarked'] = le.fit_transform(df['Embarked'])

    # Feature Selection
    features = ['Class', 'Sex', 'Age', 'Family', 'Price', 'Embarked']
    X = df[features]
    y = df['Survival']

    # Model Training
    model = RandomForestClassifier(n_estimators=800, random_state=100)
    model.fit(X, y)

    # Predict survival for the new passenger
    new_passenger = [[Class, Sex, Age, Family, Price, Embarked]]
    survival_probability = model.predict_proba(new_passenger)[0][1]  # Probability of survival

    return survival_probability

# Get user input for passenger information
Class = int(input("Enter passenger's class (1st, 2nd, or 3rd): "))
Sex = int(input("Enter passenger's sex (0 for male, 1 for female): "))
Age = float(input("Enter passenger's age: "))
Family = int(input("Enter number of siblings/spouses aboard: "))
Price = float(input("Enter passenger's fare: "))
Embarked = int(input("Enter passenger's embarked code (0 for CherBourg, 1 for Queenstown, 2 for Southhampton): "))

# Calculate survival probability
survival_probability = calculate_survival_probability(Class, Sex, Age, Family, Price, Embarked)
print(f"Survival Probability: {survival_probability * 100:.2f}%")
