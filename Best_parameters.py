import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Data Preprocessing
# Load the dataset
df = pd.read_csv("data2.csv", delimiter=";")

# Drop irrelevant columns
df.drop(['Id', 'Name', 'Surname', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])

# Step 2: Feature Selection
features = ['Class', 'Sex', 'Age', 'Family', 'Embarked']
X = df[features]
y = df['Survival']

# Step 3: Model Training
# Train the Random Forest classifier
model = RandomForestClassifier(n_estimators=80, random_state=10)
model.fit(X, y)

# Step 4: Find Parameters with Highest Chance of Survival
def suggest_parameters():
    highest_survival_prob = 0
    best_parameters = None
    for Class in range(1, 4):
        for Sex in range(2):
            for Age in range(1, 101):
                for Family in range(8):
                    for Embarked in range(3):
                        new_passenger = [[Class, Sex, Age, Family, Embarked]]
                        survival_prob = model.predict_proba(new_passenger)[0][1]  # Probability of survival
                        if survival_prob > highest_survival_prob:
                            highest_survival_prob = survival_prob
                            best_parameters = (Class, Sex, Age, Family, Embarked)
    return best_parameters, highest_survival_prob

best_params, survival_prob = suggest_parameters()

print("Parameters with the highest chance of survival:")
print(f"Class: {best_params[0]}, Sex: {best_params[1]}, Age: {best_params[2]}, Family: {best_params[3]}, Embarked: {best_params[4]}")
print(f"Survival Probability: {survival_prob}")
