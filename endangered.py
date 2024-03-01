# Load the data (including prey classification)
data = {
    'Species': ['Tiger', 'Elephant', 'Rhino', 'Panda', 'Gorilla'],
    'Habitat Type': ['Forest', 'Grassland', 'Grassland', 'Forest', 'Forest'],
    'Population Size': [3, 5, 2, 1, 0.5],
    'Geographic Range': [100000, 200000, 150000, 50000, 30000],
    'Human Activity Impact': [8, 7, 9, 6, 5],
    'Food Chain': ['Carnivore', 'Herbivore', 'Herbivore', 'Herbivore', 'Omnivore'],
    'Prey': [1, 0, 0, 0, 0],  # 1 for prey, 0 for non-prey
    'Extinction Risk': [1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, columns=['Habitat Type', 'Food Chain'])

# Split the data into features and target variable
X = df.drop(['Species', 'Extinction Risk'], axis=1)
y = df['Extinction Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
