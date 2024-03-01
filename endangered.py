import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

df = pd.get_dummies(df, columns=['Habitat Type', 'Food Chain'])

X = df.drop(['Species', 'Extinction Risk'], axis=1)
y = df['Extinction Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

species = ['Tiger', 'Elephant', 'Rhino', 'Panda', 'Gorilla']
population_size = [3, 5, 2, 1, 0.5]
extinction_risk = [1, 1, 1, 0, 0]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=population_size, y=extinction_risk, hue=extinction_risk, s=100, palette='coolwarm')
plt.title('Population Size vs. Extinction Risk')
plt.xlabel('Population Size (thousands)')
plt.ylabel('Extinction Risk (1 = Yes, 0 = No)')
plt.legend(title='Extinction Risk', loc='upper right')
plt.grid(True)
plt.show()
