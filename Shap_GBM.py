import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

data = pd.read_excel("your_datafile_path")
X = data.drop('EU Prices', axis=XX, errors='ignore')
y = data['EU Prices']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=XX, random_state=XX)

model = GradientBoostingRegressor(n_estimators=XXX, random_state=XX)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

explainer = shap.explainers.Tree(model, X_train)
shap_values = explainer.shap_values(X_test)

plt.rcParams.update({'font.size': 12,
                     'font.family': 'Times New Roman',
                    })

shap.summary_plot(shap_values, X_test, plot_type="bar")

shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.show()