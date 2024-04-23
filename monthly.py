import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

data_path = 'your_datafile_path'
df = pd.read_excel(data_path)

df['date'] = pd.to_datetime(df['Date'])
df['month'] = df['date'].dt.to_period('M')

df['EU_price_next_month'] = df.groupby('month')['EU_price'].shift(-1).rolling(window=30, min_periods=1).mean()
df.dropna(inplace=True)

hf = h2o.H2OFrame(df)

train, test = hf.split_frame(ratios=[.XXX], seed=XXX)

y = 'EU_price_next_month'
X = [name for name in train.columns if name not in [y, 'Date', 'month']]  # 移除不作为特征的列

aml = H2OAutoML(max_models=XXX, seed=XX, max_runtime_secs=XXX)
aml.train(x=X, y=y, training_frame=train)

print(aml.leaderboard)

preds = aml.leader.predict(test)
print(preds)

performance = aml.leader.model_performance(test)
print(f"R^2: {performance.r2()}")

leader_model = aml.leader

if isinstance(leader_model, h2o.estimators.stackedensemble.H2OStackedEnsembleEstimator):
    print("Leader model is a Stacked Ensemble.")

    metalearner = leader_model.metalearner()

    try:
        base_model_ids = metalearner.coef_norm().keys()
    except AttributeError:
        base_model_ids = []

    print("Base model IDs:", base_model_ids)

    for base_model_id in base_model_ids:
        if base_model_id != 'Intercept':
            base_model = h2o.get_model(base_model_id)
            print(f"Base model {base_model_id} feature importance:")
            if hasattr(base_model, 'varimp'):
                print(base_model.varimp())
            else:
                print("This model does not support variable importance.")
else:
    print("Leader model is not a Stacked Ensemble.")