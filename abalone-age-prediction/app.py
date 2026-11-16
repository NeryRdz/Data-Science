# Import helpful libraries
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

"""
We want to predict the age of the abalones.

"""

# ==========================================
# Data Preprocessing
# ==========================================

data = pd.read_csv("data/abalone.csv")
print(data.head())
print(data.info())

# Identify Nulls
print("number of Null values:", data.isnull().sum().sum())
print("Number of duplicated values: ",data.duplicated().sum())

# Identify target
y = data['Rings']*1.5
X = data.drop(['Rings'], axis=1)

# Indentify Categoric columns
categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns

# Identify the cardinality
low_cardinality_cols = [cname for cname in categorical_cols if X[cname].nunique() <= 10]

high_cardinality_cols = [cname for cname in categorical_cols if X[cname].nunique() > 10]

print("Low card:",low_cardinality_cols)
print("High card:", high_cardinality_cols)

# Split values
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=42)

# Feature Engineering
X_train['volume'] = X_train['Length'] * X_train['Diameter'] * X_train['Height']
X_val['volume'] = X_val['Length'] * X_val['Diameter'] * X_val['Height']


X_train['Shell_to_Whole_Ratio'] = X_train['Shell weight'] / X_train['Whole weight']
X_val['Shell_to_Whole_Ratio'] = X_val['Shell weight'] / X_val['Whole weight']


X_train['Meat_to_Whole_Ratio'] = X_train['Shucked weight'] / X_train['Whole weight']
X_val['Meat_to_Whole_Ratio'] = X_val['Shucked weight'] / X_val['Whole weight']


X_train['Meat_to_Shell_Ratio'] = X_train['Shucked weight'] / X_train['Shell weight']
X_val['Meat_to_Shell_Ratio'] = X_val['Shucked weight'] / X_val['Shell weight']

X_train['Viscera_to_Whole_Ratio'] = X_train['Viscera weight'] / X_train['Whole weight']
X_val['Viscera_to_Whole_Ratio'] = X_val['Viscera weight'] / X_val['Whole weight']

X_train = X_train.drop(['Length', 'Diameter', 'Height'], axis=1)
X_val = X_val.drop(['Length', 'Diameter', 'Height'], axis=1)


# Data Encoding (Categorical) --- One Hot Encoding
onehot_encoder = OneHotEncoder(sparse_output=False)
onehot_encoder.set_output(transform="pandas")

encoded_train = onehot_encoder.fit_transform(X_train[low_cardinality_cols])
encoded_val = onehot_encoder.transform(X_val[low_cardinality_cols])


X_train = X_train.drop(columns=low_cardinality_cols)
X_train = pd.concat([X_train, encoded_train], axis=1)

X_val = X_val.drop(columns=low_cardinality_cols)
X_val = pd.concat([X_val, encoded_val], axis=1)


# ==========================================
# Model Training
# ==========================================

#Define a XGBoost model
reg_model = XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=50,
    max_depth=3,
    learning_rate=0.08,
    subsample=0.8,
    random_state=42
)

reg_model.fit(X_train,y_train)

reg_model_predictions = reg_model.predict(X_val)

# ==========================================
# Results
# ==========================================

# MAE of the XGBRegressor model
reg_model_mae =  mean_absolute_error(y_val, reg_model_predictions)  #MAE: 2.25
reg_model_mape = mean_absolute_percentage_error(y_val, reg_model_predictions)
print("-------- Model results --------")
print(f"Validation MAE for optimized XGBRegressor Model: {reg_model_mae:.2f}")
print(f"Validation MAPE for optimized XGBRegressor Model: {reg_model_mape*100:.2f}%")


# Dummy Model
dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)
dummy_predictions = dummy_model.predict(X_val)
dummy_mae = mean_absolute_error(y_val, dummy_predictions)


print("-------- Performance comparison --------")
print(f"Dummy Model MAE : {dummy_mae:,.2f}")
print(f"XGBoost Model MAE: {reg_model_mae:,.2f}")
print(f"XGBoost Model MAPE: {reg_model_mape*100:,.2f}%")


# ==========================================
# Visualization
# ==========================================

#Real Age vs Predicted Age
plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_val, y=reg_model_predictions, alpha=0.5)

plt.title("Real Age vs Predicted Age")
plt.xlabel("Real Age")
plt.ylabel("Predicted Age")
#plt.savefig("figures/real_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()


#Residual Plot
plt.figure(figsize=(8, 6))

residual  = y_val - reg_model_predictions
sns.scatterplot(x=reg_model_predictions, y=residual, alpha=0.5)

plt.title("Residual Plot")
plt.xlabel("Predicted Age")
plt.ylabel("Residual")
#plt.savefig("figures/residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()


#Feature Importance
features = reg_model.feature_names_in_
score = reg_model.feature_importances_

importance = pd.DataFrame({
    'Feature': features,
    'Score' : score
}).sort_values(by="Score",ascending=False)

plt.figure(figsize=(16, 7))

sns.barplot(x=importance.Score, y=importance.Feature, palette='viridis')

plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
#plt.savefig("figures/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


#Small method to find the best parameters
def best_params_optimized(X_train, y_train):
    model = XGBRegressor(random_state=42)
    
    # Parameters
    param_grid = {
        'n_estimators': [50, 100, 200, 300, 400],
        'max_depth': [1, 2, 3, 4, 5],  
        'learning_rate': [0.03, 0.05, 0.08, 0.1],
        'subsample': [0.3, 0.5, 0.8, 1.0],
    }
    
    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=5,          
        n_jobs=-1,     
        verbose=1
    )
    
    # Execute search
    grid_search.fit(X_train, y_train)
    
    # Get the best results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_ 
    
    print(f"Best MAE: {best_score:,.2f}")
    print(f"Best params: {best_params}")
    
    return grid_search.best_estimator_
# Best params: {'learning_rate': 0.08, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8}
#best_params_optimized(X_train, y_train)
