# Import helpful libraries
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

"""
We are trying to predict the prices of diamonds based on their attributes
That's why this problem is classified as regression, which the XGBoost is the most suitable model to train. 
"""



# ==========================================
# Data Manipulation
# ==========================================

# Load the data
data = pd.read_csv('data/diamonds.csv', index_col=0) 

#print("Dataset cols:", data.columns.tolist())
#print("\nData types:\n", data.dtypes)

#Separate target
y = data.price
#Eliminate price to avoid leakage
X = data.drop(['price'], axis=1)


#print(data.describe())
#print(data.head())

#Find Categoric Columns
categorical_cols = X.select_dtypes(include=['object', 'category', 'str']).columns

low_cardinality_cols = [cname for cname in categorical_cols if X[cname].nunique() <= 10]

high_cardinality_cols = [cname for cname in categorical_cols if X[cname].nunique() > 10]

#There are 3 categorical variables: cut, color, clarity and all of them are Ordinal
print("Low card:",low_cardinality_cols)
print("High card:", high_cardinality_cols)

#Returns True if there is at least one null value in the table
print(X.isnull().values.any()) 
# There are no null values in our dataset so we wont bother on aplying null codification techniques


#Split values
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1)


# Feature engineering
X_train['volume'] = X_train['x'] * X_train['y'] * X_train['z']
X_train = X_train.drop(['x', 'y', 'z'], axis=1)

X_val['volume'] = X_val['x'] * X_val['y'] * X_val['z']
X_val = X_val.drop(['x', 'y', 'z'], axis=1)


# Ordinal encoding
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

ordinal_encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
cat_cols = ['cut', 'color', 'clarity']

X_train[cat_cols] = ordinal_encoder.fit_transform(X_train[cat_cols])
X_val[cat_cols] = ordinal_encoder.transform(X_val[cat_cols])



# ==========================================
# Model Training
# ==========================================

#Define a XGBoost model
reg_model = XGBRegressor(
    n_estimators=500,
    max_depth=20,
    learning_rate=0.1,
    subsample=1,
    random_state=42
)

reg_model.fit(X_train,y_train)

reg_model_predictions = reg_model.predict(X_val)



# ==========================================
# Results
# ==========================================

# MAE of the XGBRegressor model
reg_model_mae =  mean_absolute_error(reg_model_predictions, y_val)  #MAE: 282.59
reg_model_mape = mean_absolute_percentage_error(y_val, reg_model_predictions)

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

#Real Price vs Predicted Price
plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_val, y=reg_model_predictions, alpha=0.5)

plt.title("Real Price vs Predicted Price")
plt.xlabel("Real Price")
plt.ylabel("Predicted Price")
#plt.savefig("figures/real_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()


#Residual Plot
plt.figure(figsize=(8, 6))

residual  = y_val - reg_model_predictions
sns.scatterplot(x=reg_model_predictions, y=residual, alpha=0.5)

plt.title("Residual Plot")
plt.xlabel("Predicted Prices")
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

plt.figure(figsize=(8, 6))

sns.barplot(x=importance.Feature, y=importance.Score, palette='viridis')

plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
#plt.savefig("figures/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()



# ==========================================
# Serialization
# ==========================================
#
## Save the model
#joblib.dump(reg_model, "models/xgb_diamond_model.pkl")
#joblib.dump(ordinal_encoder, "models/ordinal_encoder.pkl")
#
#print("Model y OrdinalEncoder saved succesfully.")

## Load the model
# 1. Load the previously saved model and encoder
#loaded_model = joblib.load("xgb_diamond_model.pkl")
#loaded_encoder = joblib.load("ordinal_encoder.pkl")

# 2. (Example) You have a new DataFrame with a new diamond
# new_data = pd.read_csv('new_diamond.csv')

# 3. Apply the same feature engineering 
# new_data['volume'] = new_data['x'] * new_data['y'] * new_data['z']
# new_data = new_data.drop(['x', 'y', 'z'], axis=1)

# 4. Transform the categorical columns with the loaded encoder
# cat_cols = ['cut', 'color', 'clarity']
# new_data[cat_cols] = loaded_encoder.transform(new_data[cat_cols])

# 5. Make the prediction
# predictions = loaded_model.predict(new_data)


#Small method to find the best parameters
def best_params_optimized(X_train, y_train):
    model = XGBRegressor(random_state=42)
    
    # Parameters
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600],
        'max_depth': [10, 15, 20],  
        'learning_rate': [0.1, 0.2, 0.3],
        'subsample': [0.5, 0.8, 1.0]
    }
    
    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=3,          
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

#best_params_optimized(X_train,y_train)
#Best params: {'learning_rate': 0.1, 'max_depth': 20, 'n_estimators': 500, 'subsample': 1.0}