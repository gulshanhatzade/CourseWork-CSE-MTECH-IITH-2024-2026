# the foml hackthon code
# Group NAME rEBooT rEBeLs
# Group member nmae Anurag Sarva and Gulshan Shriram Hatzade
# Group memeber rollnumber cs24mtech14003 and cs24mtech14006
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import argparse

# here are i am loading the training dataset
train_data_set_read_f = pd.read_csv('train.csv')

# below i am creating a list of feachers, which is the feachers i am going to consider the test and train datasets.
features_to_keep_in_teh_data_set = [
    'UID', 'CropFieldConfiguration', 'CropSpeciesVariety', 'CultivatedAndWildArea', 'CultivatedAreaSqft1',
    'FarmClassification', 'FieldConstructionType', 'FieldEstablishedYear', 'FieldShadeCover', 'FieldSizeSqft',
    'FieldZoneLevel', 'HarvestProcessingType', 'HarvestStorageSqft', 'HasGreenHouse', 'HasPestControl',
    'LandUsageType', 'Latitude', 'Longitude', 'MainIrrigationSystemCount', 'NaturalLakePresence',
    'NumberGreenHouses', 'PartialIrrigationSystemCount', 'PerimeterGuardPlantsArea', 'ReservoirType',
    'ReservoirWithFilter', 'SoilFertilityType', 'TotalCultivatedAreaSqft', 'TotalReservoirSize',
    'TotalTaxAssessed', 'TotalValue', 'TypeOfIrrigationSystem', 'UndergroundStorageSqft', 'WaterAccessPoints',
    'WaterAccessPointsCalc', 'WaterReservoirCount', 'Target'
]
train_data_set_read_f = train_data_set_read_f[features_to_keep_in_teh_data_set]

# After droping the columns, i am again droping the coloum with has >70% missing values.
train_data_set_read_f = train_data_set_read_f.loc[:, train_data_set_read_f.isnull().mean() < 0.7]

# in the below line of code i am, applying the feature engineering.
# i am checking in the dataset if both the 'PrimaryCropAreaSqft' and 'FarmEquipmentArea' feachers are present in the DataFrame
if 'PrimaryCropAreaSqft' in train_data_set_read_f.columns and 'FarmEquipmentArea' in train_data_set_read_f.columns:
    # here i am calculating the area per unit in the farm equipment and going ot store this, in a new column which is 'Area_per_Equipment'
    # to avoiding the error, i am adding 1 to this coloumn 'FarmEquipmentArea'
    train_data_set_read_f['Area_per_Equipment'] = train_data_set_read_f['PrimaryCropAreaSqft'] / (train_data_set_read_f['FarmEquipmentArea'] + 1)
    # and calculating the square root of this coloumn 'PrimaryCropAreaSqft' and going to store it into the new column which is 'Area_Sqrt'
    train_data_set_read_f['Area_Sqrt'] = np.sqrt(train_data_set_read_f['PrimaryCropAreaSqft'])

# now i am separating the features of the variable X and the targeting the variable y from the DataFrame
# and also droping the columns like 'UID' and 'Target' from the X because they are not the features
# and then ignoring the errors, only if this olumns are don't present
X = train_data_set_read_f.drop(columns=['UID', 'Target'], errors='ignore')
# here i am maping the target variable such as 'Target' to the numerical values: 'low' to 0, 'medium' to 1, and 'high' to 2
# This thing will helps to converting the categorical target labels to the numerical value for the training of a model.
y = train_data_set_read_f['Target'].map({'low': 0, 'medium': 1, 'high': 2})

# here i am going to imputing the missing values and ,then also scaling the features by using RobustScaler
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
X = pipeline.fit_transform(X)

# here i am spliting the data training and testing
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# here i am defining and configuring the Random Forest with the hyperparameters
clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# here is the updated parameter grid, only for the focused tuning
param_dist = {
    'n_estimators': [150, 250, 350],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 7],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2']
}
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=15,
                                   scoring='f1_macro', cv=3, random_state=42, n_jobs=-1)

# now we are fiting the model with the hyperparameter tuning metric
random_search.fit(X_train, y_train)
best_clf = random_search.best_estimator_

# here we are validating the models performance
y_pred = best_clf.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
print("Optimized Validation Macro F1 Score:", f1)

# after all this i am load and prepare the test data
test_df = pd.read_csv('test.csv')
test_df = test_df[features_to_keep_in_teh_data_set[:-1]]  # Exclude 'Target' as it's not in the test data

# Dropping columns missing values greater than 70%
test_df = test_df.loc[:, test_df.isnull().mean() < 0.7]

# in the below line of code i am, applying the feature engineering.
# i am checking in the dataset if both the 'PrimaryCropAreaSqft' and 'FarmEquipmentArea' feachers are present in the DataFrame
if 'PrimaryCropAreaSqft' in test_df.columns and 'FarmEquipmentArea' in test_df.columns:
     # here i am calculating the area per unit in the farm equipment and going ot store this, in a new column which is 'Area_per_Equipment'
     # to avoiding the error, i am adding 1 to this coloumn 'FarmEquipmentArea'
    test_df['Area_per_Equipment'] = test_df['PrimaryCropAreaSqft'] / (test_df['FarmEquipmentArea'] + 1)
     # and calculating the square root of this coloumn 'PrimaryCropAreaSqft' and going to store it into the new column which is 'Area_Sqrt'
    test_df['Area_Sqrt'] = np.sqrt(test_df['PrimaryCropAreaSqft'])

# now here we are transforming the test dataset by dropping UID
X_test = pipeline.transform(test_df.drop(columns=['UID'], errors='ignore'))

# now are predicting on test set
test_predictions = best_clf.predict(X_test)

# now in below code we are preparing submission file as uid & target
submission = pd.DataFrame({
    'UID': test_df['UID'],
    'Target': test_predictions
})
submission['Target'] = submission['Target'].map({0: 'low', 1: 'medium', 2: 'high'})
submission.to_csv('output.csv', index=False)

# This is function for handling predictions & file output
def make_predictions(test_fname, predictions_fname):
    # now we are Loading test data set
    test = pd.read_csv(test_fname)

    # Applying same pre processing steps to test data set
    test = test[features_to_keep_in_teh_data_set[:-1]]  # Drop 'Target' column
    test = test.loc[:, test.isnull().mean() < 0.7]  # Drop columns with >70% missing values

    # in the below line of code i am, applying the feature engineering.
    # i am checking in the dataset if both the 'PrimaryCropAreaSqft' and 'FarmEquipmentArea' feachers are present in the DataFrame
    if 'PrimaryCropAreaSqft' in test.columns and 'FarmEquipmentArea' in test.columns:
        # here i am calculating the area per unit in the farm equipment and going ot store this, in a new column which is 'Area_per_Equipment'
        # to avoiding the error, i am adding 1 to this coloumn 'FarmEquipmentArea'
        test['Area_per_Equipment'] = test['PrimaryCropAreaSqft'] / (test['FarmEquipmentArea'] + 1)
        # and calculating the square root of this coloumn 'PrimaryCropAreaSqft' and going to store it into the new column which is 'Area_Sqrt'
        test['Area_Sqrt'] = np.sqrt(test['PrimaryCropAreaSqft'])

    # Transforming data of test using same pipeline
    X_test = pipeline.transform(test.drop(columns=['UID'], errors='ignore'))

    # now here i am making predictions on dataset of test
    predictions = best_clf.predict(X_test)

    # preparing result DataFrame in below code
    result = pd.DataFrame({
        'UID': test['UID'],
        'Target': predictions
    })
    result['Target'] = result['Target'].map({0: 'low', 1: 'medium', 2: 'high'})

    # Saving these predictions to o/p file
    result.to_csv(predictions_fname, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, help="Test CSV file")
    parser.add_argument("--predictions-file", type=str, help="File to save predictions")
    args = parser.parse_args()

    # Make predictions based on the passed arguments
    make_predictions(args.test_file, args.predictions_file)

    print(f"{args.predictions_file} file created successfully.")





