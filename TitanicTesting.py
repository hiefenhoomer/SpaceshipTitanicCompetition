import pandas as pd
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from CreatePipeline import create_pipeline

set_config(transform_output='pandas')


def get_features():
    return {
        'bin': ['CryoSleep', 'cab_Side'],   # 'VIP'],
        'num': ['Age', 'num_Room'],
        'cat': ['cab_Deck', 'HomePlanet', 'Destination'],
        'exp': ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
    }


def train(X, y, test_proportion):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in a forest.
        'max_features': ['sqrt', 'log2'],  # Number of features considered in a split
        'max_depth': [None, 10, 20, 30],  # Maximum tree depths
        'min_samples_split': [2, 5, 10],  # Minimum number of samples to split a node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of sample required to be at a leave node
        'bootstrap': [True, False]  # Each tree is trained on a bootstrap sample of the data.
    }

    cv = 2
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    test_acc = best_model.score(X_test, y_test)

    acc_str = f'{(test_acc * 100):.2f}'
    test_size = f'{(test_proportion * 100):.2f}'
    best_score_str = f'Best score: {best_score}'
    best_model_str = f'Best model: {best_model}'
    test_acc_str = f'Test accuracy: {test_acc}'
    test_size_str = f'Test size: {test_size}'
    metrics = f'{test_acc_str}\n{test_size_str}\n{best_score_str}\n{best_model_str}'

    print(metrics)

    with open(f'test_files/test_size_{test_size}_accuracy_{acc_str}_cv{cv}.scores.txt', 'w') as f:
        f.write(metrics)
        f.close()

    feature_importance = best_model.feature_importances_
    mdi_importance = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 8), dpi=400)
    plt.barh(mdi_importance.index, mdi_importance.values)
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance')

    plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.10)
    plt.savefig(f'pictures/test_size_{test_size_str}_accuracy{acc_str}_feature_importance.png')
    plt.show()

    return best_model, test_acc


def test(X, best_model, passenger_id, test_proportion):
    predictions = best_model.predict(X)

    out = pd.DataFrame({
        'PassengerId': passenger_id,
        'Transported': predictions
    })

    out.to_csv(f'submission_files/submission.test_size_{int(test_proportion * 100)}.csv', index=False)


if __name__ == '__main__':
    csv_files = ['csv_files/train.csv', 'csv_files/test.csv']
    target = 'Transported'
    feature_dict = get_features()
    pipe = create_pipeline(feature_dict)
    train_df = pd.read_csv(csv_files[0])
    test_df = pd.read_csv(csv_files[1])
    pipe.fit(train_df)
    transformed_train_df = pipe.transform(train_df)
    transformed_test_df = pipe.transform(test_df)
    # feature_dict['cat'].remove('Destination')
    print(feature_dict)

    test_size = .70
    while test_size > .50:
        features = [feature for feature_list in feature_dict.values() for feature in feature_list]
        model, test_accuracy = train(transformed_train_df[features], transformed_train_df[target],
                                     test_proportion=test_size)

        test(transformed_test_df[features], model, transformed_test_df['PassengerId'],
             test_proportion=test_size)
        print(f'Finished test for test size: {test_size}')
        test_size -= .05
