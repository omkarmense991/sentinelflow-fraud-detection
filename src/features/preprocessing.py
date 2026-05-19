from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(df):

    X = df.drop("Class", axis=1)

    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    # test data must use SAME mean/std learned from training data, as we dont learn from test data, to avoid data leakage
    X_test_scaled = scaler.transform(X_test)
    # use same scaler which is  used during training.
    return scaler, X_train_scaled, X_test_scaled
