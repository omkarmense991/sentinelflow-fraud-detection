from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


def apply_random_undersampling(X, y):

    sampler = RandomUnderSampler(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled


def apply_random_oversampling(X, y):

    sampler = RandomOverSampler(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled


def apply_smote(X, y):

    sampler = SMOTE(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled
