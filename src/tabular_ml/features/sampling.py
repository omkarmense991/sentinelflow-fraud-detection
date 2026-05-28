from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler, SMOTE


def get_sampling_methods():

    return {
        "none": None,
        "undersampling": RandomUnderSampler(random_state=42),
        "oversampling": RandomOverSampler(random_state=42),
        "smote": SMOTE(random_state=42),
    }
