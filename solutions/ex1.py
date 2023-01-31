def train_sklearn(trial, obj, x, y):
    x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, train_size=0.5, random_state=0)
    obj.fit(x_train, y_train)
    score = obj.score(x_valid, y_valid)
    return score

def svc_objective(trial, x, y):
    sug_C = trial.suggest_float('C', 1e-10, 1e10, log=True)
    sug_gamma_kind = trial.suggest_categorical('gamma_kind', ['auto', 'scale', 'float'])
    if(sug_gamma_kind == 'float'):
        sug_gamma = trial.suggest_float('gamma', 1e-3, 10., log=True)
    else:
        sug_gamma = sug_gamma_kind
    classifier_obj = sklearn.svm.SVC(C=sug_C, gamma=sug_gamma)
    return train_sklearn(trial, classifier_obj, x, y)

def random_forest_classifier_objective(trial, x, y):
    sug_max_depth = trial.suggest_int('max_depth', 2, 32)
    sug_min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 20)
    classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=sug_max_depth, min_samples_leaf=sug_min_samples_leaf)
    return train_sklearn(trial, classifier_obj, x, y)

def gaussian_Process_classifier_objective(trial, x, y):
    sug_length_scale = trial.suggest_float('length_scale', 1e-3, 10, log=True)
    kernel = RBF(length_scale=sug_length_scale)
    classifier_obj = GaussianProcessClassifier(kernel=kernel)
    return train_sklearn(trial, classifier_obj, x, y)

def objective_ex1(trial, x, y):
    classifier_name = trial.suggest_categorical('classifier', ['SVC', 'RandomForestClassifier', 'GaussianProcessClassifier'])
    if(classifier_name == 'SVC'):
        score = svc_objective(trial, x, y)
    elif(classifier_name  == 'RandomForestClassifier'):
        score = random_forest_classifier_objective(trial, x, y)
    elif(classifier_name  == 'GaussianProcessClassifier'):
        score = gaussian_Process_classifier_objective(trial, x, y)
    return score

# ---------------- Running the study ------------- #
timeout = 30
objective = partial(objective_ex1, x=x_train, y=y_train)
study = optuna.create_study(direction='maximize')  # Create a new study.
study.optimize(objective, timeout=timeout)  # Invoke optimization of the objective function.
print('Best score={:0.5f} obtained for parameters={} after {} trials.'.format(study.best_value, study.best_params, n_trials))