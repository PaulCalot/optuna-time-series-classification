timeout = 60
objective = partial(objective_ex1, x=x_train_, y=y_train_)
study = optuna.create_study(direction='maximize')  # Create a new study.
study.optimize(objective(trial), timeout=timeout, show_progress_bar=True)  # Invoke optimization of the objective function.
print('Best score={:0.5f} obtained for parameters={} after {} trials.'.format(study.best_value, study.best_params, n_trials))