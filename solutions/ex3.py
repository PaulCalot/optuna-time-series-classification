def train_torch(trial, obj, train_data, valid_data, epochs=10):
    train_loader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=64, 
                                              shuffle=True, 
                                              num_workers=1)
    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                               batch_size=64, 
                                               shuffle=True, 
                                               num_workers=1)
    
    obj.train(train_loader, epochs)
    
    return 1.0 - obj.score(valid_loader)

def cnn_naive_objective(trial, train_data, valid_data, pruning=True, epochs=10):
    n_layers = trial.suggest_int('n_layers', 1, 5) # 1, 3
    c_in_list = [1]
    c_out_list = []
    k_list = []
    h = train_data.data[0].shape[0]
    w = train_data.data[0].shape[1]
    for n in range(n_layers):
        if(n != 0):
            c_in_list.append(c_out_list[-1])
        c_out_list.append(trial.suggest_int(f'c_out_{n}', 1, 2)) 
        k_list.append(trial.suggest_int(f'k_{n}', 3, 5, 2)) # 3, 3, 2
        w = get_w_out(w, k_list[-1])
        h = get_h_out(h, k_list[-1])

    convnet = SmallConvNet(
        c_in_list=c_in_list,
        c_out_list=c_out_list,
        kernel_size_list=k_list,
        output_size=int(h*w*c_out_list[-1])
    )

    sug_lr = trial.suggest_float('lr', 1e-5, 1, log=True)
    classifier = TorchClassifier(
        convnet,
        lr=sug_lr,
        verbose=False
    )
    return train_torch(trial, classifier, train_data, valid_data, epochs)


# ---------------- Running the study ------------- #
timeout = 180
objective = partial(cnn_naive_objective, train_data=train_data, valid_data=test_data, pruning=False)
study_no_pruning = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())
study_no_pruning.optimize(objective, timeout=timeout, show_progress_bar=True)  # Invoke optimization of the objective function.
print('Best score={:0.5f} obtained for parameters={} after {} trials.'.format(study_no_pruning.best_value, study_no_pruning.best_params, len(study_no_pruning.trials)))