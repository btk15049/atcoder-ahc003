import optuna
import tuner

if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=tuner.STUDY_NAME, storage=f'sqlite:///{tuner.DB_PATH}', load_if_exists=True, direction='maximize', pruner=pruner)
    print(study.best_trial)
    print(study.best_params)
    optuna.visualization.plot_intermediate_values(study).show()
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_edf(study).show()
    optuna.visualization.plot_contour(study).show()
    # fig = optuna.visualization.plot_param_importances(study)
    print(study.best_trial)
    print(study.best_params)
