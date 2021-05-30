import optuna
import uuid
import os
import simulate
import glob
from dataclasses import dataclass, field
from typing import List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY_NAME = 'chiwawa-study'
DB_PATH = f'{ROOT}/optuna.db'
TESTCASES = sorted(glob.glob(f'{ROOT}/tools/in/*'))


@dataclass
class Result:
    scores: List[int] = field(default_factory=list)

    def add(self, score: int) -> None:
        self.scores.append(score)

    def average(self) -> float:
        return sum(self.scores) / len(self.scores)

    def percentile(self, percent: float):
        return sorted(self.scores)[len(self.scores) * percent / 100 - 1]


def objective(trial):
    ucb1_bias = trial.suggest_uniform('ucb1_bias', 0, 1000)
    initial_distance = trial.suggest_int('initial_distance', 500, 5000)
    estimate_count = trial.suggest_int('estimate_count', 0, 40)
    # smooth_count = trial.suggest_int('smooth_count', 0, 100)
    position_bias = trial.suggest_uniform('position_bias', 0, 30)

    cpp = f'{ROOT}/main.cpp'
    bin = f'/tmp/bin-{str(uuid.uuid4())}.out'
    simulate.compile(cpp, bin, UCB1_BIAS_PARAM=str(ucb1_bias),
                     INITIAL_DISTANCE_PARAM=str(initial_distance),
                     ESTIMATE_COUNT_PARAM=str(estimate_count),
                     # SMOOTH_COUNT_PARAM=str(smooth_count),
                     POSITION_BIAS_PARAM=str(position_bias),
                     )

    result = Result()

    steps = 0
    for f in TESTCASES:
        score = simulate.simulate(bin, f)
        result.add(score)
        print(f'score: {score}, average: {result.average()}')
        steps += 1

        # これと pruner を入れておくと枝刈りしてくれる
        trial.report(result.average(), steps)
        if trial.should_prune():
            raise optuna.TrialPruned()

    print(result)
    return result.average()


if __name__ == '__main__':
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f'sqlite:///{DB_PATH}', load_if_exists=True, direction='maximize', pruner=pruner)

    study.optimize(objective, n_trials=200)
    print(study.best_trial)
    print(study.best_params)
