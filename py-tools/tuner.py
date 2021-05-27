import optuna
import uuid
import os
import simulate
import glob
from dataclasses import dataclass, field
from typing import List


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY_NAME = 'ucb-bias1-only'
DB_PATH = f'{ROOT}/optuna.db'
TESTCASES = glob.glob(f'{ROOT}/tools/in/*')


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
    initial_distance = trial.suggest_uniform('initial_distance', 0, 10000)
    estimate_count = trial.suggest_int('estimate_count', 0, 50)
    smooth_count = trial.suggest_int('smooth_count', 0, 50)

    cpp = f'{ROOT}/main.cpp'
    bin = f'/tmp/bin-{str(uuid.uuid4())}.out'
    simulate.compile(cpp, bin, UCB1_BIAS_PARAM=str(ucb1_bias),
                     INITIAL_DISTANCE_PARAM=str(initial_distance),
                     ESTIMATE_COUNT_PARAM=str(estimate_count),
                     SMOOTH_COUNT_PARAM=str(smooth_count))

    result = Result()

    steps = 0
    for f in TESTCASES:
        score = simulate.simulate(bin, f)
        result.add(score)
        print(f'score: {score}, average: {result.average()}')
        steps += 1
        trial.report(result.average(), steps)

    print(result)
    return result.average()


if __name__ == '__main__':
    # 枝刈り https://poyo.hatenablog.jp/entry/2019/03/25/003519#%E6%9E%9D%E5%88%88%E3%82%8A%E3%81%A8%E3%81%AF
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=f'sqlite:///{DB_PATH}', load_if_exists=True, direction='maximize', pruner=pruner)

    study.optimize(objective, n_trials=100)
    print(study.best_trial)
    print(study.best_params)
