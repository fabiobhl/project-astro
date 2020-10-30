from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter

config = {
    "x": tune.uniform(0, 20)
}

def my_func(config):

    tune.report(value=(config["x"]-1)*(config["x"]-1)-5)

bayesopt = BayesOptSearch(random_search_steps=2)
bayesopt = ConcurrencyLimiter(bayesopt, 1)

result = tune.run(my_func, config=config, metric="value", mode="min", search_alg=bayesopt, num_samples=10)

print(result.get_best_config())
print(result.get_all_configs())