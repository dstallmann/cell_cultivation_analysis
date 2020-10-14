from bayes_opt import BayesianOptimization
from learner import main

# Based on https://github.com/fmfn/BayesianOptimization

def crossval(lr, result_dir, inter_dim_i=1024, bneck_i=512, d_l_f=1e6, r_l_f=1e2, KLD_l_f=1e2):
    score = main(lr, inter_dim_i, bneck_i, d_l_f, r_l_f, KLD_l_f, result_dir)
    print("The score is: " + str(score))
    return score

optimizer = BayesianOptimization(
    f=crossval,
    pbounds={"lr":(1e-5,1e-3),"result_dir":(27,27)},#, "inter_dim_i":[1024,1024], "bneck_i":[512,512]},
    random_state=1,
    verbose=2
)

# Tests a given set of parameters
if __name__ == '__main__':
    optimizer.probe(params={"lr":1.3e-4, "result_dir":51}, lazy=False)#, "inter_dim_i":1024, "bneck_i":512},lazy=False)

    #Tries to find parameters by Bayesian Optimization (Gaussian Process Regressor)
    #optimizer.probe(params=[5e-2,4e-2],lazy=True)
    #optimizer.maximize(n_iter=10, init_points = 2)

    print("Final result:", optimizer.max)
    print("All tested hyperparameters:", optimizer.res)