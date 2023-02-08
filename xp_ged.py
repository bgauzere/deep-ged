'''
Perform an experience on predicting molecule property by learning ged edit costs
'''
import numpy as np
from main import run, parse_arguments_main

if __name__ == "__main__":
    args = parse_arguments_main()
    lis_train = []
    lis_test = []
    for i in range(10):
        perf_app, perf_test = run(args)
        lis_train.append(perf_app)
        lis_test.append(perf_test)
        print(f"xp {i} : {perf_app},{perf_test}")
    print("Perf in train : ", np.mean(lis_train), "±", np.std(lis_train))
    print("Perf in test : ", np.mean(lis_test), "±", np.std(lis_test))
