import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from utils import MyTabNetClassifier, DatasetHelper
from cet import CounterfactualExplanationTree


def convergence(dataset='g', model='L', params=(0.01, 1.0)):
    np.random.seed(0)
    l,g = params

    print('# Convergence Analysis (lambda = {} / gamma = {})'.format(l, g))
    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    print('* Dataset:', D.dataset_fullname)
    for d in range(D.n_features): print('\t* x_{:<2}: {} ({}{})'.format(d+1, D.feature_names[d], D.feature_types[d], ':'+D.feature_constraints[d] if D.feature_constraints[d]!='' else ''))

    if(model=='L'):
        print('* Classifier: LogisticRegression')
        mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
        print('\t* C: {}'.format(mdl.C)); print('\t* penalty: {}'.format(mdl.penalty));
    elif(model=='X'):
        print('* Classifier: LightGBM')
        mdl = LGBMClassifier(n_estimators=100, num_leaves=16, class_weight='balanced')
        print('\t* n_estimators: {}'.format(mdl.n_estimators)); print('\t* num_leaves: {}'.format(mdl.num_leaves));
    elif(model=='T'):
        print('* Classifier: TabNet')
        mdl = MyTabNetClassifier(D.feature_types, verbose=0, class_weight='balanced')

    mdl = mdl.fit(D.X, D.y, X_vl=D.X, y_vl=D.y) if model=='T' else mdl.fit(D.X, D.y)
    X = D.X[mdl.predict(D.X)==1]
    print('\t* train score: ', mdl.score(D.X, D.y)); print('\t* train denied: ', X.shape[0]); print();

    cet = CounterfactualExplanationTree(mdl, D.X, D.y, max_iteration=MAX_ITERATION, lime_approximation=(model!='L'),
                                        feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    cet = cet.fit(X, max_change_num=3, cost_type=COST_TYPE, C=l, gamma=g, time_limit=60, verbose=True)
    print('## Learned CET')
    cet.print_tree()
    pd.DataFrame(cet.objs_).to_csv('./res/convergence/{}/cet_{}_objective_{}_{}.csv'.format(model, D.dataset_name, l, g), index=False)



MAX_ITERATION = 10000
COST_TYPE = 'MPS'

if(__name__ == '__main__'):

    convergence(dataset='g', model='L', params=(0.01, 0.75))
    convergence(dataset='g', model='L', params=(0.03, 0.75))
    convergence(dataset='g', model='L', params=(0.05, 0.75))

    convergence(dataset='g', model='L', params=(0.01, 1.0))
    convergence(dataset='g', model='L', params=(0.03, 1.0))
    convergence(dataset='g', model='L', params=(0.05, 1.0))

    convergence(dataset='g', model='L', params=(0.01, 1.25))
    convergence(dataset='g', model='L', params=(0.03, 1.25))
    convergence(dataset='g', model='L', params=(0.05, 1.25))


    convergence(dataset='i', model='L', params=(0.01, 0.75))
    convergence(dataset='i', model='L', params=(0.03, 0.75))
    convergence(dataset='i', model='L', params=(0.05, 0.75))

    convergence(dataset='i', model='L', params=(0.01, 1.0))
    convergence(dataset='i', model='L', params=(0.03, 1.0))
    convergence(dataset='i', model='L', params=(0.05, 1.0))

    convergence(dataset='i', model='L', params=(0.01, 1.25))
    convergence(dataset='i', model='L', params=(0.03, 1.25))
    convergence(dataset='i', model='L', params=(0.05, 1.25))


