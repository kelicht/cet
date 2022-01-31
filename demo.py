import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from utils import MyTabNetClassifier, DatasetHelper
from cet import CounterfactualExplanationTree


def demo_cet(dataset='t', model='X'):
    np.random.seed(0)
    LAMBDA = 0.01
    GAMMA = 1.0

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    print('# CET Demonstration')
    print('* Dataset:', D.dataset_fullname)
    for d in range(D.n_features): print('\t* x_{:<2}: {} ({}{})'.format(d+1, D.feature_names[d], D.feature_types[d], ':'+D.feature_constraints[d] if D.feature_constraints[d]!='' else ''))
    # print(D.to_markdown())

    if(model=='L'):
        print('* Classifier: LogisticRegression')
        mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
        print('\t* C: {}'.format(mdl.C)); print('\t* penalty: {}'.format(mdl.penalty));
    elif(model=='X'):
        print('* Classifier: LightGBM')
        mdl = LGBMClassifier(n_estimators=100, num_leaves=16)
        print('\t* n_estimators: {}'.format(mdl.n_estimators)); print('\t* num_leaves: {}'.format(mdl.num_leaves));
    elif(model=='T'):
        print('* Classifier: TabNet')
        mdl = MyTabNetClassifier(D.feature_types, verbose=0)

    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr, X_vl=X_ts, y_vl=y_ts) if model=='T' else mdl.fit(X_tr, y_tr)
    X = X_tr[mdl.predict(X_tr)==1]; X_vl = X_ts[mdl.predict(X_ts)==1];
    print('\t* train score: ', mdl.score(X_tr, y_tr)); print('\t* train denied: ', X.shape[0]); 
    print('\t* test score: ', mdl.score(X_ts, y_ts)); print('\t* test denied: ', X_vl.shape[0]); print();

    print('## Counterfactual Explanation Tree (CET)')
    cet = CounterfactualExplanationTree(mdl, X_tr, y_tr, max_iteration=100, lime_approximation=(model!='L'),
                                        feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    cet = cet.fit(X_vl, max_change_num=1, cost_type='MPS', C=LAMBDA, gamma=GAMMA, max_leaf_size=3, time_limit=180)
    print('* Parameters:'); print('\t* lambda: {}'.format(cet.lambda_)); print('\t* gamma: {}'.format(cet.gamma_)); print('\t* max_iteration: {}'.format(cet.max_iteration_));
    print('\t* leaf size bound:', cet.leaf_size_bound_); print('\t* LIME approximation:', cet.lime_approximation_); print('\t* leaf size:', cet.n_leaves_); print('\t* Time[s]:', cet.time_); print();
    print('### Learned CET')
    cet.print_tree()


if(__name__ == '__main__'):
    demo_cet(model='X')