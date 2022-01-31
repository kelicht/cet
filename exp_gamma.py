import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from utils import MyTabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KDTree

from ce import ActionExtractor
from utils import DatasetHelper, synthetic_dataset


def sensitivity(dataset='g', model='L', N=10, M=100, gammas=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5], choice='neighbor'):
    np.random.seed(0)

    print('# Hold-Out gamma-Sensitivity Analysis')
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

    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr, X_vl=X_ts, y_vl=y_ts) if model=='T' else mdl.fit(X_tr, y_tr)
    X = X_tr[mdl.predict(X_tr)==1]; X_vl = X_ts[mdl.predict(X_ts)==1];
    print('\t* train score: ', mdl.score(X_tr, y_tr)); print('\t* train denied: ', X.shape[0]); 
    print('\t* test score: ', mdl.score(X_ts, y_ts)); print('\t* test denied: ', X_vl.shape[0]); print();

    ce = ActionExtractor(mdl, X_tr, Y=y_tr, max_candidates=100,
                         feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                         feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels,
                         lime_approximation=True, n_samples=10000, alpha=1.0)

    kdtree = KDTree(X_vl)
    res = {'iteration':[], 'gamma':[], 'cost':[], 'loss':[], 'obj':[]}
    for m in range(M):
        print('## Iteration = {}'.format(m+1))
        if(choice=='neighbor'):
            i = np.random.choice(range(X_vl.shape[0]))
            _, indices = kdtree.query(X_vl[i].reshape(1,-1), k=N)
            X_m = X_vl[indices[0]]
        else:
            X_m = X_vl[np.random.choice(range(X_vl.shape[0]), N, replace=False)]
        for g in gammas:
            print('- gamma = {}'.format(g))
            action = ce.extract(X_m, max_change_num=3, cost_type=COST_TYPE, tradeoff_parameter=g)
            c = action['cost'].mean(); l = (1-action['active']).mean(); o = action['objective']/N
            print('\t- Cost:', c); print('\t- Loss:', l); print('\t- Obj.:', o);
            res['iteration'].append(m); res['gamma'].append(g);
            res['cost'].append(c); res['loss'].append(l); res['obj'].append(o)
        print()
    print(); print();
    pd.DataFrame(res).to_csv('./res/gamma/{}/sensitivity_{}.csv'.format(model, D.dataset_name), index=False)





COST_TYPE = 'MPS'

if(__name__ == '__main__'):

    sensitivity(dataset='i', model='L', N=10, M=100, gammas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    sensitivity(dataset='g', model='L', N=10, M=100, gammas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # sensitivity(dataset='i', model='L', N=10, M=100, gammas=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    # sensitivity(dataset='g', model='L', N=10, M=100, gammas=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
