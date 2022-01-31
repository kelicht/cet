import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from utils import MyTabNetClassifier, DatasetHelper
from sklearn.model_selection import StratifiedKFold

from ares import AReS
from clustering import Clustering
from cet import CounterfactualExplanationTree


def compare_cv(dataset='g', model='L', n_splits=5):
    np.random.seed(0)
    LAMBDA = HARES_PARAMS[dataset][model]['lambda']
    GAMMA = HARES_PARAMS[dataset][model]['gamma']

    print('# {}-fold CV Performance Comparison'.format(n_splits))
    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    print('* Dataset:', D.dataset_fullname)
    for d in range(D.n_features): print('\t* x_{:<2}: {}'.format(d+1, D.feature_names[d]))

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

    dict_clustering = {'cost_train':[], 'loss_train':[], 'obj_train':[], 'cost_test':[], 'loss_test':[], 'obj_test':[], 'time':[], 'n_actions':[]}
    dict_ares = {'cost_train':[], 'loss_train':[], 'obj_train':[], 'cost_test':[], 'loss_test':[], 'obj_test':[], 'time':[], 'n_actions':[], 'uncover_train':[], 'conflict_train':[], 'uncover_test':[], 'conflict_test':[]}
    dict_cet = {'cost_train':[], 'loss_train':[], 'obj_train':[], 'cost_test':[], 'loss_test':[], 'obj_test':[], 'time':[], 'n_actions':[]}
    k = 1; 
    for train, test in StratifiedKFold(n_splits=n_splits).split(D.X, D.y):
        print(); print('## Fold: k = {}'.format(k)); 
        X_tr, X_ts, y_tr, y_ts = D.X[train], D.X[test], D.y[train], D.y[test]
        mdl = mdl.fit(X_tr, y_tr, X_vl=X_ts, y_vl=y_ts) if model=='T' else mdl.fit(X_tr, y_tr)
        X = X_tr[mdl.predict(X_tr)==1]; X_vl = X_ts[mdl.predict(X_ts)==1]; 
        print('* labbel: ', y_tr.mean()); print('* train score: ', mdl.score(X_tr, y_tr)); print('* test score: ', mdl.score(X_ts, y_ts)); print('* train denied: ', X.shape[0]); print('* test denied: ', X_vl.shape[0]); print();
        if(X.shape[0]<10 or X_vl.shape[0]<1): continue


        print('### Clusterwise Actionable Recourse Summary')
        clustering = Clustering(mdl, X_tr, Y=y_tr, n_clusters=8, print_centers=True, lime_approximation=(model!='L'),
                        feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
        clustering = clustering.fit(X, max_change_num=3, cost_type=COST_TYPE, gamma=GAMMA, time_limit=180)
        print('* Parameters:'); print('\t* clusters: {}'.format(clustering.n_clusters_)); print('\t* gamma: {}'.format(clustering.gamma_)); 
        print('\t* LIME approximation:', clustering.lime_approximation_); print('\t* Time[s]:', clustering.time_); print();
        print('#### Learned Clusterwise AReS')
        print(clustering)
        
        print('#### Score (Clusterwise AReS):')
        clustering_cost = clustering.cost(X, cost_type=COST_TYPE); clustering_loss = clustering.loss(X);
        dict_clustering['cost_train'].append(clustering_cost); dict_clustering['loss_train'].append(clustering_loss); dict_clustering['obj_train'].append(clustering_cost + GAMMA * clustering_loss);
        print('- Train:'); print('\t- cost: {}'.format(clustering_cost, cost_type=COST_TYPE)); print('\t- loss: {}'.format(clustering_loss)); print('\t- obj.: {}'.format(clustering_cost + GAMMA * clustering_loss));
        clustering_cost = clustering.cost(X_vl, cost_type=COST_TYPE); clustering_loss = clustering.loss(X_vl);
        dict_clustering['cost_test'].append(clustering_cost); dict_clustering['loss_test'].append(clustering_loss); dict_clustering['obj_test'].append(clustering_cost + GAMMA * clustering_loss);
        print('- Test:'); print('\t- cost: {}'.format(clustering_cost, cost_type=COST_TYPE)); print('\t- loss: {}'.format(clustering_loss)); print('\t- obj.: {}'.format(clustering_cost + GAMMA * clustering_loss));
        dict_clustering['time'].append(clustering.time_); dict_clustering['n_actions'].append(clustering.n_clusters_);
        print()


        print('### Actionable Recourse Summary')
        ares = AReS(mdl, X_tr, max_rule=8, max_rule_length=4, minimum_support=MINSUP[dataset], discretization_bins=10,
                    feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                    feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
        ares = ares.fit(X, max_change_num=3, cost_type=COST_TYPE, lambda_acc=ARES_PARAMS[dataset][model]['acc'], lambda_cov=ARES_PARAMS[dataset][model]['cov'], lambda_cst=ARES_PARAMS[dataset][model]['cst'])
        print('* Parameters:'); print('\t* lambda_acc: {}'.format(ares.lambda_acc)); print('\t* lambda_cov: {}'.format(ares.lambda_cov)); print('\t* lambda_cst: {}'.format(ares.lambda_cst));
        print('\t* minimum support: {}'.format(ares.rule_miner_.minsup_)); print('\t* discretization bins: {}'.format(ares.rule_miner_.fd_.bins)); print('\t* pre-processing time[s]: {}'.format(ares.preprocess_time_)); 
        print('\t* rule candidates: {}'.format(ares.P_)); print('\t* max rule: {}'.format(ares.max_rule_)); print('\t* max rule length: {}'.format(ares.max_rule_length_)); print('\t* Time[s]:', ares.time_); print()
        print('#### Learned AReS')
        print(ares)

        print('#### Score (AReS):')
        ares_cost = ares.cost(X, cost_type=COST_TYPE); ares_loss = ares.loss(X);
        dict_ares['cost_train'].append(ares_cost); dict_ares['loss_train'].append(ares_loss); dict_ares['obj_train'].append(ares_cost + GAMMA * ares_loss); 
        print('- Train:'); print('\t- cost: {}'.format(ares_cost)); print('\t- loss: {}'.format(ares_loss)); print('\t- obj.: {}'.format(ares_cost + GAMMA * ares_loss));
        ares_uncov = ares.uncover(X); ares_conf = ares.conflict(X);
        dict_ares['uncover_train'].append(ares_uncov); dict_ares['conflict_train'].append(ares_conf);
        print('\t- uncover: {}'.format(ares_uncov)); print('\t- conflict: {}'.format(ares_conf)); 
        ares_cost = ares.cost(X_vl, cost_type=COST_TYPE); ares_loss = ares.loss(X_vl);
        dict_ares['cost_test'].append(ares_cost); dict_ares['loss_test'].append(ares_loss); dict_ares['obj_test'].append(ares_cost + GAMMA * ares_loss); 
        print('- Test:'); print('\t- cost: {}'.format(ares_cost)); print('\t- loss: {}'.format(ares_loss)); print('\t- obj.: {}'.format(ares_cost + GAMMA * ares_loss));
        ares_uncov = ares.uncover(X_vl); ares_conf = ares.conflict(X_vl);
        dict_ares['uncover_test'].append(ares_uncov); dict_ares['conflict_test'].append(ares_conf);
        print('\t- uncover: {}'.format(ares_uncov)); print('\t- conflict: {}'.format(ares_conf)); 
        dict_ares['time'].append(ares.time_); dict_ares['n_actions'].append(ares.max_rule_);
        print()


        print('### Counterfactual Explanation Tree')
        cet = CounterfactualExplanationTree(mdl, X_tr, y_tr, max_iteration=MAX_ITERATION, lime_approximation=(model!='L'),
                                            feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                            feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
        cet = cet.fit(X, max_change_num=3, cost_type=COST_TYPE, C=LAMBDA, gamma=GAMMA, time_limit=180)
        print('* Parameters:'); print('\t* lambda: {}'.format(cet.lambda_)); print('\t* gamma: {}'.format(cet.gamma_)); print('\t* max_iteration: {}'.format(cet.max_iteration_));
        print('\t* leaf size bound:', cet.leaf_size_bound_); print('\t* leaf size:', cet.n_leaves_); print('\t* LIME approximation:', cet.lime_approximation_); print('\t* Time[s]:', cet.time_); print();
        print('#### Learned CET')
        cet.print_tree()

        print('#### Score (CET):')
        cet_cost = cet.cost(X, cost_type=COST_TYPE); cet_loss = cet.loss(X);
        dict_cet['cost_train'].append(cet_cost); dict_cet['loss_train'].append(cet_loss); dict_cet['obj_train'].append(cet_cost + GAMMA * cet_loss);
        print('- Train:'); print('\t- cost: {}'.format(cet_cost)); print('\t- loss: {}'.format(cet_loss)); print('\t- obj.: {}'.format(cet_cost + GAMMA * cet_loss));
        cet_cost = cet.cost(X_vl, cost_type=COST_TYPE); cet_loss = cet.loss(X_vl);
        dict_cet['cost_test'].append(cet_cost); dict_cet['loss_test'].append(cet_loss); dict_cet['obj_test'].append(cet_cost + GAMMA * cet_loss);
        print('- Test:'); print('\t- cost: {}'.format(cet_cost)); print('\t- loss: {}'.format(cet_loss)); print('\t- obj.: {}'.format(cet_cost + GAMMA * cet_loss));
        dict_cet['time'].append(cet.time_); dict_cet['n_actions'].append(cet.n_leaves_);
        print()
        pd.DataFrame(cet.objs_).to_csv('./res/compare/{}/convergence/cet_{}_objective_{}_{}_{}.csv'.format(model, D.dataset_name, LAMBDA, GAMMA, k), index=False)


        k += 1


    print('## Overall Score:')
    for title, key1 in zip(['Train Performance', 'Test Performance'], ['train', 'test']):
        print('### ', title)
        for key2 in ['cost', 'loss', 'obj']:
            print('- Average {}'.format(key2))
            print('\t- Clustering:', np.mean(dict_clustering['{}_{}'.format(key2, key1)])); 
            print('\t- AReS      :', np.mean(dict_ares['{}_{}'.format(key2, key1)])); 
            print('\t- CET       :', np.mean(dict_cet['{}_{}'.format(key2, key1)]));
    print('### Runing Time[s]')
    print('- Clustering:', np.mean(dict_clustering['time'])); 
    print('- AReS      :', np.mean(dict_ares['time'])); 
    print('- CET       :', np.mean(dict_cet['time']));
    print('### Num. of Actions')
    print('- Clustering:', np.mean(dict_clustering['n_actions'])); 
    print('- AReS      :', np.mean(dict_ares['n_actions'])); 
    print('- CET       :', np.mean(dict_cet['n_actions']));
    print(); print();
    pd.DataFrame(dict_clustering).to_csv('./res/compare/{}/clustering_{}_{}_{}.csv'.format(model, D.dataset_name, LAMBDA, GAMMA), index=False)
    pd.DataFrame(dict_ares).to_csv('./res/compare/{}/ares_{}_{}_{}.csv'.format(model, D.dataset_name, LAMBDA, GAMMA), index=False)
    pd.DataFrame(dict_cet).to_csv('./res/compare/{}/cet_{}_{}_{}.csv'.format(model, D.dataset_name, LAMBDA, GAMMA), index=False)



MAX_ITERATION = 3000
COST_TYPE = 'MPS'
MINSUP = {'g':0.05, 'i':0.05}
ARES_PARAMS = {'g':
                    {'T': {'acc':1.0, 'cov':1.0, 'cst':0.01}, 
                     'X': {'acc':10.0, 'cov':1.0, 'cst':10.0},
                     'L': {'acc':10.0, 'cov':1.0, 'cst':100.0},}, 
               'i':
                    {'T': {'acc':1.0, 'cov':1.0, 'cst':100.0}, 
                     'X': {'acc':1.0, 'cov':1.0, 'cst':0.01},
                     'L': {'acc':1.0, 'cov':1.0, 'cst':10.0},}
                }
HARES_PARAMS = {'g':
                    {'T': {'lambda':0.02, 'gamma':1.0}, 
                     'X': {'lambda':0.02, 'gamma':1.0},
                     'L': {'lambda':0.02, 'gamma':1.0},}, 
                'i':
                    {'T': {'lambda':0.02, 'gamma':1.0}, 
                     'X': {'lambda':0.02, 'gamma':1.0},
                     'L': {'lambda':0.02, 'gamma':1.0},}
                }

if(__name__ == '__main__'):

    compare_cv(dataset='i', model='X', n_splits=10)
    compare_cv(dataset='g', model='X', n_splits=10)
    compare_cv(dataset='i', model='T', n_splits=10)
    compare_cv(dataset='g', model='T', n_splits=10)

