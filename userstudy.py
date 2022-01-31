import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from utils import MyTabNetClassifier

from ares import AReS
from clustering import Clustering
from cet import CounterfactualExplanationTree
from ce import ActionExtractor
from utils import DatasetHelper, submodular_picking


def instance_to_markdown(x, feature_names, feature_types, feature_categories):
    s = '| | Feature | Value |\n'
    s += '| --- | --- | ---: |\n'    
    i = 1
    for d, x_d in enumerate(x):
        if(d not in sum(feature_categories, [])):
            if(feature_types[d]=='C'):
                s += '| {} | {} | {.4f} |\n'.format(i, feature_names[d], x_d) 
            if(feature_types[d]=='B'):
                s += '| {} | {} | {} |\n'.format(i, feature_names[d], bool(x_d)) 
            else:
                s += '| {} | {} | {} |\n'.format(i, feature_names[d], int(x_d))
        else:
            if(x_d!=1): continue
            prv, nxt = feature_names[d].split(':')
            s += '| {} | {} | {} |\n'.format(i, prv, nxt)
        i += 1
    return s

def actions_to_markdown(action_dicts, x, feature_names, feature_types, feature_categories):
    feature_categories_inv = []
    for d in range(len(feature_names)):
        g = -1
        if(feature_types[d]=='B'):
            for i, cat in enumerate(feature_categories):
                if(d in cat): 
                    g = i
                    break
        feature_categories_inv.append(g)            

    s = '| | HowToChange |\n'
    s += '| --- | :--- |\n'
    for k, action_dict in enumerate(action_dicts):
        a = action_dict['action']
        acc = action_dict['acc']; cost = action_dict['cost'];
        s += '| Action {} | '.format(k+1 if len(action_dicts)>1 else '')
        for d in np.where(abs(a)>1e-8)[0]:
            g = feature_categories_inv[d]
            if(g==-1):
                if(feature_types[d]=='C'):
                    s += '{}: {:.4f} -> {:.4f} ({:+.4f}) <br>'.format(feature_names[d], x[d], x[d]+a[d], a[d])
                elif(feature_types[d]=='B'):
                    if(a[d]==-1):
                        s += '{}: True -> False <br> '.format(feature_names[d], a[d])
                    else:
                        s += '{}: False -> True <br> '.format(feature_names[d], a[d])
                else:
                    s += '{}: {} -> {} ({:+}) <br>'.format(feature_names[d], x[d].astype(int), x[d].astype(int)+a[d].astype(int), a[d].astype(int))
            else:
                if(a[d]==-1): continue
                cat_name, nxt = feature_names[d].split(':')
                cat = feature_categories[g]
                prv = feature_names[cat[np.where(a[cat]==-1)[0][0]]].split(':')[1]
                s += '{}: \"{}\" -> \"{}\" <br> '.format(cat_name, prv, nxt)
        s += '(Acc: {} / Cost: {:.3}) |\n'.format(acc, cost)

    return s

def _instances_to_markdown(X, feature_names, feature_types, feature_categories):
    s = '| Feature ' 
    for n in range(X.shape[0]): s += '| Instance {} '.format(n+1)
    s += '|\n| --: ' + '| --: '*X.shape[0] + '|\n'
    for d, X_d in enumerate(X.T):
        if(d in sum(feature_categories, [])):
            s += '| {} '.fomrat(feature_names[d])
            for x_d in X_d:
                if(feature_types[d]=='C'):
                    s += '| {.4f} '.format(x_d) 
                if(feature_types[d]=='B'):
                    s += '| {} '.format(bool(x_d)) 
                else:
                    s += '| {} '.format(int(x_d))
            s += '|\n'
        else:
            if(x_d!=1): continue
            prv, nxt = feature_names[d].split(':')
            s += '| {} | {} |\n'.format(prv, nxt)
    return s


def demo(dataset='t', model='X'):
    np.random.seed(0)
    LAMBDA = 0.01
    GAMMA = 1.0

    D = DatasetHelper(dataset=dataset, feature_prefix_index=False)
    print('# Demonstration:', D.dataset_fullname)
    # for d in range(D.n_features): print('\t* x_{:<2}: {} ({}{})'.format(d+1, D.feature_names[d], D.feature_types[d], ':'+D.feature_constraints[d] if D.feature_constraints[d]!='' else ''))
    print(D.to_markdown())

    if(model=='L'):
        print('* Classifier: LogisticRegression')
        mdl = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
        print('\t* C: {}'.format(mdl.C)); print('\t* penalty: {}'.format(mdl.penalty));
    elif(model=='X'):
        print('* Classifier: LightGBM')
        mdl = LGBMClassifier(n_estimators=50, num_leaves=8)
        print('\t* n_estimators: {}'.format(mdl.n_estimators)); print('\t* num_leaves: {}'.format(mdl.num_leaves));
    elif(model=='T'):
        print('* Classifier: TabNet')
        mdl = MyTabNetClassifier(D.feature_types, verbose=0)

    X_tr, X_ts, y_tr, y_ts = D.train_test_split()
    mdl = mdl.fit(X_tr, y_tr, X_vl=X_ts, y_vl=y_ts) if model=='T' else mdl.fit(X_tr, y_tr)
    X = X_tr[mdl.predict(X_tr)==1]; X_vl = X_ts[mdl.predict(X_ts)==1];
    print('\t* train score: ', mdl.score(X_tr, y_tr)); print('\t* train denied: ', X.shape[0]); 
    print('\t* test score: ', mdl.score(X_ts, y_ts)); print('\t* test denied: ', X_vl.shape[0]); print();


    print('## Submodular Pick')
    ce = ActionExtractor(mdl, X_tr, Y=y_tr, max_candidates=100,
                         feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                         feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels,
                         lime_approximation=(model!='L'), n_samples=10000, alpha=1.0)
    action_candidates = []
    for n, x in enumerate(X):
        action_dict = ce.extract(x.reshape(1,-1), max_change_num=2, cost_type='MPS', tradeoff_parameter=100.0)
        if(action_dict['active']): action_candidates += [ action_dict ]
    indices = submodular_picking([a['action'] for a in action_candidates], budget=4)
    sp_actions = [(action_candidates[i]['instance'][0], {'action': action_candidates[i]['action'], 'acc': action_candidates[i]['active'][0], 'cost': action_candidates[i]['cost'][0]}) for i in indices]
    for i, (x, action_dict) in enumerate(sp_actions): 
        print('### Instance {}'.format(i+1))
        print(instance_to_markdown(x, D.feature_names, D.feature_types, D.feature_categories))
        print(actions_to_markdown([action_dict], x, D.feature_names, D.feature_types, D.feature_categories))


    print('## Clustering')
    clustering = Clustering(mdl, X_tr, Y=y_tr, n_clusters=4, print_centers=True, lime_approximation=(model!='L'),
                            feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                            feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    clustering = clustering.fit(X, max_change_num=2, cost_type='MPS', gamma=100.0, time_limit=180)
    print('* Parameters:'); print('\t* clusters: {}'.format(clustering.n_clusters_)); print('\t* gamma: {}'.format(clustering.gamma_)); 
    print('\t* LIME approximation:', clustering.lime_approximation_); print('\t* Time[s]:', clustering.time_); print();
    print('### Learned Clustering')
    print(clustering.to_markdown())

    print('## Actionable Recourse Summary')
    ares = AReS(mdl, X_tr, max_rule=4, max_rule_length=4, minimum_support=0.05, discretization_bins=10, print_objective=False,
                feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    ares = ares.fit(X, max_change_num=2, cost_type='MPS', lambda_acc=1.0, lambda_cov=1.0, lambda_cst=1.0)
    print('* Parameters:')
    print('\t* lambda_acc: {}'.format(ares.lambda_acc)); print('\t* lambda_cov: {}'.format(ares.lambda_cov)); print('\t* lambda_cst: {}'.format(ares.lambda_cst));
    print('\t* minimum support: {}'.format(ares.rule_miner_.minsup_)); print('\t* discretization bins: {}'.format(ares.rule_miner_.fd_.bins)); print('\t* pre-processing time[s]: {}'.format(ares.preprocess_time_)); 
    print('\t* max rule: {}'.format(ares.max_rule_)); print('\t* max rule length: {}'.format(ares.max_rule_length_)); print('\t* Time[s]:', ares.time_); 
    print('\t* uncover test: {}'.format(ares.uncover(X_vl))); print('\t* conflict: {}'.format(ares.conflict(X_vl))); print();
    print('### Learned AReS')
    print(ares.to_markdown())

    print('## Counterfactual Explanation Tree')
    cet = CounterfactualExplanationTree(mdl, X_tr, y_tr, max_iteration=1000, lime_approximation=(model!='L'),
                                        feature_names=D.feature_names, feature_types=D.feature_types, feature_categories=D.feature_categories, 
                                        feature_constraints=D.feature_constraints, target_name=D.target_name, target_labels=D.target_labels)
    cet = cet.fit(X, max_change_num=2, cost_type='MPS', C=LAMBDA, gamma=GAMMA, max_leaf_size=4, time_limit=180)
    print('* Parameters:'); print('\t* lambda: {}'.format(cet.lambda_)); print('\t* gamma: {}'.format(cet.gamma_)); print('\t* max_iteration: {}'.format(cet.max_iteration_));
    print('\t* leaf size bound:', cet.leaf_size_bound_); print('\t* leaf size:', cet.n_leaves_); print('\t* LIME approximation:', cet.lime_approximation_); print('\t* Time[s]:', cet.time_); print();
    print('### Learned CET')
    cet.print_tree()

    print('## Question')
    i_candidates = []; optimal_actions = [];
    for n, x in enumerate(X_vl):
        action_dict = ce.extract(x.reshape(1,-1), max_change_num=2, cost_type='MPS', tradeoff_parameter=100.0)
        optimal_actions += [ action_dict ]
        if(action_dict['active']): i_candidates += [ n ]

    for n in i_candidates:
        print('### n = {}'.format(n))
        x = X_vl[n]
        actions_dicts = []

        action_dict = optimal_actions[n]
        actions_dicts.append({'action': action_dict['action'], 'acc': action_dict['active'][0], 'cost': action_dict['cost'][0]})

        action = clustering.predict_random(x.reshape(1,-1))[0]
        cost = clustering.cost(x.reshape(1,-1), cost_type='MPS', random=True)
        active = bool(1-clustering.loss(x.reshape(1,-1), random=True))
        actions_dicts.append({'action': action, 'acc': active, 'cost': cost})

        action = clustering.predict(x.reshape(1,-1))[0]
        cost = clustering.cost(x.reshape(1,-1), cost_type='MPS')
        active = bool(1-clustering.loss(x.reshape(1,-1)))
        actions_dicts.append({'action': action, 'acc': active, 'cost': cost})

        action = ares.predict(x.reshape(1,-1))[0]
        cost = ares.cost(x.reshape(1,-1), cost_type='MPS')
        active = bool(1-ares.loss(x.reshape(1,-1)))
        actions_dicts.append({'action': action, 'acc': active, 'cost': cost})

        action = cet.predict(x.reshape(1,-1))[0]
        cost = cet.cost(x.reshape(1,-1), cost_type='MPS')
        active = bool(1-cet.loss(x.reshape(1,-1)))
        actions_dicts.append({'action': action, 'acc': active, 'cost': cost})

        print(instance_to_markdown(x, D.feature_names, D.feature_types, D.feature_categories))
        print(actions_to_markdown(actions_dicts, x, D.feature_names, D.feature_types, D.feature_categories))



if(__name__ == '__main__'):
    demo(model='X')