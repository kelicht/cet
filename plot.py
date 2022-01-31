import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


MODEL = {'T': 'TabNet', 'X':'LightGBM'}
DATASET = {'i':'attrition', 'g':'german'}
DATASETNAME = {'i':'Attrition', 'g':'German'}
METHODS = {'clustering':'Clustering', 'ares':'AReS', 'cet':'CET'}
MARKER = {'clustering':'v', 'ares':'s', 'cet':'o'}


def latex_compare(model='T', datasets=['i','g'], l=0.02, g=1.0):
    filenames = [ ['./res/compare/{}/{}_{}_{}_{}.csv'.format(model, method, DATASET[dataset], l, g) for method in METHODS.keys()] for dataset in datasets ]
    for i, dataset in enumerate(datasets):
        print(DATASETNAME[dataset])
        for j, method in enumerate(METHODS.keys()):
            df = pd.read_csv(filenames[i][j])
            s = '& {} '.format(METHODS[method])
            for key1 in ['train', 'test']:
                for key2 in ['cost', 'loss', 'obj']:
                    s += '& {:.3} $\pm$ {:.2} '.format(df['{}_{}'.format(key2, key1)].mean().round(3), df['{}_{}'.format(key2, key1)].std().round(2))
            s += '\\\\'
            print(s)

# latex_compare(model='X')
# latex_compare(model='T')


def latex_compare_time(models=['X','T'], datasets=['i','g'], l=0.02, g=1.0):
    filenames = [ [ ['./res/compare/{}/{}_{}_{}_{}.csv'.format(model, method, DATASET[dataset], l, g) for method in METHODS.keys()] for dataset in datasets ] for model in models]
    average = [0, 0, 0]
    for k, model in enumerate(models):
        for i, dataset in enumerate(datasets):
            s = '& {} '.format(DATASETNAME[dataset])
            for j, method in enumerate(METHODS.keys()):
                df = pd.read_csv(filenames[k][i][j])
                s += '& {:.6} $\pm$ {:.6} '.format(df['time'].mean().round(3), df['time'].std().round(2))
                average[j] += df['time'].mean() / 4
            s += '\\\\'
            print(s)
    print('& {:.6} & {:.6} & {:.6} \\\\'.format(average[0], average[1], average[2]))

# latex_compare_time()


def plot_sens_comp(model='L', datasets=['i','g'], gamma=1.0):
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(8,4))
    for i, dataset in enumerate(datasets):
        plt.subplot(2, 2, (i+1))
        plt.title(DATASETNAME[dataset], fontsize=16)
        for method in METHODS.keys():
            df = pd.read_csv('./res/complexity/{}/{}_{}_{}.csv'.format(model, method, DATASET[dataset], gamma))
            plt.plot(df['n_actions'], df['cost_test'], marker=MARKER[method], label='{}'.format(METHODS[method]))
        plt.xlabel(r'\#Actions', fontsize=14); plt.ylabel('Cost (test)', fontsize=14); plt.xticks([4,8,12,16,20], fontsize=12); plt.yticks(fontsize=12); plt.tight_layout(); 
        # if(i==0): plt.legend()
        plt.subplot(2, 2, i+3)
        for method in METHODS.keys():
            df = pd.read_csv('./res/complexity/{}/{}_{}_{}.csv'.format(model, method, DATASET[dataset], gamma))
            plt.plot(df['n_actions'], df['loss_test'], marker=MARKER[method], label='{}'.format(METHODS[method]))
        plt.xlabel(r'\#Actions', fontsize=14); plt.ylabel('Loss (test)', fontsize=14); plt.xticks([4,8,12,16,20], fontsize=12); plt.yticks(fontsize=12); plt.tight_layout(); 
        if(i==1): plt.legend(fontsize=12)
    plt.savefig('./res/complexity/{}/tradeoff.png'.format(model), bbox_inches='tight', pad_inches=0.05)
    plt.savefig('./res/complexity/{}/tradeoff.pdf'.format(model), bbox_inches='tight', pad_inches=0.05)
    plt.clf()

# plot_sens_comp(model='L', datasets=['i', 'g'], gamma=1.0)

def plot_sens_comp_pareto_frontier(model='L', datasets=['i', 'g'], gamma=1.0):
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(8,2.75))

    for i, dataset in enumerate(datasets):
        plt.subplot(1, 2, (i+1))
        plt.title(DATASETNAME[dataset], fontsize=16)
        for method in METHODS.keys():
            df = pd.read_csv('./res/complexity/{}/{}_{}_{}.csv'.format(model, method, DATASET[dataset], gamma))
            plt.scatter(df['cost_test'], df['loss_test'], marker=MARKER[method], label='{}'.format(METHODS[method]), s=75, linewidth=0.5, edgecolor='black')
            for n, (x,y) in zip(df['n_actions'], zip(df['cost_test'], df['loss_test'])):
                text_fontsize=12
                offset = 0.004
                if(dataset=='i'):
                    if(method=='ares' and n==4):
                        plt.text(x-offset, y-0.002, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='ares' and n==8):
                        plt.text(x-offset, y+0.06, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='ares' and n==12):
                        plt.text(x-offset, y+0.03, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='ares' and n==16):
                        plt.text(x-offset, y, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='ares' and n==20):
                        plt.text(x-offset, y-0.03, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='cet' and n==4):
                        plt.text(x+offset, y-offset, "{}".format(n), horizontalalignment='left', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='cet' and n==11):
                        plt.text(x-offset, y+offset, "{}".format(n), horizontalalignment='right', verticalalignment='bottom', fontsize=text_fontsize)
                    elif(method=='cet' and n==16):
                        plt.text(x+offset, y-offset, "{}".format(n), horizontalalignment='left', verticalalignment='top', fontsize=text_fontsize)
                    else:
                        plt.text(x-offset, y-offset, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                if(dataset=='g'):
                    if(method=='ares' and n==4):
                        plt.text(x-offset, y+offset, "{}".format(n), horizontalalignment='right', verticalalignment='bottom', fontsize=text_fontsize)
                    elif(method=='ares' and n==8):
                        plt.text(x+offset, y-offset, "{}".format(n), horizontalalignment='left', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='ares' and n==12):
                        plt.text(x-offset, y+0.05, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
                    elif(method=='cet' and n==4):
                        plt.text(x-offset, y+offset, "{}".format(n), horizontalalignment='right', verticalalignment='bottom', fontsize=text_fontsize)
                    elif(method=='cet' and n==8):
                        plt.text(x+offset, y+offset, "{}".format(n), horizontalalignment='left', verticalalignment='bottom', fontsize=text_fontsize)
                    elif(method=='cet' and n==16):
                        plt.text(x+offset, y+offset, "{}".format(n), horizontalalignment='left', verticalalignment='bottom', fontsize=text_fontsize)
                    elif(method=='cet' and n==19):
                        plt.text(x+offset, y-offset, "{}".format(n), horizontalalignment='left', verticalalignment='top', fontsize=text_fontsize)
                    else:
                        plt.text(x-offset, y-offset, "{}".format(n), horizontalalignment='right', verticalalignment='top', fontsize=text_fontsize)
        plt.xlabel('Cost (test)', fontsize=14); plt.ylabel('Loss (test)', fontsize=14); 
        if(dataset=='i'): plt.xlim(left=0.215); 
        if(dataset=='g'): plt.xlim(left=0.03); 
        plt.xticks(fontsize=12); plt.yticks(fontsize=12); plt.tight_layout(); 
        if(i==1): plt.legend(fontsize=12)

    plt.savefig('./res/complexity/{}/tradeoff_pareto.png'.format(model), bbox_inches='tight', pad_inches=0.05)
    plt.savefig('./res/complexity/{}/tradeoff_pareto.pdf'.format(model), bbox_inches='tight', pad_inches=0.05)
    plt.clf()

plot_sens_comp_pareto_frontier(model='L', datasets=['i', 'g'], gamma=1.0)


def plot_sens_comp_all(model='L', dataset='i', gamma=1.0):
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(10,4))
    for j, key2 in enumerate(['train', 'test']):
        for i, key1 in enumerate(['cost', 'loss', 'obj']):
            plt.subplot(2, 3, i + j*3 + 1)
            for method in METHODS.keys():
                df = pd.read_csv('./res/complexity/{}/{}_{}_{}.csv'.format(model, method, DATASET[dataset], gamma))
                plt.plot(df['n_actions'], df[key1+'_'+key2], marker=MARKER[method], label='{}'.format(METHODS[method]))
            plt.xlabel(r'\#Actions'); plt.ylabel('{} ({})'.format('Invalidity' if key1=='obj' else key1.capitalize(), key2)); plt.xticks([4,8,12,16,20]); plt.tight_layout(); 
            if(i==1 and j==0): plt.legend()
    plt.savefig('./res/complexity/{}/tradeoff_{}.png'.format(model, DATASET[dataset]), bbox_inches='tight', pad_inches=0.05); 
    plt.savefig('./res/complexity/{}/tradeoff_{}.pdf'.format(model, DATASET[dataset]), bbox_inches='tight', pad_inches=0.05);
    plt.clf()

# plot_sens_comp_all(model='L', dataset='i', gamma=1.0)
# plot_sens_comp_all(model='L', dataset='g', gamma=1.0)


def plot_sens_gamma(model='L', datasets=['i','g'], gammas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(8,2.5))
    for i, dataset in enumerate(datasets):
        plt.subplot(1, 2, i+1)
        plt.title(DATASETNAME[dataset])
        df = pd.read_csv('./res/gamma/{}/sensitivity_{}.csv'.format(model, DATASET[dataset]))
        plt.plot(gammas, [df[df['gamma']==g]['cost'].mean() for g in gammas], marker='o', label=r'cost $c(a \mid x)$ (MPS)')
        plt.plot(gammas, [df[df['gamma']==g]['loss'].mean() for g in gammas], marker='s', label=r'loss $l(f(x+a), +1)$')
        plt.xlabel(r'$\gamma$')
        if(i==0): plt.legend()
        plt.tight_layout()
    plt.savefig('./res/gamma/{}/sensitivity.png'.format(model), bbox_inches='tight', pad_inches=0.05)
    plt.savefig('./res/gamma/{}/sensitivity.pdf'.format(model), bbox_inches='tight', pad_inches=0.05)
    plt.clf()

# plot_sens_gamma()


def plot_sens_conv(model='L', dataset='g', gammas=[0.75, 1.0, 1.25], lambdas=[0.01, 0.03, 0.05]):
    plt.rcParams["font.family"] = 'arial'
    plt.rcParams['text.usetex'] = True
    if(len(lambdas)==1):
        plt.figure(figsize=(6,6))
        res_name = './res/convergence/{}/convergence_{}_partial'.format(model, DATASET[dataset])
    else:
        plt.figure(figsize=(14,7))
        res_name = './res/convergence/{}/convergence_{}'.format(model, DATASET[dataset])
    for i, g in enumerate(gammas):
        for j, l in enumerate(lambdas):
            plt.subplot(len(gammas), len(lambdas), i*len(lambdas) + j + 1)
            plt.title(r'$\gamma={}$, $\lambda={}$'.format(g,l))
            df = pd.read_csv('./res/convergence/{}/cet_{}_objective_{}_{}.csv'.format(model, DATASET[dataset], l, g))
            plt.plot(df['Iteration'], df['obj'], label=r'$t$-th objective value $o_{\gamma, \lambda}(h^{(t)})$')
            plt.plot(df['Iteration'], df['obj_bound'], label='Best objective value $o_{\gamma, \lambda}(h^{*})$')
            plt.xlabel(r'Iteration $t$'); plt.ylabel(r'Objective value $o_{\gamma, \lambda}(h)$')
            plt.tight_layout()
            if(i==0 and j==0): plt.legend()
            plt.tight_layout()
    plt.savefig(res_name+'.png', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(res_name+'.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.clf()

# plot_sens_conv(dataset='g', gammas=[0.75, 1.0, 1.25], lambdas=[0.01, 0.03, 0.05])
# plot_sens_conv(dataset='i', gammas=[0.75, 1.0, 1.25], lambdas=[0.01, 0.03, 0.05])

