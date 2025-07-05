import pandas as pd
from scipy import stats

def prepare_data(data):
    data = data.ffill()
    data = data.set_index('Model')

    metrics = data.columns
    models = data.index.unique()

    statistical_significance = {}

    for metric in metrics:
        for idm1, model1 in enumerate(models):
            better = []
            for idm2, model2 in enumerate(models):
                if model1 != model2:
                    if model1 == model2:
                        continue
                    data1 = data.loc[model1, metric].values
                    data2 = data.loc[model2, metric].values

                    # print(data1.mean(), data2.mean())
                    res = stats.ttest_rel(data1, data2)
                    # print(res)
                    if res.pvalue < 0.05:
                        if res.statistic > 0:
                            # print(f'{model1} is better than {model2} in {metric}, statistic: {res.statistic}')
                            better.append(idm2 + 1)

            if better:
                statistical_significance[model1, metric] = better
    return data, metrics, models, statistical_significance

def prepare_table(data, metrics, models, statistical_significance, task_name):
    table = '\\begin{table*}[h]\n'
    table += f'''
    \\centering
    \\caption{{Average cross-validation results of models on the {task_name} task.}}
    \\resizebox*{{\\textwidth}}{{!}}{{
    \\begin{{tabular}}{{ccccccc}}
        \\# & \\textbf{{Model}}'''
    for metric in metrics:
        table += f' & \\textbf{{{metric}}} '
    table += '\\\\ \\hline \n        '

    for idm, model in enumerate(models):
        table += f'\\multirow{{2}}{{*}}{{\\textit{{{idm +1}}}}} & '
        table += f'\\multirow{{2}}{{*}}{{\\textsc{{{model}}}}} & '

        for jdx, metric in enumerate(metrics):
            better = statistical_significance.get((model, metric))
            if better:
                if jdx % 2 == 1:
                    if len(better) == 1:
                        table += f'\\textit{{{better[0]}}} & '
                    else:
                        table += f'\\textit{{{",".join(map(str,better))}}} & '
                else:
                    if len(better) == 1:
                        table += f'\\textit{{{better[0]}}} \cellcolor{{gray!10}} & '
                    else:
                        table += f'\\textit{{{",".join(map(str,better))}}} \cellcolor{{gray!10}} & '
            else:
                if jdx % 2 == 1:
                    table += f'& '
                else:
                    table += f'\cellcolor{{gray!10}} & '

        table = table[:-2] + '\\\\\n        '        
        table += ' & & '

        for jdx, metric in enumerate(metrics):
            if jdx % 2 == 1:
                table += f'{data.loc[model, metric].mean():.3f}$\\pm${data.loc[model, metric].std():0.3f} & '
            else:
                table += f'{data.loc[model, metric].mean():.3f}$\\pm${data.loc[model, metric].std():0.3f} \cellcolor{{gray!10}} & '
        table = table[:-2] + '\\\\ \\hline \n        '

    table = table[:-4] + f'''\\end{{tabular}}
    }}
    \\label{{tab:{task_name}}}
\\end{{table*}}'''

    with open(f'{task_name}.tex', 'w') as f:
        f.write(table)
    
    return table


def prepare_doc(table):
    doc = '''
    \\documentclass{article}
    \\usepackage{multirow}
    \\usepackage{graphicx}
    \\begin{document}

    '''

    doc += table
    doc += '''
    \\end{document}
    '''

    with open('analiza.tex', 'w') as f:
        f.write(doc)

if __name__ == '__main__':
    
    data = pd.read_excel('ppo.xlsx', sheet_name='Arkusz1',usecols='A:F', skiprows=0, nrows=25, header=0)

    data, metrics, models, statistical_significance = prepare_data(data)
    table = prepare_table(data, metrics, models, statistical_significance, task_name='PPO')
    prepare_doc(table)

    data = pd.read_excel('ppc.xlsx', sheet_name='Arkusz1',usecols='A:F', skiprows=0, nrows=25, header=0)

    data, metrics, models, statistical_significance = prepare_data(data)
    table = prepare_table(data, metrics, models, statistical_significance, task_name='PPC')
    prepare_doc(table)
