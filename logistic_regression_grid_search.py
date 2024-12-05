# l1 uses only liblinear and saga
# elastic net only uses saga and requires l1_ratios
# l2 can use all three solvers

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import numpy as np
from imblearn.over_sampling import SMOTE
import time


# FIND COMPOSITE SCORES OF ROC_AUC AND F1
# NEXT< SUBTRACT THE STANDARD DEVIATION FROM THE MEAN SCORE TO FIND THE BEST MODEL



df = pd.read_csv('final_breast_cancer.csv')
X = df.drop('Status', axis = 1)
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

scaler = StandardScaler()
smote = SMOTE(random_state=101)

f1_scorer = make_scorer(f1_score, average='macro')
acc_scorer = make_scorer(accuracy_score)

def evaluate_model(degree, interaction_only, c, penalty, solver, metric, _smote):
    model = LogisticRegression(max_iter=10000, penalty = penalty, C = c, solver = solver, random_state=101)
    poly_converter = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=interaction_only)
    
    X_train_poly = poly_converter.fit_transform(X_train)
    
    if _smote:
        X_train_poly, y_train_resampled = smote.fit_resample(X_train_poly, y_train)
    else:
        y_train_resampled = y_train
        
    X_train_scaled = scaler.fit_transform(X_train_poly)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)
    
    if metric == 'roc_auc':
         scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=cv, scoring='roc_auc')
         mean_scores = np.mean(scores)
         return mean_scores

    else:
        f1_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=cv, scoring=f1_scorer)
        accuracy = cross_val_score(model, X_train_scaled, y_train_resampled, cv=cv, scoring=acc_scorer)

        mean_f1_scores = np.mean(f1_scores)
        mean_accuracy = np.mean(accuracy)
        return mean_f1_scores, mean_accuracy

def find_best_model(_smote):
   
    top_models = []
    best_f1_mean = 0  
    best_model = None  
    for degree in range(1, 2):
        for interaction_only in [True, False]:
            for c in [0.001, 0.01, 0.1, 1, 10, 100]:
                for penalty in ['l1', 'l2']:
                    for solver in ['saga', 'liblinear']:
                        
                        roc_auc_mean = evaluate_model(degree, interaction_only, c, penalty, solver, 'roc_auc', _smote)
                        model_statistics = {
                            'best_roc_auc':roc_auc_mean,
                            'best_degree':degree,
                            'interaction_only':interaction_only,
                            'best_C':c,
                            'best_penalty':penalty,
                            'best_solver':solver,
                            'smote':_smote
                        }
                    
                        top_models.append(model_statistics)
    
    
    highest_auc_roc_models = sorted(top_models, key = lambda x: x['best_roc_auc'], reverse=True)[:10]                
    
    for model in highest_auc_roc_models:
        f1_mean, acc_mean = evaluate_model(model['best_degree'], model['interaction_only'], model['best_C'], model['best_penalty'], model['best_solver'], 'f1', _smote)
        model['f1_mean'] = f1_mean
        model['accuracy'] = acc_mean
        if f1_mean > best_f1_mean:
            best_f1_mean = f1_mean
            best_model = model
    
    return best_model

start_time = time.time()    
model_with_smote = find_best_model(True)
end_time = time.time()    
print(f"Time taken to find the best model with SMOTE: {end_time - start_time:.2f} seconds")

start_time = time.time()    
model_without_smote = find_best_model(False)
end_time = time.time()    
print(f"Time taken to find the best model without SMOTE: {end_time - start_time:.2f} seconds")

print('Best model with SMOTE:', model_with_smote)
print('Best model without SMOTE:', model_without_smote)

