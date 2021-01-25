import logging as LOGGER
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scikitplot as skplt
import pandas as pd
import numpy as np
import pickle
import shap

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from skopt import BayesSearchCV
from scipy.stats import pearsonr

from src.data.data import create_directory

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
LOGGER.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=LOGGER.INFO)

class BaseModel:
    def __init__(self):
        self.model = None

    def fit(self, x, y):
        if self.model is not None:
            self.X = x
            self.model = self.model.fit(x if isinstance(x, np.ndarray) else x.values, y)
            if isinstance(self.model, BayesSearchCV):
                LOGGER.info(f"Best model's F1: {self.model.best_score_}")
                LOGGER.info(f"Best model's parameters: {self.model.best_params_}")
                self.model = self.model.best_estimator_
            return
        raise NotImplementedError()

    def predict(self, x):
        if self.model is not None:
            try:
                return self.model.predict(x.values)
            except:
                return self.model.predict(x)
        raise NotImplementedError()

    def predict_proba(self, x):
        if self.model is not None:
            try:
                return self.model.predict_proba(x.values)
            except:
                return self.model.predict_proba(x)
        raise NotImplementedError()

    def evaluate(self, x, y_true, output_directory, cols=None, transition=0):
        """
        Evaluate Model regarding performance metrics
        and some performance-measuring plots
        """
        if cols is None:
            cols = self.X.columns
        
        try:
            pd.DataFrame(x, columns=cols).to_parquet(output_directory / 'x_te.parquet')
        except:
            pass
        y_pred = self.predict(x)
        
        y_pred_prob = self.predict_proba(x)
#         print(y_true.shape)
#         print(y_pred_prob.shape)
#         pd.DataFrame({'y_true': y_true.reshape(-1)}).to_parquet(output_directory / "y_te_true.parquet")
#         pd.DataFrame(y_pred_prob.reshape(-1,2), columns=[f'class{i}' for i in range(2 if transition == 0 else 4)]).to_parquet(output_directory / "y_te_pred.parquet")
        
        
        # Metrics
        metrics = {}
        metrics["model"] = output_directory.stem
        if transition == 2:
            for i in range(4):
                truth = y_true == i
                prediction = y_pred == i

                truth = truth.reshape(-1,)
                prediction = prediction.reshape(-1,)
                
                print(y_pred_prob.shape)
                
                metrics[f"f1_{i}"] = f1_score(truth, prediction)
                metrics[f"precision_{i}"] = precision_score(truth, prediction)
                metrics[f"recall_{i}"] = recall_score(truth, prediction)
                metrics[f"accuracy_{i}"] = accuracy_score(truth, prediction)
                metrics[f"au-roc_{i}"] = roc_auc_score(truth, y_pred_prob.reshape(-1,4)[:,i])
                
        metrics["f1"] = f1_score(y_true.reshape(-1,), y_pred.reshape(-1,), average='macro')
        metrics["precision"] = precision_score(y_true.reshape(-1,), y_pred.reshape(-1,), average='macro')
        metrics["recall"] = recall_score(y_true.reshape(-1,), y_pred.reshape(-1,), average='macro')
        metrics["accuracy"] = accuracy_score(y_true.reshape(-1,), y_pred.reshape(-1,))
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_prob[:,1])
        
#         if transition == 0:
#             metrics["au-roc"] = roc_auc_score(y_true, y_pred_prob[:,1], average='micro')
#             metrics["average_precision"] = average_precision_score(y_true, y_pred)
#         metrics["log-loss"] = log_loss(y_true, y_pred_prob)

        pd.DataFrame([metrics], columns=metrics.keys()).to_csv(
            output_directory / "metrics.csv", index=False
        )
        
        # Plots
#         with PdfPages(output_directory / "plots.pdf", "w") as pdf:
#             skplt.metrics.plot_confusion_matrix(y_true, y_pred)
#             pdf.savefig()
#             plt.close()
#             skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
#             pdf.savefig()
#             plt.close()
#             skplt.metrics.plot_precision_recall(y_true, y_pred_prob)
#             pdf.savefig()
#             plt.close()
#             skplt.metrics.plot_roc(y_true, y_pred_prob)
#             pdf.savefig()
#             plt.close()
#             if hasattr(self.model, "feature_importances_"):
#                 skplt.estimators.plot_feature_importances(self.model)
#                 pdf.savefig()
#                 plt.close()
#             cls_name = self.__class__.__name__.lower()
#             if "xgb" in cls_name or "forest" in cls_name:
#                 plt.figure(figsize=(15, 7))
#                 explainer = shap.TreeExplainer(self.model, self.X.values)
#                 shap_values = explainer.shap_values(self.X.values)
#                 shap.summary_plot(
#                     shap_values, self.X.values, [i.replace('_', '-') for i in cols], show=False
#                 )
#                 pdf.savefig()
#                 plt.close()
#             elif "logistic" in cls_name:
#                 plt.figure(figsize=(15, 7))
#                 explainer = shap.LinearExplainer(
#                     self.model, self.X.values, feature_dependence="independent"
#                 )
#                 shap_values = explainer.shap_values(self.X.values)
#                 shap.summary_plot(
#                     shap_values, self.X.values, [i.replace('_', '-') for i in cols], show=False
#                 )
#                 pdf.savefig()
#                 plt.close()

#             elif "nn" in cls_name or "gru" in cls_name or "lstm" in cls_name:
# #                 explainer = shap.DeepExplainer(self.model, self.X[:10])
# #                 shap_values = explainer.shap_values(x[1:5])
# #                 shap.summary_plot(shap_values, x[1:5], show=False)
# #                 pdf.savefig()
# #                 plt.close()
                
#                 plt.figure(figsize=(15, 7))
#                 plt.title('Feature-Target Pearson\'s Correlation')
#                 corrs = [pearsonr(y_pred_prob[:, 1], col)[0] for col in x[:,0,:].transpose()]
#                 col_corrs = pd.DataFrame({'cols' : [i.replace('_', '\_') for i in cols], 'corrs':corrs, 'abs':np.abs(corrs)})
#                 col_corrs = col_corrs.sort_values('abs', ascending=False)
#                 rng = range(10)
#                 plt.barh(rng, col_corrs.iloc[:10, 1].values[::-1])
#                 plt.yticks(rng[::-1], col_corrs.iloc[:10, 0])
#                 pdf.savefig()
#                 plt.close()
                

#             if transition == 0:
#                 skplt.metrics.plot_ks_statistic(y_true, y_pred_prob)
#                 pdf.savefig()
#                 plt.close()
#                 skplt.metrics.plot_calibration_curve(y_true, [y_pred_prob])
#                 pdf.savefig()
#                 plt.close()
#                 skplt.metrics.plot_lift_curve(y_true, y_pred_prob)
#                 pdf.savefig()
#                 plt.close()
#                 skplt.metrics.plot_cumulative_gain(y_true, y_pred_prob)
#                 pdf.savefig()
#                 plt.close()

    def pickle_model(self, output_directory):
        if self.model is not None:
            pickle.dump(self.model, open(output_directory / "model.pickle", "wb"))