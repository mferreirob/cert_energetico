import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import os

def plot_feature_importance(model, X_train):
    """Genera un gráfico de barras con la importancia de las variables."""
    feat_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns, columns=["Importance"])
    feat_importances.sort_values(by='Importance', ascending=False, inplace=True)
    
    fig = px.bar(feat_importances, x=feat_importances.index, y='Importance', 
                 title='Feature Importances', labels={'x': 'Features', 'Importance': 'Importance'}, 
                 template='plotly_white')
    fig.show()
    return feat_importances

def saca_metricas(y_real, y_pred):
    """Calcula y muestra métricas de clasificación y la curva ROC."""
    print('Matriz de Confusión')
    print(confusion_matrix(y_real, y_pred))
    print('Accuracy:', accuracy_score(y_real, y_pred))
    print('Precision:', precision_score(y_real, y_pred))
    print('Recall:', recall_score(y_real, y_pred))
    print('F1 Score:', f1_score(y_real, y_pred))

    # Curva ROC
    false_positive_rate, recall, _ = roc_curve(y_real, y_pred)
    roc_auc = auc(false_positive_rate, recall)
    print('AUC:', roc_auc)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=false_positive_rate, y=recall, mode='lines', name='Curva ROC'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Línea base', line=dict(dash='dash')))
    fig.update_layout(title=f'Curva ROC (AUC = {roc_auc:.2f})', template='plotly_white')
    folder = "reports"
    if not os.path.exists(folder):
            os.makedirs(folder)    
            
    path_final = os.path.join(folder, nombre_archivo)
    
    # 3. Guardamos
    fig.write_html(path_final)
    print(f"📊 Gráfico guardado en {path_final}")   