import pickle
import os
import json

with open('config.json', 'r') as f:
    config_json = json.load(f)

def load_data():
    import pandas as pd
    DATA_PATH = config_json['DATA_DIR']
    spotify_df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))

    X = spotify_df[config_json['DATA_COLUMNS']['X_COLUMNS']]
    y = spotify_df[config_json['DATA_COLUMNS']['Y_COLUMN']]
    
    print("Өгөгдлийг импортлосон")
    
    return X, y


def y_to_cat(y):
    y = y.map(lambda x: 1 if x>=70 else 0)
    
    return y

def scale_process_data():
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    
    X, y = load_data()
    
    num_columns = [col for col in X.columns if X[col].dtype in ['float','int']]
    cat_columns = [col for col in X.columns if X[col].dtype not in ['float','int']]
    
    
    with open(os.path.join(config_json['MODEL_DIR'], 'scaler.pickle'), 'rb') as f:
        scaler = pickle.load(f)
    
    X[num_columns] = scaler.transform(X[num_columns])
        
    print(f"{len(num_columns)} тоон хувьсагчийг хувиргав")
        
   

    for col in cat_columns:
        encoder_name = 'labelencoder_'+col+".pickle"
        with open(os.path.join(config_json['MODEL_DIR'], encoder_name), 'rb') as f:
            labelencoder = pickle.load(f)
            
        X[col] = labelencoder.transform(X[col])
            
    print(f"{len(cat_columns)} категори хувьсагчийг хувиргав")
            
    y = y_to_cat(y)  
    
    return X, y 

def load_model():
    
    with open(os.path.join(config_json['MODEL_DIR'],'model.pickle'),'rb') as f:
        model = pickle.load(f)

    return model

def inference(model, data):
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    
    y_pred = model.predict(data[0])
    y_pred_proba = model.predict_proba(data[0])[::,1]
    
    print("Recall score:", recall_score(data[1], y_pred))
    print("Precision score:", precision_score(data[1], y_pred))
    print("Accuracy score:", accuracy_score(data[1], y_pred))
    
    return y_pred, y_pred_proba

def visualize_inference(y, y_pred, y_pred_proba):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    import matplotlib.pyplot as plt
    
    if not os.path.exists("./Charts"):
        os.mkdir("./Charts")
    else:
        pass
    
    fpr, tpr, thres = roc_curve(y, y_pred)
    
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, 'b', label='AUC %0.2f'%roc_auc)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.savefig("./Charts/ROC.png")
    plt.show()
    
    precision, recall, thresh = precision_recall_curve(y, y_pred_proba)
    plt.plot(recall, precision, 'r')
    plt.savefig("./Charts/PRC.png")
    plt.show()
    
    
    
def main():

    data = scale_process_data()
    model = load_model()
    y_pred, y_pred_proba = inference(model, data)
    visualize_inference(data[1], y_pred, y_pred_proba)
    
if __name__ == "__main__":
    main()