import pickle
import os

def load_data():
    import pandas as pd
    DATA_PATH = os.getenv('DATA_DIR')
    spotify_df = pd.read_csv(os.path.join(DATA_PATH, 'Spotify/spotify_songs.csv'))
    
    X_columns = ['playlist_genre', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms']
    y_column = 'track_popularity'

    X = spotify_df[X_columns]
    y = spotify_df[y_column]
    
    print("Өгөгдлийг импортлосон")
    
    return X, y


def y_to_cat(y):
    y = y.map(lambda x: 1 if x>=70 else 0)
    
    return y


def train_test(X, y):
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    print("Сургалтын Х-н хэмжээ:", X_train.shape, "у-н хэмжээ:", y_train.shape)
    print("Тестийн Х-н хэмжээ:", X_test.shape, "у-н хэмжээ:", y_test.shape)
    
    return X_train, X_test, y_train, y_test

def scale_process_data():
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test(X, y)
    
    num_columns = [col for col in X_train.columns if X_train[col].dtype in ['float','int']]
    cat_columns = [col for col in X_train.columns if X_train[col].dtype not in ['float','int']]
    
    scaler = MinMaxScaler()
    scaler.fit(X_train[num_columns])
    X_train[num_columns] = scaler.transform(X_train[num_columns])
    X_test[num_columns] = scaler.transform(X_test[num_columns])
    
    with open(os.path.join(os.getenv('MODEL_DIR'), 'scaler.pickle'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"{len(num_columns)} тоон хувьсагчийг хувиргав")
        
    labelencoder = LabelEncoder()

    for col in cat_columns:
        labelencoder.fit(X_train[col])
        X_train[col] = labelencoder.transform(X_train[col])
        X_test[col] = labelencoder.transform(X_test[col])
        encoder_name = 'labelencoder_'+col+".pickle"
        
        with open(os.path.join(os.getenv('MODEL_DIR'), encoder_name), 'wb') as f:
            pickle.dump(labelencoder, f)
            
    print(f"{len(cat_columns)} категори хувьсагчийг хувиргав")
            
    y_train = y_to_cat(y_train)
    y_test = y_to_cat(y_test)  
    
    return X_train, X_test, y_train, y_test

def build_model(max_depth, min_samples_leaf, min_samples_split):
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(random_state=123,
                                   max_depth=max_depth, 
                                   min_samples_leaf=min_samples_leaf, 
                                   min_samples_split=min_samples_split)

    return model

def train_eval_model(model, data):
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    
    model.fit(data[0], data[2])
    y_pred = model.predict(data[1])
    
    print("Recall score:", recall_score(data[3], y_pred))
    print("Precision score:", precision_score(data[3], y_pred))
    print("Accuracy score:", accuracy_score(data[3], y_pred))
    
    return model 

def save_model(model):
    with open(os.path.join(os.getenv('MODEL_DIR'), 'model.pickle'),'wb') as f:
        pickle.dump(model, f)
        
    print("Model saved")
    
    
def main():
    max_depth = int(os.getenv('MAX_DEPTH'))
    min_samples_leaf = int(os.getenv('MIN_SAMPLES_LEAF'))
    min_samples_split = int(os.getenv('MIN_SAMPLES_SPLIT'))

    data = scale_process_data()
    model = build_model(max_depth, min_samples_leaf, min_samples_split)
    model_trained = train_eval_model(model, data)
    save_model(model_trained)
    
if __name__ == "__main__":
    main()