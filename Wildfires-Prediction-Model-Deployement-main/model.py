import pandas as pd
import sklearn as sk

def predictions(inputt):

    Wildfires_data = pd.read_table ('Wildfires_pred_Dataset.txt',sep=',')

    y1= Wildfires_data.CLASS

# add a new column to the dataset (0 ~~ no_fire , 1 ~~ fire)

    Binary_class=pd.Series(1713,)

    for i in range(1713):
        if y1[i]== 'fire':
            Binary_class[i]= 1
        else:
            Binary_class[i]= 0


    Wildfires_data['Binary_CLASS'] = Binary_class

    #choosing target and features
    Y = Wildfires_data.Binary_CLASS
    wildfires_features = ['NDVI','LST','BURNED_AREA']
    X=Wildfires_data[wildfires_features]
   
    #standarsization
    from sklearn.preprocessing import StandardScaler

    scaler= StandardScaler().fit(X)
    standard_X=scaler.transform(X)

    #Data oversampling
    from imblearn.over_sampling import SMOTE

    balanced_data=  SMOTE(sampling_strategy='auto',
                      k_neighbors=1, 
                      random_state=100)

    X_balanced,y_balanced=balanced_data.fit_resample(standard_X,Y)

    #splitting the data to training and testing

    from sklearn.model_selection import train_test_split

    X_train ,X_test ,y_train ,y_test = train_test_split (X_balanced ,
                                                    y_balanced ,
                                                     random_state=0)
    #build and train the model

    from sklearn.svm import SVC
    fire_classifier = SVC(kernel='rbf',gamma=100, C=1)
    fire_classifier.fit(X_train, y_train)

    
    #Standardize the input
    standard_inputt=scaler.transform(inputt)
    prediction=fire_classifier.predict(inputt)

    return prediction




