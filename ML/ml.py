"""

"""

__author__ = "Bruno Faria & Dylan Perdigão"
__date__ = "May 2022"


from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import exists
import pandas as pd
import pickle
import psycopg2
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time


def connect_to_db():
    """ 
    Connect to the PostgreSQL database server 

    Returns: 
        connection: a connection object
    """

    return psycopg2.connect(
		user="bi2022",
		password="in2022$$",
		host="bi2022.postgres.database.azure.com",
		port="5432",
		database="postgres"
	)

def read_data():
    """
    Reads the data from pickle, if exists, or from the database

    Returns:
        df: a dataframe with the data
    """

    if exists("ML/data.pkl"):
        print("Reading data from pickle...")
        df = read_data_from_pickle()
    else:
        print("Reading data from database...")
        df = read_data_from_db()

        #saves to pickle
        print("Saving data to pickle...")
        df.to_pickle("ML/data.pkl")

    return df

def read_data_from_pickle():
    """
    Read the data from the pickle file

    Returns:
        data: a dataframe with the data
    """
    
    return pd.read_pickle("ML/data.pkl")

def read_data_from_db():
    """
    Read the data from the database

    Returns:
        data: a dataframe with the data
    """
    connection = connect_to_db()


    sql_query = pd.read_sql_query ('''
                                SELECT *
                                FROM fact, flight, person, satisfaction, ticket
                                    WHERE fact.flight_id = flight.id
                                        AND fact.person_id = person.id
                                        AND fact.satisfaction_id = satisfaction.id
                                        AND fact.ticket_id = ticket.id;
                               ''', connection)

    df = pd.DataFrame(sql_query)

    connection.close()

    return df

def treat_data(df):
    """
    Treats the data

    Parameters:
        df: a dataframe with the data

    Returns:
        df: a dataframe with the treated data
    """
    
    #print(df.columns)
    #print(df.shape)

    ## Remove columns with ids
    cols = [c for c in df.columns if c.lower()[-2:] != 'id']
    df=df[cols]

    #print(df.columns)
    #print(df.shape)


    ## Change object columns to int
    #print(df.dtypes)

    # Check conflicting columns and object values
    print('Non-int columns')
    for col in df.columns:
        if df[col].dtypes == 'object':
            print(col, df[col].unique())

    # Replace values with int
    df.loc[:,'overall_satisfaction'] = df['overall_satisfaction'].replace({'satisfied':1, 'neutral or dissatisfied':0})
    
    df.loc[:,'gender'] = df['gender'].replace({'Male':1, 'Female':0})
    df.loc[:,'loyalty'] = df['loyalty'].replace({'Loyal Customer':1, 'disloyal Customer':0})
    df.loc[:,'flight_class'] = df['flight_class'].replace({'Business':2, 'Eco Plus':1, 'Eco':0 })
    df.loc[:,'type_travel'] = df['type_travel'].replace({'Business travel':1,'Personal Travel':0})
    
    #print(df.dtypes)

    return df

def visualize_data(df, img_path):
    """
    Visualizes the data
    
    Parameters:
        df: a dataframe with the data
        img_path: the path to the images folder
    """

    print("===== Data Head =====")
    print(df.head)
    print("===== Data Description =====")
    print(df.describe())

    #print("===== Data y =====")

    ## Overall satisfaction Counts
    plt.figure(figsize=(5,5))
    ax = sns.countplot(x="overall_satisfaction", data=df, palette=['#fa7256','#f89649'])
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.5, p.get_height()), ha='center', va='top', color='white', size=18)
    
    plt.savefig(img_path+"/data_visualization/overall_satisfaction_counts.png")
    #plt.show()


    ## Correlation matrix
    correlations = df.corr(method='pearson')
    
    # heatmap of correlations
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    
    cmap = create_colormap('0b2f5b','fa7256','f7d380')
    cax = ax.matshow(correlations, vmin=-1, vmax=1,  cmap=cmap)
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.columns),1)
    #ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    #ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.columns)
    plt.savefig(img_path+"/data_visualization/correlation_matrix.png")
    #plt.show()

    ## Histograms
    df.hist()
    plt.savefig(img_path+"/data_visualization/histgrams.png")
    #plt.show()

    ## Density Plots
    df.plot(kind='density', subplots=True, layout=(5,5), sharex=False, sharey=False)
    #plt.show()
    plt.savefig(img_path+"/data_visualization/density.png")

def plot_3d_scatter(x,y, title, img_path, size=10):
    """
    Plots and saves a 3d scatter plot

    Parameters:
        x: the 3 features array to plot
        y: the labels
        title: the title
        img_path: the path to the images folder
        size: the size of the points
    """

    axis_labels = x.columns
    x=x.values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc= ax.scatter(x[:,0],x[:,1],x[:,2], c=y, s = size)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    plt.legend(handles=sc.legend_elements()[0], 
           labels=['satisfied', 'neutral or dissatisfied'],
           title="overall satisfaction")


    #plt.show()
    plt.savefig(img_path+"/feature_selection/"+title+".png")

def feature_selection(X, y, N, img_path = None):
    """
    Performs feature selection

    Parameters:
        X: the features array
        y: the labels
        N: the number of features to select
        img_path: the path to the images folder

    Returns:
        us: a dataframe with the selected features from Univariate Selection
        rfe: a dataframe with the selected features from RFE
        pca: a dataframe with the reduced features from PCA
        fi: a dataframe with the selected features from Feature Importance
    """

    print("\n======================================== Feature Selection ========================================")

    col_names = X.columns.tolist()

    ## Univariate Statistical Tests (Chi-squared for classification)
    print("\n==================== \tUnivariate Statistical Tests\t ====================")

    test= SelectKBest(score_func=chi2, k=N)
    fit = test.fit(X, y)
    #print(fit)

    # sumarize scores
    #print(fit.scores_)
    #us = fit.transform(X)
   
    selected_cols = [col_names[i] for i in fit.scores_.argsort()[-N:]]
    us = X[selected_cols]

    if img_path is not None:
        plot_3d_scatter(us,y, "univariate_selection", img_path, 5)

    print("Univariate Selection best Features:\t", selected_cols)
    print(us.head())
    # proof that they are the same
    #print(np.unique(us == X[selected_cols].values))


    ## Recursive Feature Elimination (RFE)
    print("\n==================== \tRecursive Feature Elimination\t ====================")
    model = LogisticRegression(solver='lbfgs', max_iter=3000)
    #lbfgs solver - uses the Limited Memory Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm

    rfe = RFE(model, n_features_to_select=N)
    fit = rfe.fit(X, y)

    #rfe = fit.transform(X)

    selected_cols = [col_names[i] for i in range(len(col_names)) if fit.support_[i]]
    rfe = X[selected_cols]

    if img_path is not None:
        plot_3d_scatter(rfe,y, "rfe", img_path)

    print("RFE best Features:\t\t\t", selected_cols)
    print(rfe.head())


    ## Principal Component Analysis
    print("\n==================== \tPrincipal Component Analysis\t ====================")

    #scale
    X_N = StandardScaler().fit_transform(X)

    pca_f = PCA(n_components=N)
    pca_np = pca_f.fit_transform(X_N)

    # summarize components
    #print("Explained Variance: %s" % pca_f.explained_variance_ratio_)

    cols = ['PC '+str(i+1) for i in range(N)]
    pca = pd.DataFrame(pca_np, columns=cols)
    print(pca.head())

    if img_path is not None:
        plot_3d_scatter(pca,y, "pca", img_path, 5)

    #plt.plot(pca_f.explained_variance_ratio_, 'bd-')
    #plt.show()

    
    ## Feature Importance
    print("\n==================== \tFeature Importance\t ====================")

    model = ExtraTreesClassifier(n_estimators=100, random_state=0)
    model.fit(X, y)

    idx = model.feature_importances_.argsort()[-N:]
    selected_cols = [col_names[i] for i in idx]

    fi = X[selected_cols]

    if img_path is not None:
        plot_3d_scatter(fi,y, "feature_importance", img_path)

    print("Feature Importance best Features:\t", selected_cols)
    print(fi.head())

    return [us, rfe, pca, fi]

def classify(X,y, seed, data_path, red_name):
    """
    Performs classificationwith diferent models
        
    Parameters:
        X: the features
        y: the labels
        seed: the seed for the random state
        data_path: the path to the data folder
        red_name: the name of the reduceding algorithm
    """
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("\n===================== \t", red_name, "\t =====================")
    
    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=3000, multi_class='auto')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('SVM', SVC(gamma='scale')))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
    
        begin = time.time()
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        after = time.time()

        results.append(cv_results)
        names.append(name)

        line = "%s,%f,%f,%d" % (name,cv_results.mean()*100, cv_results.std()*100,after - begin)
        #write line to file
        with open(data_path+"/"+red_name+".csv", "a") as f:
            f.write(line+"\n")

        print("%s: \t%f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100))

    # boxplot algorithm comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    
    #plt.show()
    plt.savefig(data_path+"/"+red_name+".png")

def visualize_classification(data_path):
    """
    Visualizes the classification results
    
    Parameters:
        data_path: the path to the data folder
    """

    # get all the csv files in the data folder
    files = [f for f in listdir(data_path) if f.endswith(".csv")]

    # get the names of the algorithms
    file_names = [f.split(".")[0] for f in files]

    means_all = np.zeros((6,len(files)))
    times_all = np.zeros((6,len(files)))
    
    for i,file in enumerate(files):
        # read the file
        df = pd.read_csv(data_path+"/"+file)

        # get the means and times)
        means = df.iloc[:,1].values
        times = df.iloc[:,3].values

        # store the means and times
        means_all[:,i] = means
        times_all[:,i] = times

    model_names = df.iloc[:,0].values


    cmap = create_colormap('fa7256','f7d380')

    # plot heatmaps using seaborn
    sns.heatmap(means_all, annot=True, xticklabels=file_names, yticklabels=model_names, cmap=cmap)
    #plt.show() 
    plt.savefig(data_path+"\heatmaps_means.png")
    plt.close()

    sns.heatmap(times_all, annot=True, xticklabels=file_names, yticklabels=model_names, cmap=cmap)
    #plt.show()
    plt.savefig(data_path+"\heatmaps_time.png")
    plt.close()

    plt.figure()
    plt.bar(file_names, times_all[-1,:], color='#f89649')
    plt.ylabel("Time (s)")
    #plt.show()
    plt.savefig(data_path+"\svm_time.png")
    plt.close()

    plt.figure()
    plt.bar(model_names, times_all[:,-1], color='#f89649')
    plt.ylabel("Time (s)")
    #plt.show()
    plt.savefig(data_path+"\whole_time.png")
    plt.close()

def create_colormap(hex1,hex2, hex3 = None):
    r1,g1,b1 = (int(hex1[i:i+2], 16)/255 for i in (0,2,4))
    r2,g2,b2 = (int(hex2[i:i+2], 16)/255 for i in (0,2,4))

    myColors=[]

    if hex3 is not None:
        r3,g3,b3 = (int(hex3[i:i+2], 16)/255 for i in (0,2,4))
        
        for i in range(500):
            myColors.append((r1+(r2-r1)*(i+1)/500, g1+(g2-g1)*(i+1)/500, b1+(b2-b1)*(i+1)/500, 1.0))
            
        for i in range(500):
            myColors.append((r2+(r3-r2)*(i+1)/500, g2+(g3-g2)*(i+1)/500, b2+(b3-b2)*(i+1)/500, 1.0))

    else:
        for i in range(1000):
            myColors.append((r1+(r2-r1)*(i+1)/1000, g1+(g2-g1)*(i+1)/1000, b1+(b2-b1)*(i+1)/1000, 1.0))

    return LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

def plot_confusion_matrix(cm, best_path):
    """
    Plots and saves the confusion matrix
    
    Parameters:
        cm: the confusion matrix
        best_path: the path to the best model
    """
   
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cm.flatten()/np.sum(cm)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    cmap = create_colormap('ff4b41','f7d380')

    ax = sns.heatmap(cm, annot=labels, fmt='', cmap=cmap)

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    #plt.show()
    plt.savefig(best_path+"/confusion_matrix.png",)

def train_best_model(best_path):
    """
    Trains and saves the best model

    In our case the best model was CART on the whole Dataset

    Parameters:
        best_path: the path to the best model files
    """

    # get the data
    data = read_data()

    data = treat_data(data)

    # get the features and labels
    y=data['overall_satisfaction']
    X=data.drop(['overall_satisfaction'], axis=1)

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # evaluate the model - accuracy, precision, sensitivity, specificity from confusion matrix
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    print("Accuracy: ", (tp + tn) / (tp + fp + fn + tn))
    print("Precision: ", tp / (tp + fp))
    print("Sensitivity: ", tp / (tp + fn))
    print("Specificity: ", tn/(tn+fp))
    print("AUC Score: ", roc_auc_score(y_test, predictions))

    # plot the confusion matrix
    plot_confusion_matrix(cm, best_path)


    # pickle the model
    pickle.dump(model, open(best_path+"/best_model.pkl", "wb"))

    
def main():

    data = read_data()
    #print(data.shape)

    data = treat_data(data)
    #print(data.shape)

    visualize_data(data,IMG_PATH)

    y=data['overall_satisfaction']
    X=data.drop(['overall_satisfaction'], axis=1)

    [us3, rfe3, pca3, fi3] = feature_selection(X,y, 3, IMG_PATH)
    [us10, rfe10, pca10, fi10] = feature_selection(X,y, 10)

    print("\n======================================== CLASSIFICATION ========================================")
    for X,red_name in zip([X, us3, rfe3, pca3, fi3, us10, rfe10, pca10, fi10], ['WHOLE','US_3', 'RFE_3', 'PCA_3', 'FI_3', 'US_10', 'RFE_10', 'PCA_10', 'FI_10']):
        #create file for each algorithm with header
        with open(DATA_PATH+"/"+red_name+".csv", "w") as f:
            f.write("model,mean,std,train_time\n")

        classify(X,y, SEED, DATA_PATH, red_name)


if __name__ == "__main__":
    pd.set_option('precision', 3)
    plt.rcParams.update({'font.size': 8})

    IMG_PATH = "ML/img"
    DATA_PATH = "ML/data"
    BEST_PATH = "ML/best"
    SEED = 123456789

    #main()
    #visualize_classification(DATA_PATH)
    #train_best_model(BEST_PATH)
    
