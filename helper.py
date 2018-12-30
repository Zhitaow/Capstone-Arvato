# import libraries here; add more as necessary
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from datetime import datetime
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVR
import imblearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
# define some helper functions below.

def flag_unknown(x):
    '''
    Input:
        x: (string) 
    Output:
        return 1 if the input string contains keywords in the data dictionary, otherwise return 0
    '''
    data_dict = set(['unknown / no main age detectable','no transaction known','no transactions known', 'unknown'])
    for key in data_dict:
        if key in x:
            return 1
    return 0
# imputation function replace the NaNs in selected column by its most frequent value

def impute_na(df, variable):
    '''
    input: 
            df: input dataframe to be imputed
            variable: the column from the dataframe you want to perform imputation
    '''
    # find most frequent category
    most_frequent_category = df.groupby([variable])[variable].count().sort_values(ascending=False).index[0]
    # replace NA
    df[variable].fillna(most_frequent_category, inplace=True)

    
def sanity_check(df):
    '''
        Usage: check data dimension, data types, number of missing values after cleaning
    '''
    print("\n Sanity Check: The dimension of the dataset is: {}".format(df.shape))
    dtype_count = 0
    for i, col in enumerate(df.columns):
        # check if the columns are numeric type
        if (is_numeric_dtype(df[col]) is False):
            print("Warning: column index {}, column name {} is not numeric dtype!".format(i, col))
            dtype_count += 1
        # check if each column contains missing values
    nan_cols = df.columns[df.isnull().any()].tolist()
    nan_count = len(nan_cols)
    if nan_count > 0:
        print("Warning: the following columns contain missing values: \n {}".format(nan_cols))
    print("There are {} columns of non-numeric dtype.".format(dtype_count))
    print("There are {} columns containing missing values.".format(nan_count))
    
# re-engineer helper function

def feature_decade(x):
    '''
    input: integer
    return: 1 stands for 40s, 2 for 50s, 3 for 60s, 4 for 70s, 5 for 80s and 6 for 90s
    '''
    if x >= 1 and x <= 2:
        return 1
    elif x >= 3 and x <= 4:
        return 2
    elif x >= 5 and x <= 7:
        return 3
    elif x >= 8 and x <= 9:
        return 4
    elif x >= 10 and x <= 13:
        return 5
    elif x >= 14 and x <= 15:
        return 6
    else:
        return np.nan

# re-engineer helper function

def feature_movement(x):
    '''
    input: integer
    return: 0 for mainstream, and 1 for avantgarde
    '''
    if x in set([1, 3, 5, 8, 10, 12, 14]):
        return 0
    elif x in set([2, 4, 6, 7, 9, 11, 13, 15]):
        return 1
    else:
        return np.nan

def feature_neighborhood(x):
    '''
    1-5 = 1-5, 7 and 8 are mapped to np.nan
    '''
    if x <= 5:
        return x
    else:
        return np.nan

def feature_rural(x):
    '''
    1-5 = not rural, return as 0
    7-8 = rural, return as 1
    '''
    if x <= 5:
        return 0
    elif (x == 7) | (x == 8):
        return 1
    else:
        return np.nan
    
def feature_homes(x):
    '''
    1-4 return as 1-4 (homes)
    5 return as np.nan 
    '''
    if x <= 4:
        return x
    else:
        return np.nan

def feature_business(x):
    '''
    1-4 return 0 (non-business)
    5 return 1(business)
    '''
    if x <= 4:
        return 0
    elif x == 5:
        return 1
    else:
        return np.nan

def feature_neighborhood_development(x):
    '''
    1-4 return as it is
    5 return np.nan
    '''
    if x <= 4:
        return x
    else:
        return np.nan

def feature_new_building_flag(x):
    '''
    1-4 flag as 0
    5 flag as 1
    '''
    if x <= 4:
        return 0
    elif x ==5:
        return 1
    else:
        return np.nan

def convert_nan(df, df_dict, verbose = False):
    '''
    Input: df - (dataframe) dataset where missing codes are to be converted to np.nan
           df_dict - (dataframe) data dictionary indicating the features and their corresponding missing codes
           feat_idx - the index in reference to the features in the data dictionary
           value_idx - the index in reference to the missing values in the data dictionary
           
    Output: None
    '''
    count = 0
    for row in df_dict.iterrows():
        feat = row[1][0]
        nans = row[1][1]
        if feat in df.columns:
            #print('Converting missing value {} in feature {}'.format(nans, feat))
            count += 1
            if isinstance(nans, int) or isinstance(nans, float):
                df[feat] = df[feat].apply(lambda x: np.nan if x == nans else x)
            else:
                df[feat] = df[feat].apply(lambda x: np.nan if x in nans else x)
    if verbose:
        print('Converted {} missing codes to np.nan'.format(count))

def reshape_cols(df, columns):
    '''
    Input:
        df - (DataFrame) after clean_data
        columns - (list) of reference columns
    Output:
        df_out - (DataFrame) where columns are missing from df will be added based on the referenced columns. These added columns are impuated with 0s.
    '''
    zeros = np.zeros([df.shape[0], len(columns)])
    df_out = pd.DataFrame(data=zeros, index=df.index, columns=columns)
    for idx, col in enumerate(df.columns):
        if col in columns:
            df_out[col] = df[col]
    return df_out


# define helper function to train kmean model
def train_kmean(data, clusters, use_batch = True, batch_portion = 0.2, print_every = 5):
    '''
        clusters: the number of clusters used to fit the data
        use_batch: if true, will use mini-batch kmean
        batch_portion: the portion of batch size
        return: list of number of kmean clusters, and the corresponding scores
    '''
    centers = []
    scores = []
    batch_size = int(data.shape[0]*batch_portion)
    print("Input data dimension: ", data.shape)
    if use_batch:
        print("Using minibatch Kmean.")
    else:
        print("Using Kmean.")
    for i, n_clusters in enumerate(clusters):
        # run k-means clustering on the data and...
        if use_batch:
            km = MiniBatchKMeans(n_clusters = n_clusters, batch_size = batch_size)
        else:
            km = KMeans(n_clusters = n_clusters)
        model = km.fit(data) 
        # compute the average within-cluster distances.
        score = np.abs(model.score(data))
        centers.append(n_clusters)
        scores.append(score)
        if i % print_every == 0:
            print("The score for {} clusters is {} ".format(n_clusters, score))
    return centers, scores

# Investigate the change in within-cluster distance across number of clusters.
def plot_kmean_score(centers, scores): 
    '''
    Input: centers - list of cluster number
           scores - list of corresponding kmean scores
    '''
    fig = plt.figure(figsize = (16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(centers, scores, linestyle='--', marker='o', color='b')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('SSE vs. Number of Clusters')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(centers, np.gradient(scores))
    plt.xlabel('Number of Clusters')
    plt.ylabel('First Derivative of SSE')
    plt.title('SSE Slope vs. Number of Clusters')

def predict_cluster(df_processed, scaler, pca, model):
    '''
    Input: df_processed - (DataFrame) after runing clean_data
           scaler - (object) of standard scaler fit from the training data
           pca - (object) of pca fit from the training data
           model - (object) of trained kmean model
    Output: df_label - (DataFrame) of cluster IDs
    '''
    df_processed_ = df_processed.copy()
    LNR = np.zeros(df_processed_.shape[0])
    if 'LNR' in df_processed_.columns:
        LNR = df_processed_.LNR.values
        df_processed_.drop(columns='LNR', inplace = True)

    # apply feature scaling using the fitted scaler
    scaled_arr = scaler.transform(df_processed_)
    # convert from ndarray to dataframe object
    df_scaled = pd.DataFrame(data = scaled_arr, index = df_processed_.index, columns = df_processed_.columns)
    #print("The dataset's dimension after standard scaling: {} \n".format(df_scaled.shape))
    # apply PCA transformation
    df_scaled_pca = pca.transform(df_scaled)
    # predict the customer cluster using the fitted model
    df_label = pd.DataFrame({"LNR": LNR, "Cluster_ID": model.predict(df_scaled_pca)})
    df_label.index = df_processed_.index
    return df_label

def segment_plot(df1, df2, df1_cluster_ID, df2_cluster_ID, labels = ['Demongraphic', 'Customer'], normalized = True):
    '''
    Input: df1, df2 - (DataFrames) to be compared (features should be exactly the same)
           df1_cluster_ID, df2_cluster_ID - (DataFrames) of predicted cluster IDs for df1 and df2
           labels - (list) of names for the two data sources
    '''
    # data preparation
    n_clusters = df1_cluster_ID.Cluster_ID.max()+1
    df1 = pd.DataFrame(data=np.zeros((df1.shape[0], 3)), index = df1.index, columns = ['Cluster_ID', 'label', 'Count'])
    df2 = pd.DataFrame(data=np.zeros((df2.shape[0], 3)), index = df2.index, columns = ['Cluster_ID', 'label', 'Count'])
    df1['label'] = labels[0]
    df2['label'] = labels[1]
    df1['Cluster_ID'] = df1_cluster_ID['Cluster_ID']
    df1['Cluster_ID'].fillna(value = n_clusters, inplace = True)
    df2['Cluster_ID'] = df2_cluster_ID['Cluster_ID']
    df2['Cluster_ID'].fillna(value = n_clusters, inplace = True)
    df = pd.concat([df1, df2])
    df_agg = df.groupby(['label', 'Cluster_ID']).count()
    df_agg.reset_index(level=['label','Cluster_ID'], inplace = True)
    df_agg['Frequency'] = 0
    for label in df.label.unique():
        df_agg.loc[df_agg.label == label, 'Frequency'] = \
            df_agg[df_agg.label == label]['Count'].transform(lambda x: (x) / x.sum()*100)
    # prepare for plotting
    fig = plt.figure(figsize = (15,8))
    if normalized:
        ax = sns.barplot(x='Cluster_ID', y="Frequency", hue='label', data=df_agg)
    else:
        ax = sns.barplot(x='Cluster_ID', y="Count", hue='label', data=df_agg)
    ax.set_xlabel("Cluster ID", fontsize = 15)
    ax.set_ylabel("Percentage (%)", fontsize = 15)
    #ax.set_title(column, fontsize = 20)
    ax.grid()
    plt.annotate("Note: Cluster {} represents high-NaN data".format(n_clusters), \
            xy=(0.35, 1.02), xycoords='axes fraction', fontsize = 15)
    return df_agg



# count plots
def count_plot(data, features, hue, normalized = False):
    '''
        Usage: show feature-based distribution
        Input: data - (DataFrame) aggreated result from step 0
               features - (list) of features to be plotted
               hue - (string) of label
               normalized - (boolean) show normalized distribution if set True, otherwise only the countplot
    '''
    nrow = 3
    ncol = int(len(features)/3)
    fig = plt.figure(figsize = (20, 20))
    if normalized is False:
        for i in np.arange(len(features)):
            ax = fig.add_subplot(ncol, nrow, i+1)
            ax = sns.countplot(x = features[i], hue = hue, data = data, ax = ax)
    else:
        for i in np.arange(len(features)): 
            ax = fig.add_subplot(ncol, nrow, i+1)
            data_ = (data.groupby([hue])[features[i]]
                             .value_counts(normalize=True)
                             .rename('percentage')
                             .mul(100)
                             .reset_index()
                             .sort_values(features[i]))
            ax = sns.barplot(x=features[i], y="percentage", hue=hue, data=data_, ax = ax)
    return ax

def scree_plot(pca, var_thres = 90):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
           var_thres - percent explained variance threshold
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    idx = np.arange(num_components)
    vals = pca.explained_variance_ratio_ * 100
    cumvals = np.cumsum(vals)
    # index of component where the culmulative sum of variance reaches the threshold
    plt_idx = np.argmin(np.abs(var_thres - cumvals))
    print("Cut point of {}% explained variance is at the {}-th component".format(var_thres,plt_idx+1))
    
    fig = plt.figure(figsize = (12, 15))
    
    # plot explained variance
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.bar(idx, vals)
    ax1.grid()
    y_vline = [np.min(vals), np.max(vals)]
    x_vline = np.ones(len(y_vline))*plt_idx
    plt.plot(x_vline, y_vline, 'r--')
      
    # plot culmulative explained variance
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.bar(idx, cumvals)
    ax2.grid()
    # threshold horizontal line
    h_line = np.ones(len(idx))*var_thres
    # threshold vertical line
    y_vline = [0, 100]
    x_vline = np.ones(len(y_vline))*plt_idx
    plt.plot(idx, h_line, 'r--')
    plt.plot(x_vline, y_vline, 'r--')
       
    # plot setting 
    ax1.xaxis.set_tick_params(width=0)
    ax1.yaxis.set_tick_params(width=2, length=12)
    ax1.set_xlim(0, len(idx))
    ax1.set_ylim(np.min(vals), np.max(vals))
    ax1.set_xlabel("Principal Component", fontsize = 12)
    ax1.set_ylabel("Explained Variance (%)", fontsize = 12)
    ax1.set_title('Explained Variance Per Principal Component', fontsize = 12)
    ax2.xaxis.set_tick_params(width=0)
    ax2.yaxis.set_tick_params(width=2, length=12)
    ax2.set_xlim(0, len(idx))
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Principal Component", fontsize = 12)
    ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize = 12)
    ax2.set_title('Culmulative Explained Variance vs. Number of Component', fontsize = 12)
    
def pca_weight_decomp(full_dataset, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''
    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained_Variance_Ratio'])
    variance_ratios.index = dimensions

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

def show_pca_weight(df, topk_components = 3, topk_features = 10, kth_component = None, figsize = (16, 16), plot = True, print_screen = False):
    '''
    Input: df - (DataFrame) after runing pca_weight_decomp
           topk_components - (int) number of components to be plotted
           topk_featuress - (int) number of top k features to be plotted
           kth_component - (int) if specified, only plot the kth component
           figsize - size of figure
           plot - (boolean) show plot if set True
           print_screen - (boolean) print message if set True
    Output:
           df - (DataFrame) of feature importance
    '''
    variance_ratios = df['Explained_Variance_Ratio']
    components = df.drop('Explained_Variance_Ratio', axis = 1)
    dimensions = df.index
    components_T = components.T
    features = components_T.index
    result = []
    # Plot the feature weights as a function of the components
    for i, column in enumerate(components_T.columns):
        if i < topk_components:    
            feature_weights = components_T[column].values 
            idx = np.argsort(-np.abs(feature_weights))
            feature_weights = feature_weights[idx][0:topk_features]
            feature_names = features[idx][0:topk_features]
            for j, (feature_name, feature_weight) in enumerate(zip(feature_names, feature_weights)):
                result.append([column, feature_name, feature_weight])
                if print_screen:
                    print("{}: feature: {} weight: {}".format(column, feature_name, feature_weight))
            
            my_colors = [(x/len(feature_weights)/2, x/len(feature_weights), 0.75) \
                         for x in range(len(feature_weights))]
            y_pos = np.arange(len(feature_names))
            if plot:
                fig = plt.figure(figsize = figsize)
                if kth_component is None:
                    ax = fig.add_subplot(topk_components, 1, i+1)
                    plt.barh(y_pos, feature_weights[::-1], color = my_colors)
                    plt.yticks(y_pos, feature_names[::-1])
                    ax.set_title(column, fontsize = 20)
                    ax.grid()
                elif i == kth_component - 1:
                    ax = fig.add_subplot(1, 1, 1)
                    plt.barh(y_pos, feature_weights[::-1], color = my_colors)
                    plt.yticks(y_pos, feature_names[::-1])
                    ax.set_title(column, fontsize = 20)
                    ax.grid()
    df = pd.DataFrame(result)
    df.columns = ['Dimension', 'feature', 'PCA_Weight']
    if kth_component is not None:
        df = df[df['Dimension'] == 'Dimension ' + str(int(kth_component))]
    return df

# What kinds of people are part of a cluster that is overrepresented/underrepresented in the
# customer data compared to the general population?

def show_attributes(cluster_over, cluster_under):
    '''
    Input: 
            cluster_over - (int) id of over-representing cluster
            cluster_under - (int) id of under-representing cluster
    Return:
            df_summary - (DataFrame) of average values of selected feature
    '''
    try:
        # restore saved models
        # load clustering model obtained from Step 1
        km = joblib.load("savefile/kmeans_model.save")
        # load standard scaler obtained from step 1
        km_scaler = joblib.load("savefile/standard_scaler.save") 
        # load pca obtained from step 1
        km_pca = joblib.load("savefile/pca.save")
        # load list of all column names after cleaning azdias data in step 1 
        azdias_processed_columns = list(pd.read_csv('savefile/all_columns.csv').drop('Unnamed: 0', axis = 1).iloc[:,0])
        # select one overrepresenting cluster, and one underrepresenting cluster
        centroid_id = [cluster_over, cluster_under]
        centroids = []
        cluster_center = km.cluster_centers_[centroid_id,:]
        df_result = pd.DataFrame(data = km_scaler.inverse_transform(km_pca.inverse_transform(cluster_center)), \
            columns = azdias_processed_columns[1:], index = ['Cluster 14 (overrepresenting)', 'Cluster 5 (underrepresenting)'])
        # income-related features
        income_col = ['HH_EINKOMMEN_SCORE','LP_STATUS_GROB_1.0','LP_STATUS_GROB_2.0',\
                    'LP_STATUS_GROB_3.0','LP_STATUS_GROB_4.0','LP_STATUS_GROB_5.0',\
                    'FINANZ_MINIMALIST', 'WEALTH']
        # age-related features
        age_col = ['ALTERSKATEGORIE_GROB', 'DECADE']
        # gender-related features
        gender_col = ['ANREDE_KZ_1','ANREDE_KZ_2']
        # movement features
        move_col = ['GREEN_AVANTGARDE_0','GREEN_AVANTGARDE_1', 'MOVEMENT']
        # building location
        location_col = ['OST_WEST_KZ_O','OST_WEST_KZ_W']

        features = income_col + age_col + gender_col + move_col + location_col

        parents = ['Income' for i in np.arange(len(income_col))] + \
                ['Age' for i in np.arange(len(age_col))] + \
                ['Gender' for i in np.arange(len(gender_col))] + \
                ['Movement' for i in np.arange(len(move_col))] + \
                ['Apartment Location' for i in np.arange(len(location_col))]

        meaning = ['Estimated household net income', 'low-income earners', 'average earners', \
                   'independents', 'houseowners', 'top earners', 'low financial interest', 'Wealth Scale',\
                  'Estimated age', 'Borned Decades', 'Male', 'Female', \
                   'mainstream', 'avantgarde', 'mainstream vs. avantgarde', 'East Germany', 'West Germany']
        df_summary = pd.DataFrame(df_result[features].T.values, \
                          index=[parents, features, meaning], columns = df_result.index)
    except:
        print("Fail to load save models. Skip generating data exhibit.")
        df_summary = None
    return df_summary

def plot_feature_importances(df, model, plot = True):
    '''
    Input: df - (DataFrame) after clean_data
           model - (object) of trained tree-based model
    Output:
           tree_result - (DataFrame) of feature importance
    '''
    importances = model.feature_importances_
    feat_names = df.columns
    tree_result = pd.DataFrame({'feature': feat_names, 'importance': importances})
    figsize= (8, 8)
    if plot:
        tree_result.sort_values(by='importance',ascending=True)[-20:].plot(x='feature', y='importance', kind='barh', figsize = figsize)
    return tree_result

def plot_class_dist(y_true, y_pred, ax = None, title = '', range = None, plot = True):
    '''
    Input: y_true - (DataFrame) of actual label
           y_pred - (DataFrame) of predicted probability
           range - (list) lower and upper bound of x-axis range
           title - (string) title name
    Return:
           auc_score - (float) of AUC score
    '''
    result = y_true[['RESPONSE']].copy()
    result['Pred'] = y_pred
    #print(mean_0, mean_1)
    if plot:
        ax = result.hist(column='Pred', by='RESPONSE', bins = 20, sharex=True, range=range, layout = [1,2], figsize=(10, 3))
        ax[0].set_title(title + ': 0')
        ax[1].set_title(title + ': 1')
    try:
        auc_score= roc_auc_score(y_true['RESPONSE'], y_pred)
        print(title + ' AUC Score: ', auc_score)
    except:
        auc_score = None
    return auc_score

def plot_class_by_cluster(data, bins = 20, plot = True):
    '''
    plot class distribution based on each cluster ID
    '''
    auc_scores = []
    clusters = []
    count0s = []
    count1s = []
    for icluster in np.arange(data.Cluster_ID.nunique()):
        data_subset = data[data['Cluster_ID'] == icluster]
        count0 = data_subset[data_subset.iloc[:, 1] == 0].shape[0]
        count1 = data_subset[data_subset.iloc[:, 1] == 1].shape[0]
        count0s.append(count0)
        count1s.append(count1)            
        try:
            auc_score = roc_auc_score(data_subset.iloc[:,1], data_subset.iloc[:,2])
        except ValueError:
            auc_score = np.nan
        auc_scores.append(auc_score)
        clusters.append(icluster)
        if plot:
            axes = data_subset.hist(column = data_subset.columns[2], by = data_subset.columns[1], bins = bins, figsize = (8, 2), range = (0, 1), layout = [1,2])
            if isinstance(axes,np.ndarray):
                for ax in axes:
                    title = ax.get_title()
                    new_title = 'Cluster: '+ str(icluster) +' Response: '+ str(title)
                    ax.set_title(new_title)
            else:
                title = axes.get_title()
                new_title = 'Cluster: '+ str(icluster) +' Response: '+ str(title)
                axes.set_title(new_title)
    df_auc = pd.DataFrame(clusters, columns = ['Cluster_ID'])
    df_auc['AUC'] = auc_scores
    df_auc['Count: 0'] = count0s
    df_auc['Count: 1'] = count1s
    return df_auc

# convert NaN codes to np.nan
def convert_nan(df, feat_nan_code):
    '''
    Input: df - (DataFrame) to be converted to np.nan
           feat_nan_code - (DataFrame) of NaN codes
    Output: df - (DataFrame) where missing codes were converted to np.nan
    '''
    for i in range(feat_nan_code.shape[0]):
        feature = feat_nan_code.iloc[i, 0]
        item = feat_nan_code.iloc[i, 1]
        if feature in df.columns:
            if isinstance(item, list) or isinstance(item, tuple):
                df[feature] = df[feature].apply(lambda x: np.nan if x in item else x)
                #print(i, feature, item)
            elif isinstance(item, int) or instance(item, float):
                df[feature] = df[feature].apply(lambda x: np.nan if x == item else x)
                #print(i, feature, item)
    return

'''
    You can skip running the previous cells in this notebook,
    except for the first cell to import the libaries and helper function
'''

#azdias, feat_nan_code, feat_cat, feature_to_drop = feature_to_drop, row_thres = 30
def clean_data(df, df_feat_nan, feat_cat, feature_to_drop, col_thres = None, row_thres = 30, drop_row_NaN = True, print_step = True):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: 
          df - (DataFrame) to be cleaned
          df_feat_nan - (DataFrame) codes of missing value
          feat_cat - (DataFrame) list of documented feature type
          feature_to_drop - (DataFrame) list of feature to be dropped out
          col_thres - (float) threshold percentage of column missing value. If it is set None, use feature_to_drop instead
          row_thres - (float) threshold percentage of row missing value
          drop_row_NaN - (boolean) drop data rows above row_thres if it is set True, otherwise do nothing.
          print_step - (boolean) print step-wise message if set to be True.
    OUTPUT: 
          data - (DataFrame) cleaned data (row-missing data will be dropped if drop_row_NaN is True)
          data_ - (DataFrame) uncleaned data above row-missing threshold
    """
    # default threshold for number of missing values in each column and row
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    #if 'LNR' in df.columns:
    #    data = df.drop(columns = 'LNR').copy()
    #else:
    #    data = df.copy()
    data = df.copy()
    # convert NaN codes to np.nan
    convert_nan(data, df_feat_nan)
    # Convert row input strings to float
    data[['CAMEO_DEUG_2015','CAMEO_INTL_2015']] = data[['CAMEO_DEUG_2015','CAMEO_INTL_2015']].astype(float)
    ################################# remove high-NaN columns  #########################################
    #if auto_select:
    if col_thres is None:
        #feature_to_drop = ['TITEL_KZ', 'AGER_TYP', 'KK_KUNDENTYP', 'KBA05_BAUMAX', 'GEBURTSJAHR', 'ALTER_HH']
        feature_to_drop = list(feature_to_drop.Feature)
    else:
        ls = []
        #calculate percentage of NaN in each data column
        for i, column in enumerate(data.columns):
            count = data[column].isnull().sum(axis=0)
            percent = count/data.shape[0]
            ls.append([column, count, percent*100])

        data_summary = pd.DataFrame.from_records(ls, columns = ['Feature', \
                'NaN Count', 'NaN Occupancy Rate']).sort_values('NaN Occupancy Rate',axis=0,ascending = False)

        feature_to_drop = data_summary[data_summary['NaN Occupancy Rate'] > col_thres].Feature.values.tolist()
        
    if print_step:
        print("\n Step 1: drop the following features with high NaN occupancy rate above {}%: \n {}".format(\
            col_thres, feature_to_drop))
    
    
    data.drop(feature_to_drop, axis=1, inplace = True)
    
    if print_step:
        print("\n {} features have been dropped. The new dataset dimension after Step 1 is: {}".format(\
            len(feature_to_drop), data.shape))
    
    ######################################## remove high NaN rows #########################################
    # remove selected columns and rows, ...
    if print_step:
        print("\n Step 2: drop rows with high NaN occupancy rate above {}%... \n".format(row_thres))
        
    ncol = data.shape[1]
    idx = data.isnull().sum(axis=1)/ncol*100 <= row_thres
    data_ = data[~idx]
    if drop_row_NaN:
        data = data[idx]
        
    if print_step:
        print("\n {} of rows have been dropped. The new dataset dimension after Step 2 is: {}".format(\
            (idx==0).sum(), data.shape))
    
    idx_ = data.isnull().sum(axis=1) == 0 
    nrow_nan = (idx_==0).sum()
    
    if print_step:
        print("\n After step 2, there are {} rows left with missing values,"+\
              " consisting of {}% of the total population".format(nrow_nan, nrow_nan/data.shape[0]*100))
        nan_series = (data.isnull().sum()/data.shape[0]*100).sort_values(axis=0,ascending = False)
        nan_cols = list(nan_series.index)
        nan_pcts = nan_series.tolist()
        for i, (nan_col, nan_pct) in enumerate(zip(nan_cols, nan_pcts)):
            if i < 10:
                print('Feature "{}" has {}% missing values'.format(nan_col, nan_pct))
            else:
                break

    # select, re-encode, and engineer column values.
    categorical_feat_list = feat_cat[feat_cat['Type'] == 'categorical']['Attribute'].tolist()
    # list of categorical features that we have dropped in previous step
    not_found_features = set(categorical_feat_list) - set(data.columns)
    categorical_dummy = [x for x in categorical_feat_list \
                         if (x not in feature_to_drop) and (x not in not_found_features)]
    #categorical_dummy = [x for x in categorical_feat_list if x not in feature_to_drop]
    if print_step:
        print("Convert the dummy variables from these features: {}".format(categorical_dummy))
    # list of columns with missing values:
    nan_cols = data.columns[data.isnull().any()].tolist()
    
    #print("\n There is a total of {} NaN values in {} columns.".format(data.isnull().sum(), len(nan_cols)))
    if print_step:
        print("\n Step 3: replace all NaNs in each column by its corresponding mode.")
    
    # impute the most frequent value for the missing data in each column
    for col in nan_cols:
        impute_na(data, col)
    
    # sanity check: there should be no missing values in remain
    nan_ncols = len(data.columns[data.isnull().any()].tolist())
    nan_count = np.count_nonzero(data.isnull().values)
    total_count = data.shape[0]*data.shape[1]
    nan_percent = nan_count/total_count*100
    if print_step:
        print("\n After Step 3, there are {} rows with NaN values left,"+\
              " {}% of total population, in {} columns.".format(nan_count, nan_percent, nan_ncols))
    
    ###################################### perform feature engineering ###################################
    # convert categorical features to dummy variables
    if print_step:
        print("\n Step 4: create dummy variables from the categorical features:{}".format(categorical_dummy))
    data = pd.get_dummies(data, prefix = categorical_dummy, columns = categorical_dummy)
    if print_step:
        print("\n The new dataset dimension after Step 4 is: {}".format(data.shape))
        print("\n Step 5: Engineer Features")
        
    # Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
    if 'PRAEGENDE_JUGENDJAHRE' in data.columns:
        data['DECADE'] = data['PRAEGENDE_JUGENDJAHRE'].apply(lambda x: feature_decade(x))
        data['MOVEMENT'] = data['PRAEGENDE_JUGENDJAHRE'].apply(lambda x: feature_movement(x))
        data.drop(columns = 'PRAEGENDE_JUGENDJAHRE', inplace = True)
        
    # Investigate "CAMEO_INTL_2015" and engineer two new variables.
    if 'CAMEO_INTL_2015' in data.columns:
        data['WEALTH'] = data['CAMEO_INTL_2015'].astype(float).apply(lambda x: np.floor(x / 10))
        data['LIFE_STAGE'] = data['CAMEO_INTL_2015'].astype(float).apply(lambda x: (x % 10))
        data.drop(columns = 'CAMEO_INTL_2015', inplace = True)
        
    # Investigate "WOHNLAGE" and engineer two new variables.
    if 'WOHNLAGE' in data.columns:
        data['NEIGHBORHOOD'] = data['WOHNLAGE'].astype(float).apply(lambda x: feature_neighborhood(x))
        data['RURAL_FLAG'] = data['WOHNLAGE'].astype(float).apply(lambda x: feature_rural(x))
        impute_na(data, 'NEIGHBORHOOD')
        impute_na(data, 'RURAL_FLAG')
        data.drop(columns = 'WOHNLAGE', inplace = True)
        
    # Investigate "PLZ8_BAUMAX" and engineer two new variables.
    if 'PLZ8_BAUMAX' in data.columns:
        data['PLZ8_HOMES'] = data['PLZ8_BAUMAX'].astype(float).apply(lambda x: feature_homes(x))
        data['PLZ8_BUSINESS'] = data['PLZ8_BAUMAX'].astype(float).apply(lambda x: feature_business(x))
        impute_na(data, 'PLZ8_HOMES')
        impute_na(data, 'PLZ8_BUSINESS')
        data.drop(columns = 'PLZ8_BAUMAX', inplace = True)

    # Investigate KBA05_HERSTTEMP, and engineer one ordinal variable, and one binary categorical variable respectively.
    if 'KBA05_HERSTTEMP' in data.columns:
        data['KBA05_HERSTTEMP_NEIGHBORHOOD_DEV'] = data['KBA05_HERSTTEMP'].astype(float).apply(lambda x: feature_neighborhood_development(x))
        data['KBA05_HERSTTEMP_NB_FLG'] = data['KBA05_HERSTTEMP'].astype(float).apply(lambda x: feature_new_building_flag(x))
        impute_na(data, 'KBA05_HERSTTEMP_NEIGHBORHOOD_DEV')
        impute_na(data, 'KBA05_HERSTTEMP_NB_FLG')
        data.drop(columns = 'KBA05_HERSTTEMP', inplace = True)
        
    # Investigate KBA05_HERSTTEMP, and engineer one ordinal variable, and one binary categorical variable respectively.
    if 'KBA05_MODTEMP' in data.columns:
        data['KBA05_MODTEMP_NEIGHBORHOOD_DEV'] = data['KBA05_MODTEMP'].astype(float).apply(lambda x: feature_neighborhood_development(x))
        data['KBA05_MODTEMP_NB_FLG'] = data['KBA05_MODTEMP'].astype(float).apply(lambda x: feature_new_building_flag(x))
        impute_na(data, 'KBA05_MODTEMP_NEIGHBORHOOD_DEV')
        impute_na(data, 'KBA05_MODTEMP_NB_FLG')    
        data.drop(columns = 'KBA05_MODTEMP', inplace = True)
        
    # engineer year variable from EINGEFUEGT_AM
    if 'EINGEFUEGT_AM' in data.columns:
        data['EINGEFUEGT_AM_YEAR'] = data['EINGEFUEGT_AM'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).year)
        data['EINGEFUEGT_AM_MONTH'] = data['EINGEFUEGT_AM'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S" ).month)
        data.drop(columns = 'EINGEFUEGT_AM', inplace = True)
        
    # create dummy variables for D19_LETZTER_KAUF_BRANCHE
    if 'D19_LETZTER_KAUF_BRANCHE' in data.columns:
        dummies = pd.get_dummies(data['D19_LETZTER_KAUF_BRANCHE'], prefix = 'D19_LETZTER_KAUF_BRANCHE')
        data = pd.concat([data, dummies], axis = 1)
        data.drop(columns = 'D19_LETZTER_KAUF_BRANCHE', inplace = True)
        
    # create dummy variables for D19_KONSUMTYP_MAX
    if 'D19_KONSUMTYP_MAX' in data.columns:
        dummies = pd.get_dummies(data['D19_KONSUMTYP_MAX'], prefix = 'D19_KONSUMTYP_MAX')
        data = pd.concat([data, dummies], axis = 1)
        data.drop(columns = 'D19_KONSUMTYP_MAX', inplace = True)
        
    # Drop the four original features
    if 'LP_LEBENSPHASE_FEIN' in data.columns:
        data.drop(columns = 'LP_LEBENSPHASE_FEIN', inplace = True)
        
    if 'LP_LEBENSPHASE_GROB' in data.columns:
        data.drop(columns = 'LP_LEBENSPHASE_GROB', inplace = True)
        
    print("\n The new dataset dimension is: {}".format(data.shape))
    
    data = data.astype(float)
    data.reset_index(drop = True, inplace = True)
    data_.reset_index(drop = True, inplace = True)
    # perform sanity check
    sanity_check(data)
        
    # Return the cleaned dataframe.
    return data, data_

def drop_columns(X, columns):
    X_out = X.copy()
    for column in columns:
        if column in X_out.columns:
            X_out.drop(columns=column, inplace = True)
    return X_out

def my_gridsearch(estimator, param_grid, X_train, y_train, X_test, y_test, n_jobs = 2, cv = 5, plot = True):
    '''
    Tasks: 1. use gridsearchCV to find the optimal estimator
           2. obtain predicted result of trainset and testset with the optimal estimator
           3. estimate the AUC for both trainset and testset
           4. show feature importance, AUCs.
    Input: estimator - (object) input ML model
           param_grid - (dict) model parameters
           X_train, y_train - (DataFrame) of training data
           y_test, y_test - (DataFrame) of testing data
           plot - (boolean) option to show plot on the go
    Output:
           best_estimator - (object) the estimator with the optimal training score
           pred_result - (list) two DataFrames: predicted normalized probabilities of trainset and testset
           auc_score - (list) two AUC scores of trainset and testset
           feature_importances - (DataFrame) feature importance of the best estimator
    '''
    start_time = time.time()
    # Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(estimator = estimator, param_grid = param_grid, scoring='roc_auc', n_jobs=n_jobs, cv=cv)
    # fit the grid search object to the training data and find the optimal parameters
    X_train_drop = drop_columns(X_train, ['LNR', 'Cluster_ID'])
    X_test_drop = drop_columns(X_test, ['LNR', 'Cluster_ID'])

    grid_obj.fit(X_train_drop, y_train['RESPONSE'])
    print('#',grid_obj.best_params_, grid_obj.best_score_)
    # get the estimator and predict the class
    best_estimator = grid_obj.best_estimator_
    y_train_pred = best_estimator.predict(X_train_drop)
    y_test_pred = best_estimator.predict(X_test_drop)
    # evaluate the AUC
    auc_score_train = plot_class_dist(y_train, y_train_pred, title = 'Trainset ', plot = plot)
    auc_score_test = plot_class_dist(y_test, y_test_pred, title = 'Testset ', plot = plot)
    # show the feature importance
    try:
        feature_importances = plot_feature_importances(X_train_drop, best_estimator, plot = plot)
    except:
        feature_importances = None
    # format result
    df_y_train_pred = y_train.copy()
    df_y_train_pred['RESPONSE'] = y_train_pred
    df_y_test_pred = y_test.copy()
    df_y_test_pred['RESPONSE'] = y_test_pred
    pred_result = [df_y_train_pred, df_y_test_pred]
    auc_score = [auc_score_train, auc_score_test]
    elapsed_time = time.time() - start_time
    print("Training Time: {} min".format(elapsed_time/60.0))
    return best_estimator, pred_result, auc_score, feature_importances

def normal_prob(y_train_pred, y_test_pred):
    '''
    Usage: normalize y_train_pred and y_test_pred using baseline of the min and max from y_train_pred
    Input: 
           y_train_pred - (np.array) predicted relative probability of training data
           y_test_pred - (np.array) predicted realative probability of testing data
    Output:
           y_train_pred_norm - (np.array) predicted normalized probability of training data (ranged 0-1)
           y_test_pred_norm - (np.array) predicted normalized probability of testing data (ranged 0-1)
    '''
    y_train_max = np.max(y_train_pred)
    y_train_min = np.min(y_train_pred)
    y_train_pred_norm = (y_train_pred - y_train_min)/(y_train_max - y_train_min)
    y_test_pred_norm = (y_test_pred - y_train_min)/(y_train_max - y_train_min)
    return y_train_pred_norm, y_test_pred_norm

def norm_prob(y, yrange):
    '''
    scale y into range [0,1] based on yrange
    '''
    return (y-yrange[0])/(yrange[1]-yrange[0])

def train_test_split_by_cluster(X, y, test_size = 0.2, random_state = 1):
    '''
    Usage: split training and testing set evenly based on Cluster_ID
    Input: X - (DataFrame)
           y - (DataFrame)
    Output:
           X_train, X_test, y_train, y_test - (DataFrame)
    '''
    n_clusters = X.Cluster_ID.nunique()
    X_train = pd.DataFrame(data=None, columns = X.columns)
    X_test = pd.DataFrame(data=None, columns = X.columns)
    y_train = pd.DataFrame(data=None, columns = y.columns)
    y_test = pd.DataFrame(data=None, columns = y.columns)
    for Cluster_ID in range(n_clusters):
        X_subset = X[X['Cluster_ID'] == Cluster_ID]
        y_subset = y[y['LNR'].isin(X_subset['LNR'])]
        X_train_subset, X_test_subset, y_train_subset, y_test_subset = \
            train_test_split(X_subset, y_subset, test_size=test_size, stratify = y_subset.RESPONSE, random_state=random_state)
        X_train = pd.concat([X_train, X_train_subset])
        X_test = pd.concat([X_test, X_test_subset])
        y_train = pd.concat([y_train, y_train_subset])
        y_test = pd.concat([y_test, y_test_subset])
    return X_train, X_test, y_train, y_test

def oversampling(train_set, Cluster_ID):
    '''
    Usage: Perform oversampling on dataset with specific Cluster_ID
    Input: 
           train_set - (DataFrame) X and y
           Cluster_ID - (float) Cluster_ID to be oversampled
    '''
    train_cluster_set = train_set[train_set['Cluster_ID'] == Cluster_ID]
    # Class count
    count_class_0, count_class_1 = train_cluster_set['RESPONSE'].value_counts()
    # Divide by class
    train_class_0 = train_cluster_set[train_cluster_set['RESPONSE'] == 0]
    train_class_1 = train_cluster_set[train_cluster_set['RESPONSE'] == 1]
    # Over-sampling
    train_class_1_over = train_class_1.sample(count_class_0, replace=True)
    # merge over-sampled class-1 with class-0 
    train_over = pd.concat([train_class_0, train_class_1_over])
    # shuffle the data
    train_over = train_over.sample(frac=1)
    return train_over

def models_predict(models, X, y, columns = None):
    '''
    Return the average probability averaged from the models
    Input:
        models - (list) of ML models
        X - (DataFrame) input data
        y - (DataFrame) label data
        columns - (list) of ML model names
    '''
    if columns is None:
        columns = [str(i) for i in range(len(models))]
    #y_preds = []
    df = y.copy()
    df['Cluster_ID'] = X.Cluster_ID
    for i, model in enumerate(models):
        X_in = drop_columns(X, ['LNR', 'Cluster_ID'])
        y_pred = model.predict(X_in)
        df[columns[i]] = y_pred
    df['Average'] = df.iloc[:,-len(models):].sum(axis=1)/len(models)
    return df

def evaluate_models(models, X_train, y_train, X_test, y_test, plot_train = False, plot_test = False):
    '''
    Return the average probability averaged from the models
    Input:
        models - (list) of ML models
        X_train - (DataFrame) input training data
        y_train - (DataFrame) label training data
        X_test - (DataFrame) input testing data
        y_test - (DataFrame) label testing data
    Output:
        df_auc - (DataFrame) of AUC scores on Train and Test set
    '''
    y_train_pred_agg = models_predict(models, X_train, y_train)
    y_test_pred_agg = models_predict(models, X_test, y_test)

    data_agg_train = y_train_pred_agg[['LNR', 'RESPONSE', 'Average', 'Cluster_ID']]
    data_agg_train.columns= ['LNR', 'RESPONSE_x', 'RESPONSE_y', 'Cluster_ID']
    df_auc_train = plot_class_by_cluster(data_agg_train, plot = plot_train)\
            .rename(columns={"AUC":"AUC train", "Count: 0": "Train Count: 0", "Count: 1":"Train Count: 1"})

    data_agg_test = y_test_pred_agg[['LNR', 'RESPONSE', 'Average', 'Cluster_ID']]
    data_agg_test.columns= ['LNR', 'RESPONSE_x', 'RESPONSE_y', 'Cluster_ID']
    df_auc_test = plot_class_by_cluster(data_agg_test, plot = plot_test)\
            .rename(columns={"AUC":"AUC train", "Count: 0": "Train Count: 0", "Count: 1":"Train Count: 1"})
    df_auc = df_auc_train.merge(df_auc_test, on = 'Cluster_ID')
    roc_train_total = roc_auc_score(y_train['RESPONSE'], y_train_pred_agg['Average'])
    roc_test_total = roc_auc_score(y_test['RESPONSE'], y_test_pred_agg['Average'])
    print('Overall Train Score: {}, Test: {}'.format(roc_train_total, roc_test_total))
    return df_auc

def cluster_predict(model_General, model_Cluster, X, y, Cluster_ID, plot = False):
    '''
    Input: model_General - (object) model learned from all clusters
           model_Cluster - (object) model learned from one specific cluster
           X - (DataFrame) input x
           y - (DataFrame) output y
           Cluster_ID - (float) cluster id 
    '''
    X_subset = X[X.Cluster_ID == Cluster_ID]
    y_subset = y[X.Cluster_ID == Cluster_ID]
    df_subset_result = models_predict([model_General, model_Cluster],\
                               X_subset, y_subset, columns = ['General', 'Cluster'])
    df_subset_result.drop(columns = 'Average', inplace = True)
    df_subset_result['Cluster_Align']= (df_subset_result.Cluster - \
                                        df_subset_result.Cluster.mean())*df_subset_result.General.std()/df_subset_result.Cluster.std() + df_subset_result.General.mean()
    df_subset_result['General_Cluster'] = (df_subset_result['General'] + df_subset_result['Cluster_Align'])/2
    #auc_score = plot_class_dist(df_subset_result[['RESPONSE']], df_subset_result[['General']], title = 'General ', range = [0, 1], plot = True)
    #auc_score = plot_class_dist(df_subset_result[['RESPONSE']], df_subset_result[['Cluster']], title = 'Cluster ', range = [0, 1], plot = True)
    #auc_score = plot_class_dist(df_subset_result[['RESPONSE']], df_subset_result[['General_Cluster']], title = 'General&Cluster', range = [0, 1], plot = True)
    df_result = models_predict([model_General], X, y, columns = ['LGB'])
    df_result['Cluster_Align'] = np.nan
    df_result['Cluster_Align'] = df_subset_result['Cluster_Align']
    df_result['Cluster_Align'].fillna(value = df_result['Average'], inplace = True)
    df_result['General_Cluster'] = (df_result['Average'] + df_result['Cluster_Align'])/2
    auc_score_General = plot_class_dist(df_result[['RESPONSE']], df_result[['Average']], title = 'General ', range = [0, 1], plot = plot)
    auc_score_Cluster = plot_class_dist(df_result[['RESPONSE']], df_result[['General_Cluster']], title = 'General_Cluster', range = [0, 1], plot = plot)
    return df_result