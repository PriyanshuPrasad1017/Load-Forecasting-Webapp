import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px

st.set_page_config(page_title='Load Forecasting App', layout='wide')

st.write("""
# The Load Forecasting App
""")


def build_model(df, model_name):
    df.dropna(inplace=True)    
    train_size = int(len(df) * (split_size/100))
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]
    X_train = train_set.drop(df.columns[-1], axis=1)
    Y_train = train_set[df.columns[-1]]
    X_test = test_set.drop(df.columns[-1], axis=1)
    Y_test = test_set[df.columns[-1]]

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(X_train.shape)
    st.write('Test set')
    st.info(X_test.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(X_train.columns))
    st.write('Y variable')
    st.info(Y_train.name)

    if (model_name == 'RF'):
        model = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                      random_state=parameter_random_state,
                                      max_features=parameter_max_features,
                                      criterion=parameter_criterion,
                                      min_samples_split=parameter_min_samples_split,
                                      min_samples_leaf=parameter_min_samples_leaf,
                                      bootstrap=parameter_bootstrap,
                                      oob_score=parameter_oob_score,
                                      n_jobs=parameter_n_jobs)
    elif (model_name == 'KNN'):
        model = KNeighborsRegressor(n_neighbors=parameter_n_neighbors,
                                    weights=parameter_weights,
                                    algorithm=parameter_algorithm,
                                    n_jobs=parameter_n_jobs)
    model.fit(X_train, Y_train)

    st.subheader('2. Model Performance')

    st.markdown('**2.1. Training set**')
    Y_pred_train = model.predict(X_train)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Test set**')
    Y_pred_test = model.predict(X_test)
    st.write('Coefficient of determination ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    y_pred=model.predict(X_test)
    dfp = pd.DataFrame(data={'Predictions':y_pred, 'Actuals':Y_test})
    dfp['relative_error%'] = (abs(dfp['Predictions'] - dfp['Actuals']) / dfp['Actuals'])*100
    st.write('Error (MSE or MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))

    fig = px.line(dfp.iloc[:100], x=dfp.iloc[:100].index, y=['Predictions', 'Actuals'], 
              labels={'value': 'Load', 'index': 'index'},
              title='Predictions vs Actuals')
    fig.update_layout(
    xaxis=dict(tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=10)),
    font=dict(size=12)
    )
    st.plotly_chart(fig)
    st.write(dfp)
    st.write('Mean Relative Error %')
    st.info(dfp['relative_error%'].mean())

    st.subheader('3. Model Parameters')
    st.write(model.get_params())

# SIDEBAR
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])

with st.sidebar.header('2. Select Algorithm'):
    model_name = st.sidebar.selectbox('Model', ('RF', 'KNN'))

with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider(
        'Data split ratio (% for Training Set)', 10, 90, 80, 5)
    

# MODEL PARAMETERS IN SIDEBAR
if (model_name == 'RF'):
    with st.sidebar.subheader('3.1. Learning Parameters'):
        parameter_n_estimators = st.sidebar.slider(
            'Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.sidebar.select_slider(
            'Max features (max_features)', options=['auto', 'sqrt', 'log2'])
        parameter_min_samples_split = st.sidebar.slider(
            'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
        parameter_min_samples_leaf = st.sidebar.slider(
            'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

    with st.sidebar.subheader('3.2. General Parameters'):
        parameter_random_state = st.sidebar.slider(
            'Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.sidebar.select_slider(
            'Performance measure (criterion)', options=['mse', 'mae'])
        parameter_bootstrap = st.sidebar.select_slider(
            'Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.sidebar.select_slider(
            'Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
        parameter_n_jobs = st.sidebar.select_slider(
            'Number of jobs to run in parallel (n_jobs)', options=[1, -1])
        
elif (model_name == 'KNN'):
    with st.sidebar.subheader('3.1. Parameters'):
        parameter_n_neighbors = st.sidebar.slider(
            'Number of estimators (n_neighbors)', 2, 50, 5, 1)
        parameter_weights = st.sidebar.select_slider(
            'Weights', options=['uniform', 'distance'])
        parameter_algorithm = st.sidebar.select_slider(
            'Algorithm', options=['auto', 'ball_tree', 'kd_tree', 'brute'])
        parameter_n_jobs = st.sidebar.select_slider(
            'Number of jobs to run in parallel (n_jobs)', options=[1, -1])

st.markdown('Do you want to generate Pandas Profiling Report ?')
option = st.selectbox('Select Option', ('Select', 'Yes', 'No'), index=0)

# DRIVER
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if (option != 'Select' and option == 'Yes'):
        st.subheader('1. Dataset')
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        pr = ProfileReport(df, explorative=True)
        st.header('**EDA**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
        build_model(df, model_name)
    elif (option != 'Select'):
        st.subheader('1. Dataset')
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df, model_name)
else:
    st.info('Awaiting for CSV file to be uploaded.')
