import xgboost as xgb
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Commodity Price Forecast', layout='wide', page_icon='🛒')

st.markdown("<h1 style='text-align: center; color: #A8A8A8;'>🛒 Commodity Wholesale Price Forecasting 💸</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: white;'>Enter the features below👇</h3>", unsafe_allow_html=True)

@st.cache_data
def load_data(path):
    new_df = pd.read_csv(path)
    new_df.drop(columns = ['Unnamed: 0'], inplace = True)
    new_df = new_df
    return new_df

@st.cache_resource
def loading_model():
    model = xgb.XGBRegressor()
    model.load_model('model_sklearn.json')
    return model

@st.cache_data
def chart_prep(new_df, state, district, market, commodity):
    tempdf = new_df[(new_df['state'] == state) & (new_df['district'] == district) & (new_df['market'] == market) & (new_df['commodity'] == commodity)]
    tempdf['date'] = pd.to_datetime(tempdf['date'])
    avg_arr_tonne = tempdf.sort_values(by = ['date']).tail(10)['arrival'].mean()
    fig = px.area(tempdf, x="date", y="mod_rs", title='Commodity Wholesale Price')
    return fig, avg_arr_tonne

@st.cache_data
def comm_select(new_df, state, district, market):
    tempdf = new_df[(new_df['state'] == state) & (new_df['district'] == district) & (new_df['market'] == market)]['commodity'].unique()
    return tuple(tempdf)

def predict(_model, _user_input):
    predicted = _model.predict(_user_input)
    return predicted[0]

def get_season(date):
    if date.month in [12, 1, 2]:
        return 'winter'
    elif date.month in [3, 4, 5]:
        return 'spring'
    elif date.month in [6, 7, 8]:
        return 'summer'
    else:
        return 'monsoon'

url="https://drive.google.com/file/d/1xubrLuhQDEr_vz1xFBxK1rKQHh1DpT7v/view?usp=sharing"
path_csv = 'https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])
new_df = load_data(path_csv)

with open('label_mapping.pkl', 'rb') as f:
    label_map = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    var_map, dist_map, mark_map, comm_map = pickle.load(f)

with st.spinner("Loading Model"):
    model = loading_model()

categorical_columns = ['state', 'district', 'market', 'variety', 'group', 'commodity', 'season']
state,district,market,variety,group,arr,min_rs,max_rs,commodity,season,month,day,year = (None, None,None,None,None,None,None,None,None,None,None,None,None)

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with st.form('Details for the forecast!!'):
    with col1:
        selected_state = st.selectbox('Select State Name', tuple(label_map['state'].keys()), index=20, placeholder="Select State....")
        if selected_state:
            state = label_map['state'][selected_state]

    with col2:
        if selected_state:
            selected_dist = st.selectbox('Select District Name', dist_map[selected_state], index=None, placeholder="Select District....")
            if selected_dist:
                district = label_map['district'][selected_dist]

    with col3:
        if selected_dist:
            selected_market = st.selectbox('Select Market', mark_map[selected_dist], index=None, placeholder="Select Market....")
            if selected_market:
                market = label_map['market'][selected_market]

            comm2 = comm_select(new_df, selected_state, selected_dist, selected_market)

    with col4:
        if selected_dist:
            selected_commodity = st.selectbox('Select Commodity', comm2, index=None, placeholder="Select commodity....")
            if selected_commodity:
                commodity = label_map['commodity'][selected_commodity]
                group = label_map['group'][comm_map[selected_commodity]]

    with col5:
        if selected_dist:
            if selected_commodity:
                selected_variety = st.selectbox('Select Variety', var_map[selected_commodity], index=None, placeholder="Select Variety....")
                if selected_variety:
                    variety = label_map['variety'][selected_variety]

    with col6:
        selected_date = st.date_input('Select Date')
        if selected_date:
            day = selected_date.day
            month = selected_date.month
            year = selected_date.year
            ok_season = get_season(date=selected_date)
            season = label_map['season'][ok_season]

    st.markdown("<h4 style='text-align: center; color: white;'>Slide Or Enter into Input Box</h4>", unsafe_allow_html=True)

    col7, col8 = st.columns(2)
    with col7:
        arr = st.slider('Slide to Arrived Weights (in Tonnes)', min_value=0.01, max_value=25.0)
        min_rs = st.slider('Slide to minimum price', min_value=0, max_value=3000)
        max_rs = st.slider('Slide to maximum price', min_value=0, max_value=3500)

    with col8:
        arr = st.number_input('Enter Arrived Weights (in Tonnes)', value=arr)
        min_rs = st.number_input('Enter minimum price', value=min_rs)
        min_rs = st.number_input('Enter maximum price', value=max_rs)

    submitted = st.form_submit_button("Forecast for the selected date..")

if submitted:
    user_input = [[state, district, market, variety, group, arr, min_rs, max_rs, commodity, season, month, day, year]]
    predicted = predict(_model=model, _user_input=user_input)
    st.markdown(f'<p style="text-align: center;font-size:30px">Rs : <span style="font-size:60px;background-color: #428A00">{predicted:.3f}</span> per Quintal (Rs/100Kgs)</p>', unsafe_allow_html=True)
    fig, avg_arr = chart_prep(new_df, selected_state, selected_dist, selected_market, selected_commodity)
    if arr > avg_arr*1.05:
        st.success(f"More stock has been arrived than average of {avg_arr} tonnes last 10 Days!!", icon='⚠️')
    if arr < avg_arr*0.95:
        st.warning(f"Less stock has been arrived than average of {avg_arr} tonnes last 10 Days!!", icon='⚠️')
    st.plotly_chart(fig)
