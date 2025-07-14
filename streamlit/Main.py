import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title = 'Portugal Hotel Booking Demand',
    page_icon = ':hotel:'
)

st.title(':hotel: Portugal Hotel Booking Demand')

st.markdown('''
Welcome to the Portugal Hotel Booking Analysis App.

This app includes:

- A **simple interactive dashboard** to explore key booking insights.
            
- A link to a **full Tableau dashboard** for more detailed visualizations.
           
- A **Machine Learning page** where you can try out a predictive model for booking cancellations.

Use the sidebar to navigate between pages and interact with the data.
\n            
\n
\n
The data used in this app is sourced from the [Portugal Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data)
            
View the GitHub repository for the code and more details: [GitHub Repository](https://github.com/PurwadhikaDev/AlphaGroup_JC_DS_FT_JKT_28_FinalProject)

View the Tableau dashboard for more visualizations: [Tableau Dashboard](https://public.tableau.com/app/profile/vanessa.alexandra1705/viz/Alpha_FinalProject_PortugalHotel/StayorGo?publish=yes)
''')

st.sidebar.success('''
                   Purwadhika JC Data Science & Machine Learning
                   
                   Alpha Group JCDS 2804
                   ''')

df = pd.read_csv('C:/Users/Dell/Documents/Purwadhika/Final Project/df_cleaned.csv')

# Summary statistics
st.subheader('Summary Statistics')
total_bookings = len(df)
cancellation_rate = str(round((df['is_canceled'].mean() * 100), 2)) + '%'
most_frequent_hotel = df['hotel'].value_counts().idxmax()

col1, col2, col3 = st.columns(3)
col1.metric('Total Bookings', f'{total_bookings:,}')
col2.metric('Cancellation Rate', f'{cancellation_rate}')
col3.metric('Top Hotel', f'{most_frequent_hotel}')

# Country and continent distribution
continent_data = df['continent'].value_counts().reset_index()
continent_data.columns = ['Continent', '# of Bookings']

country_data = df['country'].value_counts().reset_index()
country_data.columns = ['Country', '# of Bookings']

st.subheader('Guest National')

fig_choropleth = px.choropleth(
    country_data,
    locations = 'Country',
    locationmode = 'ISO-3',
    color = '# of Bookings',
    color_continuous_scale = 'Blues',
    range_color = [0, country_data['# of Bookings'].max()]
    )

fig_choropleth.update_layout(
    coloraxis_colorbar = dict(
        title = 'Bookings',
        # x = 0.9, 
        y = -0.2,
        # len = 0.75,
        # thickness = 15,
        orientation = 'h'
    )
    )

fig_choropleth.update_geos(fitbounds = 'locations', visible = True)

st.plotly_chart(fig_choropleth, use_container_width = True)

# Top Countries
top_countries = df['country'].value_counts().nlargest(15)

fig_top_countries = px.bar(
    top_countries.sort_values(ascending = False),
    orientation = 'h',
    labels = {'value': 'Number of Bookings', 'country': 'Country'},
    title = 'Top 15 Countries by Booking Count'
)

fig_top_countries.update_layout(yaxis = dict(categoryorder = 'total ascending'))

st.plotly_chart(fig_top_countries, use_container_width = False)


# Booking distribution by continent and monthly cancellations
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader('Booking Distribution by Continent')
    fig_pie = px.pie(continent_data, 
                     names = 'Continent', 
                     values = '# of Bookings',
                     color_discrete_sequence = px.colors.sequential.RdBu)
    
    fig_pie.update_traces(
        hole = 0.4,
        textposition = 'inside', 
        textinfo = 'percent+label'
        )
    
    fig_pie.update_layout(
        legend=dict(orientation = 'h',
                    yanchor = 'bottom', y = 1.02,
                    xanchor = 'center', x = 0.5),
        uniformtext_minsize = 11,
        uniformtext_mode = 'hide'
        )
    
    st.plotly_chart(fig_pie, use_container_width = True)

with col2:
    # Hotel Booking distribution
    st.subheader('Monthly Cancellations')

    hotel_distribution = df['hotel'].value_counts().reset_index()
    hotel_distribution.columns = ['Hotel', ' Number of Bookings']

    monthly_cancellations =(
        df[df['is_canceled'] == 1].
        groupby('arrival_date_month')
        .size()
        .reindex(['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'])
        .reset_index(name = 'Cancellations')
    )

    fig1 = px.bar(
        monthly_cancellations,
        x = 'arrival_date_month', y = 'Cancellations'
        )
    
    st.plotly_chart(fig1, use_container_width = True)