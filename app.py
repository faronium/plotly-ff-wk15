#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors
import pandas as pd
import numpy as np
import json

def process_evdata():
    # Download CSV sheet at: https://drive.google.com/file/d/1lgEHD5n4_xqIhzCCFBq0V72OJV51e8Xz/view?usp=sharing

    df = pd.read_csv('data/Electric_Vehicle_Population_Data.csv')
    df['Model Year'] = df['Model Year'].astype('str')
    dfwa = df.loc[df.State == 'WA',:]

    #lets drop the rows with empty locations. There are less than 10.
    dfwa = dfwa.loc[~(dfwa['Vehicle Location'] == ''),:]

    #We aren't using all of this data. Drop unused rows.
    keeper_cols = ['County','Model Year','Make','Model','Electric Vehicle Type','Electric Range']
    dfwa = dfwa.loc[:,keeper_cols]

    #We have EVs with a model range set as ''. Have to correct or ditch. Then can set the Electric Range data type to int.
    dfwa[dfwa['Electric Range'].isna()] = 0
#    dfwa['Electric Range']=dfwa['Electric Range'].astype('float64')

    #We have a bunch of rows with range values set to zero. This causes issues later. 
    #We need to delete the ones we don't have information for and  replace the zero
    #with a mean range value for those we do.
    fixable_count = 0
    fixable_zeros_count = 0
    for carmodel in dfwa['Model'].unique():
        carranges = dfwa.loc[dfwa['Model'] == carmodel,'Electric Range'].unique()
        model_numzeros = len(dfwa.loc[((dfwa['Model'] == carmodel) & (dfwa['Electric Range'] == 0)),'Electric Range'])
        if 0 in carranges and len(carranges) > 1:
            #We can fix the missing ranges by applying the average of ranges that are non-zero
            fixable_count += 1
            fixable_zeros_count += model_numzeros
            mean_range = np.mean(carranges[~(carranges==0)])
            dfwa.loc[((dfwa['Model'] == carmodel) & (dfwa['Electric Range'] == 0)),'Electric Range'] = mean_range
        elif 0 in carranges and len(carranges) == 1:
            #Can't fix it, drop the rows of the dataframe for the model
            dfwa = dfwa.loc[~((dfwa['Model'] == carmodel) & (dfwa['Electric Range'] == 0)),:]
    dfwa.to_csv('data/Electric_Vehicle_Population_Data_WA_Processed.csv',index=False)
    return(dfwa)

# Specify the file path
file_path = "data/Electric_Vehicle_Population_Data_WA_Processed.csv"

# Check if the file exists
if os.path.exists(file_path):
    dfwa = pd.read_csv(file_path)
    dfwa['Model Year'] = dfwa['Model Year'].astype('str')
else:
    print('Processed file not found, processing')
    dfwa = process_evdata()
dfwa = dfwa.sort_values(by=['Model Year'])
yearlist = dfwa['Model Year'].unique()

county_file = open('data/WA_County_Boundaries_fromshape.geojson', 'r')
counties = json.load(county_file)
#counties

countylist = []
for i,feature in enumerate(counties['features']):
    countylist.append(feature['properties']['JURISDIC_2'])

#lets see if we have a 1:1 match of counties in the json in the dataframe
#print( all( county in dfwa.County.unique() for county in countylist))#

#lets see if we have a 1:1 match of counties in the dataframe in the json
#print( all( county in countylist for county in dfwa.County.unique()))

#print(len(countylist),len(dfwa.County.unique()))

def get_regs_pct_by_county(df_filtered):
    def count_rows(series):
        return (len(series))
    total_regs_by_county_year = df_filtered.groupby(['County','Model Year'])['County'].apply(count_rows)
    total_regs_by_county_year.name = 'count'

    #Unstack blindly. Assume this assumes the highest level first, which is year in this case.
    total_regs_by_county_year = total_regs_by_county_year.unstack()

    #Calculate the percentages per county for each year.
    total_regs_by_county_year_pct = np.log(100*total_regs_by_county_year/total_regs_by_county_year.sum())
    return(total_regs_by_county_year_pct)

evtypes = dfwa['Electric Vehicle Type'].unique()
total_regs_by_county_year_pct_bev = get_regs_pct_by_county(dfwa.loc[dfwa['Electric Vehicle Type'].isin([evtypes[0]]),:])
total_regs_by_county_year_pct_phev = get_regs_pct_by_county(dfwa.loc[dfwa['Electric Vehicle Type'].isin([evtypes[1]]),:])
total_regs_by_county_year_pct_both = get_regs_pct_by_county(dfwa)

#App will be a choropleth map colourized by percentage statewide of registrations per year with a slider element for the model, year. There will be a sunburst chart showing the 

#Lets make the app!
app = Dash(__name__,
                  external_stylesheets = [dbc.themes.SANDSTONE],
                  title = 'FF Week 15: Washington EVs'
              )
server = app.server
header = html.H4(
    "Exploring Washington State EV Range", className="bg-light p-2 mb-2 text-center"
)
evtypes = dfwa['Electric Vehicle Type'].unique().tolist()
slider = html.Div(
    [
        dbc.Label(html.H5("Select Model Year:")),
        dcc.Slider(
            min=2000,
            max=2025,
            step=None,
            value=2014,
            marks={int(i): f'{i}' for i in yearlist},
            included=False,
            id='ev-pct-slider',
            className="p-1",
        ),
    ],
    className="mb-4",
)

dropdown = html.Div(
    [
        dbc.Label(html.H5("Select BEV or PHEV")),
        dcc.Dropdown(
            options=[{'label': ev_type, 'value': ev_type} for ev_type in dfwa['Electric Vehicle Type'].unique()],
            value=evtypes,
            id="ev-type-dropdown",
            className="p-1",
            multi=True,
        ),
    ],
    className="mb-1",
)


controls = dbc.Card(
    [slider,dropdown],
    className="shadow",
    body=True,
)

statemapfig = dbc.Card([
    dbc.CardHeader([
        html.H4(
            id='map-title',
            className="text-center m-0", 
        ),
    ],
    style={"backgroundColor": "white"}
    ),
    dbc.CardBody([
        dcc.Graph(
            id='ev-pct-map',
            style={"height": "60vh"},
            config={'scrollZoom': False},
        )
    ])
],
className="shadow",
body=True,
)

sunplotfig = dbc.Card([
    dbc.CardHeader([
        html.H4(
            id='sunplot-title',
            className="text-center m-0"
        ),
    ]),
    dbc.CardBody([
        dcc.Graph(
            id='ev-sunplot',
            style={"height": "60vh"}
        )
    ])],
    className="shadow",
    body=True,
)

thebarplot = dbc.Card([
    dbc.CardHeader([
        html.H4(
            id='barplot-title',
            className="text-center m-0"
        ),
    ]),
    dbc.CardBody([
        dcc.Graph(
            id='ev-barplot',
            style={'height': "60vh"},
         ),
    ]),
    ],
    className="shadow",
    body=True,
)
app.layout = dbc.Container(
    style={"backgroundColor": 'darkseagreen', "minHeight": "100vh", "padding": "20px"},
    children=[
        dbc.Row([dbc.Col([header,],width=12)]),
        dbc.Row([
            dbc.Col([
                controls
            ],
            width=12,
            className='mb-4'),
        ]),
        dbc.Row([
            dbc.Col([
                statemapfig
            ],  width=6,
            className='mb-4'),
            dbc.Col([
                sunplotfig,
            ], width=6,
            className='mb-4'),
        ]),
        dbc.Row([
            dbc.Col([
                thebarplot
            ],  width=12),
        ]),
    ],
    className="dbc dbc-ag-grid",
)

@app.callback(
    Output('ev-pct-map','figure'),
    Input('ev-pct-slider','value'),
    Input('ev-type-dropdown','value'),
)
def map_ev_pct(year,evtype):
    year=str(year)
    if evtype == []:
        evtype = evtypes
    if evtype == ['Battery Electric Vehicle (BEV)']:
        total_regs_by_county_year_pct = total_regs_by_county_year_pct_bev
    elif evtype == ['Plug-in Hybrid Electric Vehicle (PHEV)']:
        total_regs_by_county_year_pct = total_regs_by_county_year_pct_phev
    else:
        total_regs_by_county_year_pct = total_regs_by_county_year_pct_both

    minval = total_regs_by_county_year_pct.min().min()
    maxval = total_regs_by_county_year_pct.max().max()
    fig = go.Figure(data=go.Choropleth(
        #total_regs_by_county_year_pct,
        geojson=counties,
        z = total_regs_by_county_year_pct[year],
        locations=total_regs_by_county_year_pct.index,
        featureidkey="properties.JURISDIC_2",
        zmin = minval,
        zmax = maxval,
        marker_line_color='white', # line markers between states
        colorscale='Blues',
        colorbar=dict(
            tickvals=np.log([0.1,1,5,10,50]),
            ticktext=["0.1%","1%","5%", "10%", "50%"],
            len=0.8,
            thickness=15,
            #orientation="h",
            #y=0.85,
            x=1.,
            yanchor='middle',
            title='% Statewide EVs',
        ),
        #colorbar_title = '% Statewide EVs',
        customdata=np.exp(total_regs_by_county_year_pct[year]),
        hovertemplate="<br><b>%{location}: %{customdata:.2f}%</b><br><extra></extra>",
    ))

    fig.update_geos(
        fitbounds="geojson",
        visible=False,
        projection=dict(
            type='conic conformal'
        ),
        resolution=50,
        showcountries=True,
    )
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
    )
    return(fig)

@app.callback(
    Output('sunplot-title', 'children'),
    Input('ev-pct-map', 'clickData'),
    Input('ev-type-dropdown','value'),
)
def make_sunplot_title(clickdata,evtype):
    if clickdata is None:
        thecounty = 'Snohomish'
    else:
        thecounty = clickdata['points'][0]['location']
    if evtype == []:
        evtype = evtypes
    if len(evtype) == 1:
        evtype_str = evtype[0]
    else:
        evtype_str = f'{evtype[0]} and {evtype[1]}'
    return html.P('EV Make, Model and Range for '+ evtype_str + ' in ' + thecounty + ' County')

@app.callback(
    Output('barplot-title', 'children'),
    Input('ev-type-dropdown','value'),
)
def make_barplot_title(evtype):
    if evtype == []:
        evtype = evtypes
    if len(evtype) == 1:
        evtype_str = evtype[0]
    else:
        evtype_str = f'{evtype[0]} and {evtype[1]}'
    return html.P('Evolution of EV range for '+ evtype_str + ' in Washington')


@app.callback(
    Output('map-title', 'children'),
    Input('ev-pct-slider', 'value'),
    Input('ev-type-dropdown','value'),
)
def make_map_title(year,evtype):
    if year is None:
        year = '2014'
    else:
        year=str(year)
    if evtype == []:
        evtype = evtypes
    if len(evtype) == 1:
        evtype_str = evtype[0]
    else:
        evtype_str = f'{evtype[0]} and {evtype[1]}'
    return html.P("Model Year "+year+" EV Registration Proportion for " + evtype_str + " by County. WA, USA")


@app.callback(
    Output('ev-sunplot', 'figure'),
    Input('ev-pct-slider','value'),
    Input('ev-pct-map', 'clickData'),
    Input('ev-type-dropdown','value'),
    #prevent_initial_call=True
)
def ev_sunburst_plot(year,clickdata,evtype):
    #Filter the data by year and county
    if year is None:
        year = '2014'
    else:
        year=str(year)
    if clickdata is None:
        thecounty = 'Snohomish'
    else:
        thecounty = clickdata['points'][0]['location']
    if evtype == []:
        evtype = evtypes

    df_filtered = dfwa.loc[dfwa['County'] == thecounty,:]
    df_filtered = df_filtered.loc[df_filtered['Model Year'] == year,:]
    df_filtered = df_filtered.loc[df_filtered['Electric Vehicle Type'].isin(evtype),:]
    fig = px.sunburst(
        df_filtered,
        path=['Make', 'Model'],
        values='Electric Range',
        color='Electric Range',
        color_continuous_scale='Oranges',
        #height=700,
    )
    fig.update_coloraxes( 
        cmin = 0,
        cmax = 300,
    )
    return(fig)

@app.callback(
    Output('ev-barplot','figure'),
    Input('ev-type-dropdown','value'),
)
def make_ev_barplot(evtype):
    if evtype == []:
        evtype = evtypes
    df_filtered = dfwa.loc[dfwa['Electric Vehicle Type'].isin(evtype),:]
    df_filtered['Model Year'] = df_filtered['Model Year'].astype('int')
    df_filtered = df_filtered.sort_values(by=['Model Year'])

    df_filtered = df_filtered.loc[df_filtered['Model Year'].isin(range(2010,2026,1)),:]
    years = df_filtered['Model Year'].unique()
    nyears = len(years)
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', nyears, colortype='rgb')

    fig = go.Figure()
    for coloridx,color in enumerate(colors):
        fig.add_trace(
            go.Violin(
                x=df_filtered.loc[df_filtered['Model Year'] == years[coloridx],'Electric Range'], 
                y=df_filtered.loc[df_filtered['Model Year'] == years[coloridx],'Model Year'],
                name=str(years[coloridx]),
                line_color=color
            )
         )
    fig.update_traces(orientation='h', side='positive', width=5, points=False)

    return(fig)

if __name__ == '__main__':
    app.run(debug=False)
