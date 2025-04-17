#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

def process_evdata():
    # Download CSV sheet at: https://drive.google.com/file/d/1lgEHD5n4_xqIhzCCFBq0V72OJV51e8Xz/view?usp=sharing

    df = pd.read_csv('data/Electric_Vehicle_Population_Data.csv')
    df['Model Year'] = df['Model Year'].astype('str')
    dfwa = df.loc[df.State == 'WA',:]


    #for index, row in dfwa.iterrows():
    #    try:
    #        value = row['Vehicle Location']
    #        wkt.loads(value)
    #    except Exception:
    #       print(f'{index} [{len(value)}] {value!r} {row}')

    #lets drop the rows with empty locations. There are less than 10.
    dfwa = dfwa.loc[~(dfwa['Vehicle Location'] == ''),:]

    #Make the location geoseries from the well known text (WKT) in Vehicle Location
    #gs = gpd.GeoSeries.from_wkt(dfwa['Vehicle Location'])

    #Insert into the dataframe and make the dataframe into a geopandas
    #dfwa = gpd.GeoDataFrame(dfwa, geometry=gs, crs="EPSG:4326")

    #Ditch the text WKT
    dfwa = dfwa.drop(columns=['Vehicle Location'])


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
            print(
                'fixing ',carmodel,' with ',model_numzeros,'zeros',fixable_zeros_count
            )
            dfwa.loc[((dfwa['Model'] == carmodel) & (dfwa['Electric Range'] == 0)),'Electric Range'] = mean_range
        elif 0 in carranges and len(carranges) == 1:
            #Can't fix it, drop the rows of the dataframe for the model
            print(
                "Can't fix ",carmodel,'with',model_numzeros,'rows without range information. Dropping the rows.'
            )
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
yearlist


# In[2]:


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


# In[3]:


# #Now have to find when that peak occurred...!
# def count_rows(series):
#     return (len(series))
# #lets get a count of the number of registrations per county then calculate the percentage of statewide totals in each county. Stratifies by year possibly
# total_regs_by_county = dfwa.groupby('County')['County'].apply(count_rows)
# total_regs_by_county_year = dfwa.groupby(['County','Model Year'])['County'].apply(count_rows)
# total_regs_by_county_year.name = 'count'

# #Unstack blindly. Assume this assumes the highest level first, which is year in this case.
# total_regs_by_county_year = total_regs_by_county_year.unstack()

# #Calculate the percentages per county for each year. 
# total_regs_by_county_year_pct = np.log(100*total_regs_by_county_year/total_regs_by_county_year.sum())


# In[6]:


#App will be a choropleth map colourized by percentage statewide of registrations per year with a slider element for the model, year. There will be a sunburst chart showing the 

#Lets make the app!
plotly-ff-wk15 = Dash(__name__,
                  external_stylesheets = [dbc.themes.SANDSTONE],
                  title = 'FF Week 15: Washington EVs'
              )
server = aplotly-ff-wk15.server
header = html.H4(
    "Exploring Washington State EV Range", className="bg-primary p-2 mb-2 text-center"
)
evtypes = dfwa['Electric Vehicle Type'].unique().tolist()
slider = html.Div(
    [
        dbc.Label("Select Model Year:"),
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
        dbc.Label("Select BEV or PHEV"),
        dcc.Dropdown(
            options=[{'label': ev_type, 'value': ev_type} for ev_type in dfwa['Electric Vehicle Type'].unique()],
            value=evtypes,
            id="ev-type-dropdown",
            className="p-6",
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
            style={"height": "80vh"},
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
            style={"height": "80vh"}
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
plotly-ff-wk15.layout = dbc.Container(
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
    fluid=True,
    className="dbc dbc-ag-grid",
)

@plotly-ff-wk15.callback(
    Output('ev-pct-map','figure'),
    Input('ev-pct-slider','value'),
    Input('ev-type-dropdown','value'),
)
def map_ev_pct(year,evtype):
    year=str(year)
    if evtype == []:
        evtype = evtypes
    def count_rows(series):
        return (len(series))
    df_filtered = dfwa.loc[dfwa['Electric Vehicle Type'].isin(evtype),:]
    total_regs_by_county_year = df_filtered.groupby(['County','Model Year'])['County'].apply(count_rows)
    total_regs_by_county_year.name = 'count'

    #Unstack blindly. Assume this assumes the highest level first, which is year in this case.
    total_regs_by_county_year = total_regs_by_county_year.unstack()

    #Calculate the percentages per county for each year.
    total_regs_by_county_year_pct = np.log(100*total_regs_by_county_year/total_regs_by_county_year.sum())
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
        colorscale='Reds',
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

@plotly-ff-wk15.callback(
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

@plotly-ff-wk15.callback(
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


@plotly-ff-wk15.callback(
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


@plotly-ff-wk15.callback(
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
        color_continuous_scale='Greens',
        #height=700,
    )
    fig.update_coloraxes( 
        cmin = 0,
        cmax = 300,
    )
    return(fig)

@plotly-ff-wk15.callback(
    Output('ev-barplot','figure'),
    Input('ev-type-dropdown','value'),
)
def make_ev_barplot(evtype):
    if evtype == []:
        evtype = evtypes
    df_filtered = dfwa.loc[dfwa['Electric Vehicle Type'].isin(evtype),:]
    return(px.box(df_filtered.sort_values(by=['Model Year']), y='Electric Range',x='Model Year'))

if __name__ == '__main__':
    plotly-ff-wk15.run(debug=False)


# In[ ]:




