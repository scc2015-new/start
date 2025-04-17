import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from plotly import graph_objs as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import plotly.express as px # type: ignore
import seaborn as sns # type: ignore
import io
import pickle
from PIL import Image # type: ignore
import json

wh1=pd.read_csv("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/world-happiness-report-2021.csv")
wh2=pd.read_csv("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/world-happiness-report.csv")
wh_merged=pd.read_csv("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/world_happiness.csv", index_col= 0)
wh_ml=pd.read_csv("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/world_happiness_forML.csv", index_col= 0)

title = '<h1 style="color: #00549f;">The World Happiness Report</h1>'
st.markdown(title, unsafe_allow_html=True)

## SIDEBAR ##

#image (free to use from unsplash)
#https://unsplash.com/de/fotos/eine-person-die-auf-eine-karte-mit-stecknadeln-zeigt-SFRw5GChoLA
st.sidebar.image("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/smiley1.jpg", width=200)

st.sidebar.title("Table of contents")
pages=["Introduction","Data Exploration and Preprocessing", "Data Visualization", "Data Modelling", "Prediction", "Conclusion"]
page=st.sidebar.radio("Go to", pages)

#Path to datascientest logo file
logo_url = 'https://assets-datascientest.s3-eu-west-1.amazonaws.com/notebooks/looker_studio/logo_datascientest.png'

st.sidebar.markdown(
  f"""
  <style>
    .sidebar-content {{
      text-align: left;
      font-size: 10px;}}

    .sidebar-content img {{
        width: 50px; 
        margin: 10px 0;}}

    </style>
    <div class="sidebar-content">
      <br><br>
      <p><i>A Project by Sarbani Chatterjee, Stella Edith Hoffmann and Thiago Baldo</i></p>
      <p><i>Project Mentor: Tarik Anouar</i></p>
      <img src="{logo_url}" alt="Logo">
      <p><i>January 2025</i></p>      
    </div>
    """,
    unsafe_allow_html=True)




#### INTRODUCTION ####
if page == pages[0] : 
  #header_html = "<h2 style='color: black;'>Understanding the aspects behind</h2>"
  #st.markdown(header_html, unsafe_allow_html=True)
  
  #copy paste from our report
  #st.write("In today's society, understanding the factors that contribute to human happiness has become more crucial than ever, for people, and businesses. This World Happiness Report goes deeper into the topic of overall perception of happiness and well-being, offering insights about what makes people feel satisfied with their lives and the environment they are inserted in. Through data collection and detailed analysis, this report evaluates the happiness levels of various countries, covering a variety of elements, such as social, economic, corruption and health-related that shape our perceptions of well-being. These evaluations help to clarify our understanding of the concepts of happiness around the world, and how businesses can take advantage of these insights to boost their strategies.")
  #st.write("In this project, we aim to analyse the factors that contribute to the overall perception of happiness around the world and how these factors are interconnected. By diving into the data, we look for insights into what contributes to happiness in different contexts and how these aspects can be interconnected. We will explore whether there is a universal shared perception of happiness or if it varies significantly across different regions and cultures. Companies can use this knowledge to create strategies that enhance their sales and market, improve employee satisfaction, and foster a positive work environment.")
  st.write("## Introduction") 
  st.image("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/Happy6.jpg", width= 550)
 
  header_html = "<h3 style='color: black;'>“Most people are about as happy as they make up their minds to be.”    —  Abraham Lincoln</h3>"
  st.markdown(header_html, unsafe_allow_html=True)

  st.write("***What makes people feel satisfied with their lives and the environment they are inserted in?***")
  st.write("The World Happiness Survey strives to quantify this ambiguous concept of Happiness.")
  st.write("""Here 'Happiness' has been bounded using personal freedom, social, economic, corruption and health-related elements 
           that shape our perceptions of well-being. Since our perceptions vary widely, so do the world survey results and it is 
           imperative that we are sensitive to these differences and don’t discard them as outliers. Let’s deep dive into the data, 
           analyze and deduce the components which contribute to the overall perception of happiness around the world and how these 
           factors are interconnected.""")




#### DATA EXPLORATION AND PREPROCESSING ####
if page == pages[1] : 
  header_html = "<h2 style='color: black;'>Data Exploration and Preprocessing</h2>"
  st.markdown(header_html, unsafe_allow_html=True)

  st.write("Two data sets were used for the following analysis of this report:")
  st.write("-", "world-happiness-report-2021: wh1")
  st.write("-", "world-happiness-report: wh2")

  #data sources
  st.write("Data Source:")
  st.write("https://www.kaggle.com/ajaypalsinghlo/world-happiness-report-2021")

  #st.write("Each dataset contains essential information for understanding the Ladder score or Happiness Score of numerous countries around the world. Moreover, there is also the website of The World Happiness Report, which provides further background information on the topic.")
  
  #first dataset
  st.write("**First Dataset: wh1**")
  st.dataframe(wh1.head(10))
  if st.checkbox("Show missing values for first dataset"):
    st.dataframe(wh1.isna().sum())
    
  #for variable in bold
  # st.write(f"Column name is **{columnName}**")
  #maybe in bold
  st.write("**Details & Highlights : wh1**")
  st.write("-", "Number of countries: 149")
  
  #dropdown or selection or table
  st.write("-", "Time span: 2021")
  st.write("-", "Number of Rows & Columns :",wh1.shape)
  st.write("-", "Dystopia: Lowest Laderscore")

  #second dataset
  st.write("**Second Dataset: wh2**")
  st.dataframe(wh2.head(10))
  if st.checkbox("Show missing values for second dataset"):
    st.dataframe(wh2.isna().sum())

  st.write("**Details & Highlights : wh2**")
  st.write("-", "Number of countries: 166")
  #dropdown oder selection or table
  st.write("-", "Time span: 2005-2020")
  st.write("-", "Number of Rows & Columns :",wh2.shape)

  # Preprocessing :
  #change column names
  wh1 = wh1.rename(columns = {'Logged GDP per capita': 'Log GDP per capita' })
  wh2 = wh2.rename(columns = {'Life Ladder': 'Ladder score','Healthy life expectancy at birth': 'Healthy life expectancy' })
  
  #new column for the 2021 dataset
  wh1['year'] = 2021
  
  #new column names for the 2 datasets, removing spaces in between
  wh1.columns = wh1.columns.str.strip().str.replace(' ', '_')
  wh2.columns = wh2.columns.str.strip().str.replace(' ', '_')
  
  wh = pd.concat([wh1, wh2], axis=0)
  wh=wh.sort_values(by='Country_name')
  def missing_percentage(wh):
    missing = round((wh.isnull().sum()/len(wh) * 100),2)
    return missing
  
  missing_percentage_wh = missing_percentage(wh)

  st.write("")
  st.write("")
  
  st.write("**Preprocessing**")
  st.write("**1.**", "Duplicates checked : None found in wh1 & wh2")
  st.write("")
  st.write("**2.**", "Columns of wh1 & wh2 renamed so they share the same syntax")
  st.write("")
  st.write("**3.**", "Year column added to wh1, to enable merging")
  st.write("")
  st.write("**4.**", "Concatenating wh1 & wh2 into new dataset : wh ")
  st.write("")

  buffer1 = io.StringIO()
  wh.info(buf=buffer1)
  s = buffer1.getvalue()
  st.text(s)
  
  #TOP AND FLOP 5
  wh1_sorted = wh1.sort_values('Ladder_score', ascending=False)
  top_5 = wh1_sorted.head(5)
  bottom_5 = wh1_sorted.tail(5)
  top_bottom = pd.concat([top_5, bottom_5], axis=0)

  fig_country = px.bar(top_bottom, x='Ladder_score', y='Country_name', 
                         color='Ladder_score', color_continuous_scale='plasma',
                         title='Top 5 and Bottom 5 Ladder Scores (2021)'
                         )
   
  fig_country.update_layout(showlegend=False, 
                            coloraxis_showscale=False, 
                            height=450, 
                            width=600,
                            title={
                               'x': 0.5,
                               'xanchor': 'center'},
                            title_font_size=20,
                            xaxis_title='Ladder Score', 
                            yaxis_title='Country',
                            yaxis=dict(autorange="reversed"))

  st.plotly_chart(fig_country)

  st.write("-", "*The top 5 countries all have more than 7 points and are located in Europe*")
  st.write("-", "*The flop 5 countries are mostly in Africa and have less than 4 points*")
  st.write("")
  st.write("")  

  wh=wh.sort_values(by='Country_name')
  
  #fill missing Regional indicators mapping it from the countries with RI present.
  wh.loc[wh['Regional_indicator'].isnull(), 'Regional_indicator'] = \
              wh.loc[wh['Regional_indicator'].isnull(), 'Country_name'].map(wh.loc[wh['Regional_indicator'].notnull()] \
                .set_index('Country_name')['Regional_indicator'])
  
  missing_percentage_wh = missing_percentage(wh)

  
 
  st.write("**5.**", "Regional Indicators(RI) were mapped from wh1 into the merged dataset wh. ")
  if st.checkbox("Show missing values for wh after mapping RIs"):
    st.dataframe(wh.isnull().sum())
    
  st.write("**6.**", "63 missing RI values, corresponding to 17 unique countries persisted and were filled by hand.")
  
  
  # replacing missing regional indicators by hand
  region_mapping = {
    'Cuba': 'Latin America and Caribbean',
    'Trinidad and Tobago': 'Latin America and Caribbean',
    'Central African Republic': 'Sub-Saharan Africa',
    'Guyana': 'Latin America and Caribbean',
    'Belize': 'Latin America and Caribbean',
    'Syria': 'Middle East and North Africa',
    'Djibouti': 'Sub-Saharan Africa',
    'Somaliland region': 'Sub-Saharan Africa',
    'Sudan': 'Sub-Saharan Africa',
    'Qatar': 'Middle East and North Africa',
    'Congo (Kinshasa)': 'Sub-Saharan Africa',
    'Oman': 'Middle East and North Africa',
    'Angola': 'Sub-Saharan Africa',
    'Suriname': 'Latin America and Caribbean',
    'Bhutan': 'South Asia',
    'South Sudan': 'Sub-Saharan Africa',
    'Somalia': 'Sub-Saharan Africa'}
  
  wh['Regional_indicator'] = wh['Country_name'].map(region_mapping).fillna(wh['Regional_indicator'])
  if st.checkbox("Show missing percentages for merged dataset(wh) after filling Regional Indicators"):
    st.dataframe(missing_percentage(wh))
    
  st.write("**Preprocessing for Modelling Purpose**")  
  st.write("**7.**", "Dropped 14 Columns which explained the influence on ladder score and dystopia")
  
  # Keeping only the columns with info in both, dropping rest. keeping a merged file copy with the undeleted columns just as a precaution.
  wh_Original = wh.copy()
  wh=wh.drop(['Standard_error_of_ladder_score','upperwhisker','lowerwhisker','Explained_by:_Log_GDP_per_capita',\
                        'Explained_by:_Social_support','Explained_by:_Freedom_to_make_life_choices','Explained_by:_Generosity','Explained_by:_Healthy_life_expectancy','Explained_by:_Perceptions_of_corruption',\
                        'Dystopia_+_residual','Positive_affect','Negative_affect','Dystopia_+_residual','Ladder_score_in_Dystopia'], axis=1, inplace=False)
 
  #OUTLIERS
  st.write("**Outliers**")

  st.write("To better assess the individual variables, let's take a look at the distribution of the factors that influence the ladder score.")

  fig = make_subplots(rows=2, cols=3)
  
  fig.add_trace(go.Box(
    y=wh_merged["Social_support"],
    name='Social support',
    marker_color='#00549f'), row=1, col=1)
  
  fig.add_trace(go.Box(
    y=wh_merged["Log_GDP_per_capita"],
    name='Log GDP per capita',
    marker_color='#00549f'), row=1, col=2)
  
  fig.add_trace(go.Box(
    y=wh_merged["Healthy_life_expectancy"],
    name='Healthy life expectancy',
    marker_color='#00549f'), row=1, col=3)

  fig.add_trace(go.Box(
    y=wh_merged["Freedom_to_make_life_choices"],
    name='Freedom to make life choices',
    marker_color='#00549f'), row=2, col=1)
  
  fig.add_trace(go.Box(
    y=wh_merged["Perceptions_of_corruption"],
    name='Perceptions of corruption',
    marker_color='#00549f'), row=2, col=2)
  
  fig.add_trace(go.Box(
    y=wh_merged["Generosity"],
    name='Generosity',
    marker_color='#00549f'), row=2, col=3)

  fig.update_layout(
    barmode='group',
    width=800,
    height=600,
    showlegend=False)


  st.plotly_chart(fig)

  st.write("**Social support**")  
  st.write("-", "The median for social support is relatively high and there are many downward outliers. So, the distribution is slightly negatively skewed and has positive kurtosis and low spread.")
  st.write("**Log GDP per capita**")  
  st.write("-", "Since GDP per capita tends to be negatively skewed, the log of this variable has already been performed. So, we see an almost normal distribution with even dispersion.") 
  st.write("**Healthy life expectancy**") 
  st.write("-", "The median age is around 66 years, which is relatively high. There are clearly several outlying countries that have a very low healthy life expectancy (under 44 years). ")
  st.write("**Freedom to make life choices**")    
  st.write("-", "This boxplot with a long lower whisker indicates a negatively skewed data with a high median. However, the data spread is quite large.")
  st.write("**Perception of corruption**")  
  st.write("-", "In terms of perceptions of corruption, the data is negatively skewed.The distribution is peaked with many low outliers and the data spread is low.")
  st.write("**Generosity**")  
  st.write("-", "This shows a positively skewed data with low median, small dispersion and negative kurtosis. In contrast, here there are outliers in the upper part.")
  st.write("")
  st.write("")  
  st.write("")  
  

  st.write("**8.**", "Numerical Variables were separated from the categorical variables.<br>" 
           "-     ","After taking cognizance of the variance in factors, individually and region wise and the outliers, the missing values were filled using mean of parameters grouped by the regions they belonged to.",
           unsafe_allow_html=True)
  st.write("")

  st.write("**9.**","Categorical Variables: Country_name and Regional_indicator were encoded into numbers :<br>"
            "-     ","Regional_indicator was encoded using pandas get_dummies method.<br>"
            "-     ","Country_name was encoded using Label Encoder from sklearn preprocessing library",
            unsafe_allow_html=True)
  st.write("")

  st.write("**10.**", "Numerical and categorical variables were concatenated and exported for Machine Learning purpose")

  #first adaptions for charts, otherwise the graphs in the visualisation DO NOT WORK
  wh1 = wh1.rename(columns = {'Logged GDP per capita': 'Log GDP per capita' })
  wh2 = wh2.rename(columns = {'Life Ladder': 'Ladder score','Healthy life expectancy at birth': 'Healthy life expectancy' })
  wh1['year'] = 2021
  wh1.columns = wh1.columns.str.strip().str.replace(' ', '_')
  wh2.columns = wh2.columns.str.strip().str.replace(' ', '_')

  

  





#### VISUALIZATION ####
if page == pages[2] : 
  header_html = "<h2 style='color: black;'>Data Visualization</h2>"
  st.markdown(header_html, unsafe_allow_html=True)
  
  st.write("""Several visualisations were developed to illustrate the insights gained during the pre-processing phase. 
  The titles and descriptions of the diagrams created during the analysis are presented in the following sections:""")


  #CORRELATION MATRIX
  st.write("### Correlation Matrix")
  st.write("The following heatmap shows the correlations between some variables from the merged data sets.")
 
  cor_merge = wh_merged[['Ladder_score', 'Log_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy', 'Freedom_to_make_life_choices', 'Generosity', 'Perceptions_of_corruption']]
  cor_matrix = cor_merge.corr()
  corr_rounded = cor_matrix.round(2)
  
  fig = px.imshow(
    corr_rounded,
    color_continuous_scale='plasma',
    width =800, height = 600,
    #title="Correlation between the different variables and the ladder score",
    text_auto=True)

  
  st.plotly_chart(fig)

  st.write("**Positive correlations:**")
  st.write("-", """Ladder Score and log GDP per capita **(0.79)**: Economic prosperity supports well-being""")  
  st.write("-", "Ladder score and healthy life expectancy **(0.75)**: Health is key to life satisfaction.")    
  st.write("-", "Ladder score and social support **(0.71)**: Strong social networks boost happiness.")     
  st.write("**Negative correlations:**")
  st.write("-", "Perception of corruption and ladder score **(-0.43)**: Institutional trust enhances well-being.")  
  st.write("-", "Perception of corruption and freedom to make life choices **(-0.48)**: Corruption limits perceived freedom.")     
  st.write("")
  st.write("")


  #LADDER SCORE DISTRIBUTION
  st.write("### Ladder Score Distribution")
  st.write("We have chosen several boxplots to visualise the distribution of ladder scores across regions.")
  
  #add_color = st.checkbox("Coloring the regional indicator")
  
  wh_sorted_ladder = wh_merged.sort_values(by="Ladder_score", ascending = False) 

  fig = px.box(wh_sorted_ladder,
             x="Regional_indicator",
             y="Ladder_score",
             color="Regional_indicator",
             )


  fig.update_traces(marker=dict(size=10, line=dict(width=2, color='#000000')), selector=dict(type='box'))
  
  fig.update_layout(
    height=600,  
    width=1000,
    #title_font_size=20,
    xaxis_title='Region',
    yaxis_title='Ladder Score',
    xaxis_tickangle=-45,
    legend_title_text='Region',
    legend=dict(
        orientation="h",        
        yanchor="bottom",
        y=-1.1,                 
        xanchor="center",
        x=0.5))

  st.plotly_chart(fig, use_container_width=True)

  st.write("")
  st.write("-", "Western Europe and North America & ANZ have the highest values and the lowest dispersion - showing a high and stable level of well-being.")
  st.write("-", "Sub-Saharan Africa and South Asia show the lowest values with a wide spread in some cases - well-being varies greatly here.")
  st.write("-", "Regions such as Latin America, Middle East and North Africa and Central and Eastern Europe are in the midfield, but show some major differences between countries.")  
  st.write("")
  st.write("")

  #REGION WISE LADDER SCORE OVER THE YEARS 
  st.write("### Region-wise average Ladder Score over the years")

  st.write("This graph shows the average ladder score, a representation of regionwise ‘Happiness’, over a period of 16 years , from 2005 to 2021")

  wh_sort = wh_merged.sort_values(by='year', ascending=True)
  
  val_ls = wh_sort.groupby(['year','Regional_indicator']).agg(
    Ladder_Score_Avg=('Ladder_score', 'mean'),
    Log_GDP_per_capita_Avg=('Log_GDP_per_capita', 'mean'),
    Social_support_Avg=('Social_support', 'mean'),
    Healthy_life_expectancy_Avg=('Healthy_life_expectancy', 'mean'),
    Freedom_to_make_life_choices_Avg=('Freedom_to_make_life_choices', 'mean'),
    Generosity_Avg=('Generosity', 'mean'),
    Perceptions_of_corruption_Avg=('Perceptions_of_corruption', 'mean'),
    country_count=('Country_name', 'count')).reset_index()
  
  fig = px.line(val_ls,
                x="year", 
                y="Ladder_Score_Avg", 
                color='Regional_indicator', 
                #title='LadderScore_Region wise'
                )
  
  fig.update_layout(
    height=600,  
    width=1000,
    xaxis_title='Year',
    yaxis_title='Average Ladder Score',
    legend_title_text='Region',
    legend=dict(
        orientation="h",        
        yanchor="bottom",
        y=-0.5,                 
        xanchor="center",
        x=0.5)
    )

  st.plotly_chart(fig)

  st.write("")
  st.write("-", "The North America and ANZ and Wester Europe regions have the highest scores for the entire period. Sub-Saharan Africa has the lowest ladder scores, although these have been increasing since 2016.")
  st.write("-", "South Asia, which had the second-lowest ladder score in most years an it is the only region with a downward trend.")
  st.write("-", "One example of a relatively stable value is the Middle East and North Africa region. While an upward trend can be observed in Central and Eastern Europe, for instance.")    
  st.write("")
  st.write("")


  #WORLD MAP
  st.write("### World Happiness Map")

  st.write("""This map shows the ranking points of the individual countries during this period. 
  The number of countries surveyed gradually increased: 60% in 2006, 70% in 2007 and finally almost 100% in 2011. 
  The highest number of countries was surveyed in 2021.""")
  st.write("It is interesting to note that only 64% of countries were included in the survey in 2020. This sudden drop is due to the COVID restrictions.")

  fig = px.choropleth( wh_sort,
                      locations="Country_name",
                      locationmode="country names",
                      color="Ladder_score",
                      animation_frame = 'year',
                      #title="World Happiness Levels (Ladder Score)",
                      color_continuous_scale=px.colors.sequential.Plasma,
                      projection="kavrayskiy7")

  fig.update_layout(
    height=600,  
    width=1200,
    coloraxis_colorbar=dict(
        title="Ladder Score")
    )


  
  st.plotly_chart(fig)

  st.write("")
  st.write("")


  #TREND LADDER SCORES
  st.write("### Trend of the ladder score as a function of healthy life expectancy")
 
  st.write("""This chart shows the average ladder score of each region as a function of average healthy life expectancy from 2005 to 2021. 
          The size of the bubbles indicates the number of countries considered.""")

  fig = px.scatter(val_ls,
                 x = 'Healthy_life_expectancy_Avg',
                 y = 'Ladder_Score_Avg',
                 #animation_frame = 'year',
                 size = 'country_count',
                 color='Regional_indicator',
                 #title= 'Trend of LadderScore as function of Healthy life expectancy along with the considered countries'
                 )
  
  fig.update_layout(
    height=600,  
    width=1000,
    yaxis_range=[4,7.7],
    xaxis_range=[45,80],
    xaxis_title='Average Healthy life expectancy',
    yaxis_title='Average Ladder Score',
    legend_title_text='Region',
    legend=dict(
        orientation="h",        
        yanchor="bottom",
        y=-0.5,                 
        xanchor="center",
        x=0.5)
    )

  st.plotly_chart(fig)

  st.write("")
  st.write("-", "Healthy life expectancy improves for all regions over the years")
  st.write("-", "The region with the greatest improvement in life expectancy (by almost 10 years) is sub-Saharan Africa. This suggests that the medical infrastructure and availability of medicines have improved in this region")
  st.write("")
  st.write("")









#### MODELING ####
if page == pages[3] : 
  header_html = "<h2 style='color: black;'>Data Modelling</h2>"
  st.markdown(header_html, unsafe_allow_html=True)

  if st.button("Scores and Metrics") :
    data = {
        'Models': ['Linear Regression', 'Lasso', 'LassoCV', 'Ridge', 'DecisionTree Regressor', 'RandomForest Regressor'],
        'R² train': [0.7702, 0.7728, 0.7708, 0.7706, 0.9366, 0.8875],
        'R² test': [0.7917, 0.7225, 0.8142, 0.7934, 0.7757, 0.8604],
        #'MAE train': [],
        'MAE test': [0.4043, 0.4479, 0.3937, 0.4041, 0.3645, 0.3323],
        #'MSE train': [],
        'MSE test': [0.2735,0.3151, 0.2564, 0.2712, 0.2547, 0.1927],
        #'RMSE train': [],
        'RMSE test': [0.523, 0.5614, 0.5064 ,0.5208, 0.5047, 0.439]
                } 
    table = pd.DataFrame(data)
    RF_index = table[table["Models"] == "RandomForest Regressor"].index
    style = (
        table.style
        .apply(lambda x: ['background:rgb(116, 172, 246)' if x.name in RF_index else '' for _ in x], axis=1)
        .format(precision=4)
        )
      
    st.table(style)

  if st.button("RandomForest Regressor") :
    st.subheader("Visualizations")

    #image_collage_model = Image.open("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/collage_modeling.png")
    #st.image(image_collage_model, width=700)

    image_rf = Image.open("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/scatter_prediction.png")
    st.image(image_rf, width=700)
    st.write("-", "The points follow the red line quite closely, which indicates a strong match between the actual and predicted scores")


    image_rf2 = Image.open("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/scatter_residuals.png")
    st.image(image_rf2, width=700)
    st.write("-", "The residuals are randomly distributed around zero, meaning that the model does not show systematic errors")


    image_rf3 = Image.open("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/histogram.png")
    st.image(image_rf3, width=700)
    st.write("-", "It follows a bell-shaped curve and most residuals are close to zero, which is typical for a well-performing regression model.")

  if st.button("Feature Importance") :
    st.subheader("Feature importance of the RandomForest model")
    image_rf4 = Image.open("/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/feature_importance.png")
    st.image(image_rf4, width=700)        
    


#### PREDICTION ####
if page == pages[4] : 
  # Main header for Prediction
    header_html = "<h2 style='color: black;'>Prediction</h2>"
    st.markdown(header_html, unsafe_allow_html=True)

    # Subtitle below Prediction
    model_subtitle_html = "<h5 style='color: #808080;'>Random Forest Regressor Model</h5>"  # Smaller font, neutral color (gray)
    st.markdown(model_subtitle_html, unsafe_allow_html=True)

    # Load Random Forest model
    def load_model():
        model_path = r"/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/rf.pkl"
        with open(model_path, 'rb') as file:
            rf_model = pickle.load(file)
        return rf_model

    # Load dataset
    def load_dataset():
        data_path = r"/Users/stellahoffmann/Documents/Weiterbildung/Streamlit/world_happiness_forML.csv"
        return pd.read_csv(data_path)

    # Load model and dataset
    rf_model = load_model()
    ladder_scores = load_dataset()

    # Ranges definition
    feature_limits = {
        "Logged_GDP_per_capita": {"min": 6.63, "max": 11.65},
        "Healthy_life_expectancy": {"min": 32.30, "max": 77.10},
        "Social_support": {"min": 0.29, "max": 0.99},
        "Freedom_to_make_life_choices": {"min": 0.26, "max": 0.99},
    }

    # Desired features only
    include_features = ["Logged_GDP_per_capita", "Healthy_life_expectancy", "Social_support", "Freedom_to_make_life_choices"]

    characteristics_list = []

    # Sliders based on feature ranges
    for feature in include_features:
        limits = feature_limits[feature]
        characteristic = st.slider(
            f"{feature}", 
            float(limits['min']), 
            float(limits['max']), 
            float((limits['min'] + limits['max']) / 2)
        )
        characteristics_list.append(characteristic)

    # Input conversion
    characteristics = np.array([characteristics_list])

    # Predictions using Random Forest model
    prediction = rf_model.predict(characteristics)
    predicted_score = prediction[0]

    # Predicted Ladder Score display
    st.markdown(""" 
    <style> 
    .big-font {font-size:24px !important;}
    .green-font {font-size:24px !important; color: green;}  
    </style>
    """, unsafe_allow_html=True)
    st.markdown(
        '<p class="big-font">Predicted Ladder Score: &nbsp;&nbsp; <span class="green-font">{}</span></p>'.format(predicted_score),
        unsafe_allow_html=True
    )

    # Filter for year 2021
    if 'year' not in ladder_scores.columns:
        st.error("The dataset does not contain a 'year' column.")
    else:
        ladder_scores_2021 = ladder_scores[ladder_scores['year'] == 2021].copy()

    # Check if ladder_scores_2021
    if ladder_scores_2021.empty:
        st.error("No data available for the year 2021 in the dataset.")
    else:
        # Ranking logic
        new_row = pd.DataFrame({'Ladder_score': [predicted_score]})  # Wrap in a DataFrame
        ladder_scores_2021 = pd.concat([ladder_scores_2021, new_row], ignore_index=True)

        # Sort by Ladder_score descending order
        ladder_scores_2021 = ladder_scores_2021.sort_values(by='Ladder_score', ascending=False).reset_index(drop=True)

        # Count of countries
        countries_ahead_count = len(ladder_scores_2021) - ladder_scores_2021[ladder_scores_2021['Ladder_score'] == predicted_score].index[0] - 1

        # Display result
        st.markdown(f"**With these feature settings, this country would be ahead of approximately {countries_ahead_count} countries in 2021.**")

        # Remark
        st.markdown("*Considering only these four features*", unsafe_allow_html=True)




#### CONCLUSION ####
if page == pages[5] : 
  header_html = "<h2 style='color: black;'>Conclusion</h2>"
  st.markdown(header_html, unsafe_allow_html=True)

  # Text
  report_text = """
  <div style="text-align: justify; font-size: 18px; line-height: 1.6;">
  <strong>GDP per capita</strong> is the most influencing factor among all the features studied. 
  The model shows that it is far more important than the rest of our top-4 
  (<strong>Healthy_life_expectancy</strong>, <strong>Social_support</strong>, 
  <strong>Freedom_to_make_life_choices</strong>) when predicting overall happiness. 
  However, the other features presented contribute to simulations of different scenarios 
  where their importance might be influential for a <strong>Ladder Score</strong> prediction.
  <p></p>Expanding the dataset in the future could provide valuable insights into global trends, enabling more accurate forecasts. 
  Additionally, new factors may emerge as crucial determinants of happiness in a society, enhancing even more the predictions of the model.
  </div>
  """
  st.markdown(report_text, unsafe_allow_html=True)
