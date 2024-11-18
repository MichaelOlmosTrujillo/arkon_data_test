import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import altair as alt
import plotly.express as px
import pydeck as pdk
import time
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers # Capas de las redes neuronales
from tensorflow import keras # libreria de tensorflow
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, f1_score

# configuración página
st.set_page_config(
    page_title="Arkon test - Científico de datos",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
    )

ruta_train = './data/train_sin_nulos.csv'
ruta_test = './data/test_sin_nulos.csv'
ruta_df = './data/df_set.csv'
ruta_viajes_passholder_start_station = './data/'
ruta_viajes_passholder_start_station += 'cant_viajes_'
ruta_viajes_passholder_start_station += 'passholder_start_station.csv'
ruta_viajes_passholder_end_station = './data/cant_viajes_'
ruta_viajes_passholder_end_station += 'passholder_end_station.csv' 
ruta_grupo_planes_anio_ordenado = './data/grupo_planes_anio_ordenado.csv'

df_train= pd.read_csv(ruta_train, sep = ',')
df_test = pd.read_csv(ruta_test, sep = ',')
df = pd.read_csv(ruta_df, sep = ',')
viajes_passholder_start_station = pd.read_csv(
    ruta_viajes_passholder_start_station,
    sep = ','
)
viajes_passholder_end_station = pd.read_csv(
    ruta_viajes_passholder_end_station,
    sep = ','
)
grupo_planes_anio_ordenado = pd.read_csv(ruta_grupo_planes_anio_ordenado,
                                         sep = ',')
if 'Unnamed: 0' in grupo_planes_anio_ordenado.columns:
    grupo_planes_anio_ordenado.drop('Unnamed: 0', axis = 1, inplace= True)

placeholder = st.empty()
placeholder.empty()

titulo = '<p style="font-family:sans-serif; color:Green; font-size: 24px;'
titulo += 'text-align: center">'
titulo += 'Test Arkon - Vacante: Científico de datos</p>'

st.markdown(titulo, unsafe_allow_html=True)


# Convirtiendo las variables start_time, end_time a tipo fecha
df_train['start_time'] = pd.to_datetime(df_train['start_time'],
                                        format = 'mixed')
df_train['end_time'] = pd.to_datetime(df_train['end_time'],
                                        format = 'mixed')
df_test['start_time'] = pd.to_datetime(df_test['start_time'],
                                        format = 'mixed')
df_test['end_time'] = pd.to_datetime(df_test['end_time'],
                                        format = 'mixed')
df['start_time'] = pd.to_datetime(df['start_time'],
                                        format = 'mixed')
df['end_time'] = pd.to_datetime(df['end_time'],
                                        format = 'mixed')

# grupo_planes_anio_ordenado['start_year'] = pd.to_datetime(
#     grupo_planes_anio_ordenado['start_year'],
#     format = 'mixed'
# )

menu = option_menu(None, ["Análisis exploratorio de datos", 
                            "Modelo analítico"
                         ],
                icons = ['house', 'geo-alt-fill'],
                menu_icon = 'cast',
                default_index=0,
                orientation='horizontal')

if menu == "Análisis exploratorio de datos":
    with st.sidebar:
        st.title('Filtros')
        passholder_list = list(
            viajes_passholder_start_station['passholder_type'].unique()
        )
        passholder_sel = st.multiselect('Selecciona un tipo de passholder',
                                        passholder_list
                                        )
        if len(passholder_sel) > 0:
            viajes_passholder_start_station = viajes_passholder_start_station[
                viajes_passholder_start_station[
                    'passholder_type'
                    ].isin(passholder_sel)
                    ]
            viajes_passholder_end_station = viajes_passholder_end_station[
                viajes_passholder_end_station[
                    'passholder_type'
                    ].isin(passholder_sel)
                    ]
            grupo_planes_anio_ordenado = grupo_planes_anio_ordenado[
                grupo_planes_anio_ordenado[
                    'passholder_type'
                    ].isin(passholder_sel)
                    ]
             
    titulo_graf_1 = 'Cantidad de viajes por estación de Inicio del Viaje'

    fig = px.bar(viajes_passholder_start_station, 
                    x = 'start_station', 
                    y = 'cantidad_de_viajes',
                color = 'passholder_type', barmode = 'group',
                width=800, height = 500
        )
    
    fig.update_layout(barmode = 'group', 
                        bargroupgap = 0.1, 
                        xaxis_tickangle = -45,
                        title = titulo_graf_1)
    
    fig.update_traces(width = 0.2)
    st.plotly_chart(fig)
    
    titulo_graf_1 = 'Cantidad de viajes por estación Final del Viaje'

    fig = px.bar(viajes_passholder_end_station, 
                    x = 'end_station', 
                    y = 'cantidad_de_viajes',
                color = 'passholder_type', barmode = 'group',
                width=800, height = 500
        )
    
    fig.update_layout(barmode = 'group', 
                        bargroupgap = 0.1, 
                        xaxis_tickangle = -45,
                        title = titulo_graf_1)
    
    fig.update_traces(width = 0.2)
    st.plotly_chart(fig)

    texto_1 = '<p style="font-family:sans-serif; color:Blue; font-size: 18px;'
    texto_1 += 'text-align: center">'
    texto_1 += 'Cantidad de viajes por año </p>'
    st.markdown(texto_1, unsafe_allow_html=True)
    
    texto_2 = 'La cantidad de viajes por año es fluctuant. Es decir, tuvo un '
    texto_2 += 'crecimiento en 2016, 2017 y 2018, pero decreció en 2019, 2020 '
    texto_2 += 'y 2021.'
    st.markdown(texto_2)

    viajes_por_anio = df.groupby('start_year').size().to_frame().reset_index()
    viajes_por_anio = viajes_por_anio.rename(columns = {0:'cantidad_de_viajes'})
    viajes_por_anio = viajes_por_anio.sort_values(by = 'cantidad_de_viajes',
                                                ascending=True)
    st.dataframe(viajes_por_anio)
    texto_3 = '<p style="font-family:sans-serif; color:Blue; font-size: 18px;'
    texto_3 += 'text-align: center">'
    texto_3 += 'Cantidad de planes por año </p>'
    st.markdown(texto_3, unsafe_allow_html=True)
    # Create the time series plot
    
    texto_4 = 'Se observa que los planes por año no parece tener una tendencia '
    texto_4 += 'estable, con excepción del tipo de Passholder "One Day Pass" '
    texto_4 += 'que muestra una tendencia creciente.'

    st.markdown(texto_4)
    
    fig = px.line(
        grupo_planes_anio_ordenado,
        x='start_year',
        y='cantidad_de_planes',
        color='passholder_type',  # Line for each passholder_type
        markers=True,  # Add markers to the lines
        labels={
            'start_year': 'Año',
            'cantidad_de_planes': 'Cantidad de Planes',
            'passholder_type': 'Tipo de Passholder'
        },
        title='Número de planes por año filtrato por Tipo de Passholder'
    )

    # Customize layout (optional)
    fig.update_layout(
        xaxis=dict(title='Year'),
        yaxis=dict(title='Number of Plans'),
        legend_title='Passholder Type'
    )
    st.plotly_chart(fig)

elif menu == "Modelo analítico":
    ### Selección de variables numéricas para observar correlación entre las variables
    texto_6 = 'Seleccionar las variables numéricas del conjunto de entrenamiento con el fin de eliminar ' 
    texto_6 += 'las variables correlacionadas. '
    texto_6 += 'Se toma una correlación mayor a 0.8 para expresar una correlación fuerte entre '
    texto_6 += 'las variables.'
    st.markdown(texto_6)
    var_num = df_train.select_dtypes(['float64', 'int64'])
    var_num = var_num.drop(['trip_id'], axis = 1)
    st.dataframe(var_num.head())
    ## Correlación entre las variables
    correlation_matrix = var_num.corr()

    # Select correlations above 0.8
    correlation_threshold = 0.8

    # Get the pairs of columns with correlation > 0.8
    correlation_pairs = (
        correlation_matrix.where(lambda x: abs(x) > correlation_threshold)
        .stack()  # Convert to a long format
        .reset_index()  # Reset the index
    )

    # Filter out self-correlations (e.g., A with A)
    correlation_pairs = correlation_pairs[correlation_pairs['level_0'] != correlation_pairs['level_1']]

    # Rename columns for clarity
    correlation_pairs.columns = ['Column 1', 'Column 2', 'Correlation']
    texto_7 = 'Correlación entre las variables'
    st.markdown(texto_7)
    st.dataframe(correlation_pairs)
    texto_8 = 'Eliminación de las variables correlacionadas con un valor mayor a 0.8. '
    texto_8 += 'Se eliminan en el conjutno de entrenamiento y en el conjunto de testeo.'
    st.markdown(texto_8)
    columnas_a_borrar = ['start_lat', 'end_lat', 'start_station', 
                     'start_day', 'start_month', 'start_year',
                     'start_hour']
    columnas_a_borrar
    st.markdown('Conjunto de entrenamiento')
    df_train = df_train.drop(columnas_a_borrar, axis = 1)
    st.dataframe(df_train.head())
    st.markdown('Conjunto de testeo')
    df_test = df_test.drop(columnas_a_borrar, axis = 1)
    st.dataframe(df_test.head())
    st.markdown('Borrando la columna trip_id dado que todos sus valores son únicos')
    st.markdown('Conjunto de entrenamiento')
    df_train.drop('trip_id', axis = 1, inplace=True)
    st.dataframe(df_train.head())
    st.markdown('Conjunto de testeo')
    df_test.drop('trip_id', axis = 1, inplace=True)
    st.dataframe(df_test.head())
    texto_10 = 'Borrando las columnas start_time y end_time dado que ya están desagregadas por día, '
    texto_10 += 'mes, año, hora, minuto y segundos.'
    st.markdown(texto_10)
    df_train.drop(['start_time', 'end_time'], axis = 1, inplace=True)
    df_test.drop(['start_time', 'end_time'], axis = 1, inplace=True)
    st.markdown('Conjunto de entrenamiento')
    st.dataframe(df_train.head())
    st.markdown('Conjunto de testeo')
    st.dataframe(df_test.head())
    texto_11 = 'Se elimina la variable Plan Duration dado que solo aparece en el conjunto de '
    texto_11 += 'entrenamiento y no en el conjunto de testeo.'
    st.markdown(texto_11)
    df_train.drop('plan_duration', axis = 1, inplace=True)
    st.dataframe(df_train.head())
    st.markdown('Se codifican las variables categoricas usando LabelEncoder')
    label_encoder = LabelEncoder()
    df_train['trip_route_category'] = label_encoder.fit_transform(
    df_train['trip_route_category']
    )
    df_test['trip_route_category'] = label_encoder.fit_transform(
        df_test['trip_route_category']
    )
    df_train['passholder_type'] = label_encoder.fit_transform(
        df_train['passholder_type']
    )
    df_test['passholder_type'] = label_encoder.fit_transform(
        df_test['passholder_type']
    )
    texto_13 = 'Concatenamos los conjuntos de entrenamiento para crear un conjunto de entrenamiento '
    texto_13 += 'del 80\% de los datos y un conjunto de testeo del 20\% de datos restante.'
    st.markdown(texto_13)
    df = pd.concat([df_test, df_train],ignore_index=True)
    X = df.drop('passholder_type', axis = 1)
    y = df['passholder_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
    

    st.markdown('Creamos una red neuronal MLP como modelo base')
    model = load_model('./mlp_arkon.keras')
    test_loss, test_acc = model.evaluate(X_test, y_test)
    st.markdown('Resultado del testeo del modelo MLP con la métrica accuracy')
    st.text(test_acc)
    st.markdown('Se prueba un segundo modelo, en este caso un XGBoost con métrica f1_score: ')
    xgb_model = joblib.load("xgb_model.pkl")
    y_pred = xgb_model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.idxmax(axis=1)
    metricas = classification_report(y_test, y_pred)
# Evaluate
    st.text(metricas)
    st.markdown('Conclusiones')
    conclusion_1 = '1. Los modelos de MLP y XGBoost no son buenos modelos de base por lo tanto es necesario '
    conclusion_1 += 'usar variables adicionales como Plan Duration. Para esto, es necesario imputar datos '
    conclusion_1 += 'en dicha variable. Para imputar datos se podría realizar un modelo para su imputación.'
    st.markdown(conclusion_1)
