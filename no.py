#!/usr/bin/env python
# coding: utf-8

# ## Importo la base y formo el df

# In[182]:


get_ipython().system('pip install missingno')


# In[183]:


import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[184]:


# Set the path to your CSV file in your computer
ruta_del_archivo = "/Users/vsaguier/Downloads/suscriptores_vica.csv"

# Define the names of each column in the DataFrame
columnas = ['año_nac', 'genero', 'id_documento', 'localidad', 'CP_Socio', 'lat_socio',
           'long_socio', 'prov_socio', 'tipocredencial', 'cat_credencial', 'condicion',
           'fecha', 'pago_socio', 'monto_descuento', 'porc_descuento', 'tipo_transaccion',
           'credencial_virtual', 'RazonSocial', 'NombreCuenta', 'NombreCuentaPadre', 'rubro',
           'subrubro', 'lat_comercio', 'long_comercio', 'pais_comercio', 'Prov_comercio',
           'ciudad_comercio']

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(ruta_del_archivo, names=columnas, sep=';', error_bad_lines=False)

# Print the first few rows of the DataFrame to confirm it was read correctly
print(df.head())


# ## Introducción a la base

# In[185]:


df.shape


# ## Calculo de nulos

# In[186]:


#calculo de nulos y porcentaje sobre el total

# Calcular la cantidad de nulos por campo y el porcentaje de nulos respecto al total de registros
null_counts = df.isnull().sum()
null_percentages = round((null_counts / len(df)) * 100, 2)

# Crear una tabla con la información de nulos y porcentajes
null_table = pd.concat([null_counts, null_percentages], axis=1)
null_table.columns = ['Nulos', 'Porcentaje de Nulos (%)']

# Imprimir la tabla resultante
print(null_table)


# ## Elimino columnas

# In[187]:


# Crear el DataFrame
df = pd.DataFrame(df) 

# Lista de columnas a eliminar
columns_to_drop = ['credencial_virtual', 'pais_comercio','lat_socio', 'long_socio', 'lat_comercio', 'long_comercio', 'monto_descuento', 'tipocredencial', 'RazonSocial', 'NombreCuentaPadre', 'CP_Socio', 'tipo_transaccion']

# Eliminar las columnas
df = df.drop(columns=columns_to_drop)

# Verificar el resultado
print(df.head())  # Imprime las primeras filas del DataFrame sin las columnas eliminadas


# ## Cambio tipos de datos y id_documento

# In[188]:


df['año_nac'] = pd.to_numeric(df['año_nac'], errors='coerce', downcast='integer')


# Reemplazar la coma por un punto decimal
df['pago_socio'] = df['pago_socio'].str.replace(',', '.')

# Convertir a tipo numérico
df['pago_socio'] = pd.to_numeric(df['pago_socio'], errors='coerce')

# Reemplazar la coma por un punto decimal
df['porc_descuento'] = df['porc_descuento'].str.replace(',', '.')

# Convertir a tipo numérico
df['porc_descuento'] = pd.to_numeric(df['porc_descuento'], errors='coerce')

df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

df['año_nac'] = pd.to_numeric(df['año_nac'], errors='coerce', downcast='integer')


# ## Creo edad y elimino año nac

# In[189]:


# Calculamos la edad como la resta entre el año actual y el año de nacimiento
edad = pd.Timestamp('now').year - df['año_nac']

# Agregamos la columna de edad al dataframe
df['edad'] = edad


# In[190]:


#elimino nulos por edad de df
df = df.dropna(subset=['edad'])


# In[191]:


# Crear el DataFrame
df = pd.DataFrame(df) 

# Lista de columnas a eliminar
columns_to_drop = ['año_nac']

# Eliminar las columnas
df = df.drop(columns=columns_to_drop)

# Verificar el resultado
print(df.head())  # Imprime las primeras filas del DataFrame sin las columnas eliminadas


# ## creo rango_etario

# In[192]:


# Función para definir los rangos etarios
def obtener_rango_etario(edad):
    if edad <= 15:
        return '0-15'
    elif edad <= 30:
        return '16-30'
    elif edad <= 45:
        return '31-45'
    elif edad <= 60:
        return '46-60'
    elif edad <= 75:
        return '61-75'
    elif edad <= 90:
        return '75-90'
    elif edad > 90:
        return '90-100'
    else:
        return edad

# Crear la columna de rango_etario
df['rango_etario'] = df['edad'].apply(obtener_rango_etario)

# Imprimir el dataframe con la nueva columna
print(df.head())


# ## Prov comercio limpieza y eliminacion

# In[193]:


# Creamos una lista con las provincias permitidas
provincias_permitidas = ['CAPITAL FEDERAL', 'BUENOS AIRES']

# Filtramos el dataset para obtener solo los registros con las provincias permitidas
df = df[df['Prov_comercio'].isin(provincias_permitidas)]


# In[194]:


# Crear el DataFrame
df = pd.DataFrame(df) 

# Lista de columnas a eliminar
columns_to_drop = ['Prov_comercio', 'prov_socio']

# Eliminar las columnas
df = df.drop(columns=columns_to_drop)

# Verificar el resultado
df.info()  # Imprime las primeras filas del DataFrame sin las columnas eliminadas


# ## Partido, zona

# In[195]:


def obtener_partido(localidad):
    if localidad in ['CIUDAD AUTONOMA BUENOS AIRES', 'CABA', 'Capital Federal', 'CAPITAL FEDERAL', 'Capital federal', 'CIUDAD AUTONOMA DE BUENOS AIRES']:
        return 'CABA'
    elif localidad in ['SAN ISIDRO', 'MARTINEZ', 'BECCAR', 'BOULOGNE', 'ACASSUSO', 'MARTÍNEZ', 'VILLA ADELINA', 'Boulogne']: #agregue uno
        return 'San Isidro'
    elif localidad in ['OLIVOS', 'FLORIDA', 'LA LUCILA', 'MUNRO', 'CARAPACHAY', 'VICENTE LOPEZ', 'FLORIDA OESTE', 'VILLA MARTELLI', 'Munro']:
        return 'Vicente Lopez'
    elif localidad in ['TIGRE', 'DON TORCUATO', 'GENERAL PACHECO', 'EL TALAR', 'TRONCOS DEL TALAR', 'BENAVIDEZ', 'DIQUE LUJAN', 'NORDELTA', 'RINCON DE MILBERG', 'TRONCOS DEL TALAR', 'TRONCOS DE TALAR', 'PACHECO', 'RICARDO ROJAS']:
        return 'Tigre'
    elif localidad in ['PILAR', 'MANUEL ALBERTI', 'TORTUGAS', 'TORTUGUITAS', 'GARIN', 'DEL VISO', 'VILLA ROSA', 'PRESIDENTE DERQUI', 'FATIMA ESTACION EMPALME']:
        return 'Pilar'
    elif localidad in ['VICTORIA', 'SAN FERNANDO', 'VIRREYES']:
        return 'San Fernando'
    elif localidad in ['QUILMES', 'BERNAL ESTE', 'QUILMES OESTE', 'BERNAL OESTE', 'DON BOSCO', 'EZPELETA ESTE']: #agregue uno
        return 'Quilmes'
    elif localidad in ['RAMOS MEJIA', 'TAPIALES','CIUDAD EVITA', 'SAN JUSTO', 'CIUDAD MADERO', 'LOMAS DEL MIRADOR', 'ISIDRO CASANOVA', 'TABLADA', 'VILLA LUZURIAGA', 'GREGORIO DE LAFERRERE', 'GONZALEZ CATAN']:
        return 'La Matanza'
    elif localidad in ['LOMAS DE ZAMORA', 'BANFIELD', 'TEMPERLEY', 'TURDERA', 'LLAVALLOL']:
        return 'Lomas de Zamora'
    elif localidad in ['BELLA VISTA', 'SAN MIGUEL', 'MUÑIZ']: #agregue uno
        return 'San Miguel'
    elif localidad in ['MAR DEL PLATA', 'PINAMAR', 'CARILO', 'VALERIA DEL MAR']:
        return 'Costa'
    elif localidad in ['AVELLANEDA', 'WILDE', 'SARANDI', 'GERLI', 'PIÑEYRO', 'VILLA DOMINICO', 'CRUCESITA']:
        return 'Avellaneda'
    elif localidad in ['LANUS', 'REMEDIOS DE ESCALADA', 'VALENTIN ALSINA', 'MONTE CHINGOLO', 'REMEDIOS DE ESCALADA']:
        return 'Lanus'
    elif localidad in ['CANNING', 'EZEIZA', 'TRISTAN SUAREZ', 'LA UNION']:
        return 'Ezeiza'
    elif localidad in ['MORON', 'CASTELAR', 'HAEDO', 'EL PALOMAR', 'VILLA SARMIENTO', 'HAEDO']: #aca agregue uno
        return 'Moron'
    elif localidad in ['DOLORES']:
        return 'Dolores'
    elif localidad in ['BELEN DE ESCOBAR', 'INGENIERO MASCHWITZ', 'MAQUINISTA F SAVIO', 'MATHEU', 'ESCOBAR']:
        return 'Escobar'
    elif localidad in ['LA PLATA', 'CITY BELL', 'MANUEL B GONNET']:
        return 'La Plata'
#aca empiezan los que agrego
    elif localidad in ['ITUZAINGO', 'BARRIO PARQUE LELOIR', 'VILLA GOBERNADOR UDAONDO']:
        return 'Ituzaingó'
    elif localidad in ['HURLINGHAM', 'VILLA SANTOS TESEI']:
        return 'Hurlingham'
    elif localidad in ['VILLA BALLESTER', 'GENERAL SAN MARTIN','JOSE LEON SUAREZ', 'SAN ANDRES', 'SAN MARTIN']:
        return 'General San Martín'
    elif localidad in ['GRAND BOURG', 'LOS POLVORINES','VILLA DE MAYO', 'PABLO NOGUES']:
        return 'Malvinas Argentinas'
    elif localidad in ['MONTE GRANDE', 'LUIS GUILLON']:
        return 'Esteban Echeverría'
    elif localidad in ['MORENO', 'FRANCISCO ALVAREZ', 'PASO DEL REY', 'LA REJA', 'FRANCISCO ÁLVAREZ']:
        return 'Moreno'
    elif localidad in ['CAMPANA']:
        return 'Campana'
    elif localidad in ['ALMIRANTE BROWN', 'BURZACO', 'ADROGUE', 'JOSE MARMOL', 'CLAYPOLE']:
        return 'Almirante Brown'
    elif localidad in ['SAN ANTONIO DE PADUA', 'MERLO']:
        return 'Merlo'
    elif localidad in ['GUILLERMO E HUDSON', 'HUDSON', 'BERAZATEGUI', 'RANELAGH']:
        return 'Berazategui'
    elif localidad in ['CASEROS', 'CIUDADELA', 'CIUDAD JARDIN DEL PALOMAR', 'SANTOS LUGARES', 'VILLA SAENZ PEÑA', 'VILLA BOSCH', 'MARTIN CORONADO', 'VILLA RAFFO']:
        return 'Tres de Febrero'
    elif localidad in ['JOSE CLEMENTE PAZ']:
        return 'José C. Paz'
    elif localidad in ['CORDOBA']: #este es una ciudad
        return 'Córdoba'
    elif localidad in ['LUJAN']:
        return 'Lujan'
    elif localidad in ['ZARATE']:
        return 'Zarate'
    elif localidad in ['MERCEDES']:
        return 'Mercedes'
    elif localidad in ['BAHIA BLANCA']:
        return 'Bahia Blanca'
    elif localidad in ['FLORENCIO VARELA']:
        return 'Florencio Varela'
    elif localidad in ['GENERAL RODRIGUEZ']:
        return 'General Rodriguez'

#aca termina
    elif localidad in ['No Informa']:
        return 'NA'
    else:
        return 'No especificado'

# Aplicamos la función a la columna de localidades y creamos una nueva columna llamada "Partido"
df['zona_partido_suscriptor'] = df['localidad'].apply(obtener_partido)


# In[196]:


def obtener_partido(ciudad_comercio):
    if ciudad_comercio in ['CIUDAD AUTONOMA BUENOS AIRES', 'CABA', 'Capital Federal', 'CAPITAL FEDERAL', 'Capital federal', 'CIUDAD AUTONOMA DE BUENOS AIRES']:
        return 'CABA'
    elif ciudad_comercio in ['SAN ISIDRO', 'MARTINEZ', 'BECCAR', 'BOULOGNE', 'ACASSUSO', 'MARTÍNEZ', 'VILLA ADELINA', 'Boulogne']: #aca agregue uno
        return 'San Isidro'
    elif ciudad_comercio in ['OLIVOS', 'FLORIDA', 'LA LUCILA', 'MUNRO', 'CARAPACHAY', 'VICENTE LOPEZ', 'VILLA MARTELLI']:
        return 'Vicente Lopez'
    elif ciudad_comercio in ['TIGRE', 'DON TORCUATO', 'GENERAL PACHECO', 'EL TALAR', 'TRONCOS DEL TALAR', 'BENAVIDEZ', 'DIQUE LUJAN', 'NORDELTA', 'RINCON DE MILBERG', 'TRONCOS DEL TALAR', 'TRONCOS DE TALAR', 'PACHECO', 'RICARDO ROJAS']:
        return 'Tigre'
    elif ciudad_comercio in ['PILAR', 'MANUEL ALBERTI', 'TORTUGAS', 'TORTUGUITAS', 'GARIN', 'DEL VISO', 'PRESIDENTE DERQUI', 'FATIMA ESTACION EMPALME']:
        return 'Pilar'
    elif ciudad_comercio in ['VICTORIA', 'SAN FERNANDO', 'VIRREYES']:
        return 'San Fernando'
    elif ciudad_comercio in ['QUILMES', 'BERNAL ESTE', 'QUILMES OESTE', 'BERNAL OESTE', 'DON BOSCO', 'EZPELETA ESTE']:
        return 'Quilmes'
    elif ciudad_comercio in ['RAMOS MEJIA', 'TAPIALES', 'SAN JUSTO', 'CIUDAD EVITA', 'SAN JUSTO', 'CIUDAD MADERO', 'LOMAS DEL MIRADOR', 'ISIDRO CASANOVA', 'TABLADA', 'VILLA LUZURIAGA', 'GREGORIO DE LAFERRERE', 'GONZALEZ CATAN']: #aca agregue uno
        return 'La Matanza'
    elif ciudad_comercio in ['LOMAS DE ZAMORA', 'BANFIELD', 'TEMPERLEY', 'TURDERA', 'LLAVALLOL']:
        return 'Lomas de Zamora'
    elif ciudad_comercio in ['BELLA VISTA', 'SAN MIGUEL', 'MUÑIZ']: #aca agrege uno
        return 'San Miguel'
    elif ciudad_comercio in ['MAR DEL PLATA', 'PINAMAR', 'CARILO', 'VILLA GESELL', 'VALERIA DEL MAR']:
        return 'Costa'
    elif ciudad_comercio in ['AVELLANEDA', 'WILDE', 'SARANDI', 'GERLI', 'PIÑEYRO', 'VILLA DOMINICO', 'CRUCESITA']:
        return 'Avellaneda'
    elif ciudad_comercio in ['LANUS', 'LANUS OESTE', 'VALENTIN ALSINA', 'MONTE CHINGOLO', 'REMEDIOS DE ESCALADA']: #aca agregue uno
        return 'Lanus'
    elif ciudad_comercio in ['CANNING', 'EZEIZA', 'TRISTAN SUAREZ', 'LA UNION']:
        return 'Ezeiza'
    elif ciudad_comercio in ['MORON', 'CASTELAR', 'EL PALOMAR', 'VILLA SARMIENTO', 'HAEDO']:
        return 'Moron'
    elif ciudad_comercio in ['DOLORES']:
        return 'Dolores'
    elif ciudad_comercio in ['BELEN DE ESCOBAR', 'INGENIERO MASCHWITZ', 'MAQUINISTA F SAVIO', 'MATHEU', 'ESCOBAR']:
        return 'Escobar'
    elif ciudad_comercio in ['LA PLATA', 'CITY BELL', 'MANUEL B GONNET']:
        return 'La Plata'
    elif ciudad_comercio in ['GRAND BOURG', 'MALVINAS ARGENTINAS', 'PABLO NOGUES']:
        return 'Malvinas Argentinas'
    elif ciudad_comercio in ['BAHIA BLANCA']:
        return 'Bahia Blanca'
 #Aca empiezan los que no estaban agregados
    elif ciudad_comercio in ['MORENO', 'FRANCISCO ALVAREZ', 'PASO DEL REY', 'LA REJA', 'FRANCISCO ÁLVAREZ']:
        return 'Moreno'
    elif ciudad_comercio in ['ENSENADA']:
        return 'Ensenada'
    elif ciudad_comercio in ['CASEROS', 'CIUDADELA', 'CIUDAD JARDIN DEL PALOMAR', 'SANTOS LUGARES', 'VILLA SAENZ PEÑA', 'VILLA BOSCH', 'MARTIN CORONADO', 'VILLA RAFFO']:
        return 'Tres de Febrero'
    elif ciudad_comercio in ['ITUZAINGO', 'BARRIO PARQUE LELOIR', 'VILLA GOBERNADOR UDAONDO']:
        return 'Ituzaingó'
    elif ciudad_comercio in ['GUILLERMO E HUDSON', 'HUDSON', 'BERAZATEGUI', 'RANELAGH']:
        return 'Berazategui'
    elif ciudad_comercio in ['GENERAL RODRIGUEZ']:
        return 'General Rodriguez'
    elif ciudad_comercio in ['BERNAL', 'QUILMES OESTE']:
        return 'Quilmes'
    elif ciudad_comercio in ['CAMPANA']:
        return 'Campana'
    elif ciudad_comercio in ['TANDIL']:
        return 'Tandil'
    elif ciudad_comercio in ['VILLA BALLESTER', 'GENERAL SAN MARTIN','JOSE LEON SUAREZ', 'SAN ANDRES', 'SAN MARTIN']:
        return 'General San Martín'
    elif ciudad_comercio in ['HURLINGHAM', 'VILLA SANTOS TESEI']:
        return 'Hurlingham'
    elif ciudad_comercio in ['SAN ANTONIO DE PADUA']:
        return 'Merlo'
    elif ciudad_comercio in ['MONTE GRANDE', 'LUIS GUILLON']:
        return 'Esteban Echeverría'
    elif ciudad_comercio in ['LUJAN']:
        return 'Lujan'
    elif ciudad_comercio in ['ZARATE']:
        return 'Zarate'
    elif ciudad_comercio in ['MERCEDES']:
        return 'Mercedes'
    elif ciudad_comercio in ['ALMIRANTE BROWN', 'BURZACO', 'ADROGUE', 'JOSE MARMOL', 'CLAYPOLE']:
        return 'Almirante Brown'
    elif ciudad_comercio in ['FLORENCIO VARELA']:
        return 'Florencio Varela'
    elif ciudad_comercio in ['GENERAL RODRIGUEZ']:
        return 'General Rodriguez'
        
#aca termina
    elif ciudad_comercio in ['No Informa']:
        return 'NA'
    else:
        return 'No especificado'


# Aplicamos la función a la columna de localidades y creamos una nueva columna llamada "Partido"
df['zona_partido_comercio'] = df['ciudad_comercio'].apply(obtener_partido)


# In[197]:


# Crear el DataFrame
df = pd.DataFrame(df) 

# Lista de columnas a eliminar
columns_to_drop = ['localidad', 'ciudad_comercio']

# Eliminar las columnas
df = df.drop(columns=columns_to_drop)


# In[198]:


# Contamos cuántos registros hay en cada categoría
conteos = df['zona_partido_suscriptor'].value_counts()

# Calculamos el porcentaje que representa cada categoría sobre el total
porcentajes = conteos / len(df) * 100

# Creamos un nuevo DataFrame con los conteos y los porcentajes
resultados = pd.DataFrame({'conteo': conteos, 'porcentaje': porcentajes})

# Imprimimos los resultados
print(resultados)


# crear cant_beneficios

# In[199]:


import matplotlib.pyplot as plt

def add_cant_beneficios_column(df):
    beneficios_count = df.groupby('id_documento')['NombreCuenta'].nunique()
    beneficios_count.name = 'cant_beneficios'
    df = df.join(beneficios_count, on='id_documento')
    return df

def plot_cant_beneficios(df, title):
    cant_beneficios_counts = df['cant_beneficios'].value_counts().sort_index()
    ax = cant_beneficios_counts.plot(kind='bar', figsize=(10, 6), rot=0)
    ax.set_xlabel('Cantidad de Beneficios')
    ax.set_ylabel('Frecuencia')
    ax.set_title(title)

# Asumiendo que df ya está definido
df = add_cant_beneficios_column(df)

# Graficar la columna 'cant_beneficios' en el dataframe df
plt.figure()
plot_cant_beneficios(df, "Distribución de 'cant_beneficios'")
plt.show()


# In[200]:


df.info()


# ## Limpieza de valores incoherentes
# 

# In[201]:


df = df[df['cat_credencial'] != 'Classic']
df.shape


# In[202]:


df = df[df['edad'] > 0]
df = df[df['edad'] <=100]


# In[203]:


# Eliminar valores menores a 0 de monto_descuento, pago_socio y porc_descuento
df = df[df['pago_socio'] > 0]
df = df[df['porc_descuento'] > 0]


# In[204]:


# Calcular el rango intercuartil
Q1 = df['pago_socio'].quantile(0.25)
Q3 = df['pago_socio'].quantile(0.75)
IQR = Q3 - Q1

# Calcular los límites inferior y superior
lim_inf = Q1 - 1.5 * IQR
lim_sup = Q3 + 1.5 * IQR

# Filtrar los valores que estén dentro del rango
df = df[(df['pago_socio'] >= lim_inf) & (df['pago_socio'] <= lim_sup)]


# In[205]:


df.shape


# ## **Modelo RFM Segmentation**
# base para rfm
# ## 
# Recency: Number of days since customer's last purchase
# 
# Monetory Value: Total monetary value the customer spent on
# Monetory Value = MntWines+ MntFruits+ MntMeatProducts+ MntFishProducts+ MntSweetProducts+ MntGoldProds
# Frequency = NumWebPurchases + NumCatalogPurchases + NumStorePurchases

# In[206]:


df2=df


# In[207]:


df2.info()


# In[208]:


# identificar la fecha más reciente en el conjunto de datos
fecha_mas_reciente = df2["fecha"].max()

# calcular la diferencia en días entre cada fecha de compra y la fecha más reciente
df2["Recency"] = (fecha_mas_reciente - df2["fecha"]).dt.days

# contar la cantidad de compras que hizo cada persona
frecuencia_compras = df2.groupby("id_documento").size().reset_index(name="Frequency")

# agregar la columna "Frequency" al DataFrame
df2 = pd.merge(df2, frecuencia_compras, on="id_documento")

# calcular la suma de los valores en la columna "pago_socio" por cada id
monetary_value = df2.groupby("id_documento")["pago_socio"].sum()

# agregar la columna "Monetary Value" al DataFrame y asignar los valores correspondientes
df2["Monetary Value"] = df2["id_documento"].apply(lambda x: monetary_value[x])


# In[209]:


df2.info()


# In[210]:


df2.head(10)


# In[211]:


import seaborn as sns

# Graficar boxplot de Recency
sns.boxplot(data=df2, y='Recency')
plt.title('Boxplot - Recency')
plt.show()

# Graficar boxplot de Frequency
sns.boxplot(data=df2, y='Frequency')
plt.title('Boxplot - Frequency')
plt.show()

# Graficar boxplot de Monetary Value
sns.boxplot(data=df2, y='Monetary Value')
plt.title('Boxplot - Monetary Value')
plt.show()


# In[212]:


import pandas as pd

# Definir los límites para los outliers (1.5 veces el IQR)
lower_bound = 1.5
upper_bound = 1.5

# Calcular los cuartiles y el rango intercuartílico para cada variable
Q1 = df2['Recency'].quantile(0.25)
Q3 = df2['Recency'].quantile(0.75)
IQR = Q3 - Q1

# Calcular los límites inferiores y superiores para la variable Recency
lower_bound_recency = Q1 - lower_bound * IQR
upper_bound_recency = Q3 + upper_bound * IQR

# Eliminar los outliers de la variable Recency
df2 = df2[(df2['Recency'] >= lower_bound_recency) & (df2['Recency'] <= upper_bound_recency)]

# Calcular los cuartiles y el rango intercuartílico para la variable Frequency
Q1 = df2['Frequency'].quantile(0.25)
Q3 = df2['Frequency'].quantile(0.75)
IQR = Q3 - Q1

# Calcular los límites inferiores y superiores para la variable Frequency
lower_bound_frequency = Q1 - lower_bound * IQR
upper_bound_frequency = Q3 + upper_bound * IQR

# Eliminar los outliers de la variable Frequency
df2 = df2[(df2['Frequency'] >= lower_bound_frequency) & (df2['Frequency'] <= upper_bound_frequency)]

# Calcular los cuartiles y el rango intercuartílico para la variable Monetary Value
Q1 = df2['Monetary Value'].quantile(0.25)
Q3 = df2['Monetary Value'].quantile(0.75)
IQR = Q3 - Q1

# Calcular los límites inferiores y superiores para la variable Monetary Value
lower_bound_monetary = Q1 - lower_bound * IQR
upper_bound_monetary = Q3 + upper_bound * IQR

# Eliminar los outliers de la variable Monetary Value
df2 = df2[(df2['Monetary Value'] >= lower_bound_monetary) & (df2['Monetary Value'] <= upper_bound_monetary)]


# In[213]:


import seaborn as sns

# Graficar boxplot de Recency
sns.boxplot(data=df2, y='Recency')
plt.title('Boxplot - Recency')
plt.show()

# Graficar boxplot de Frequency
sns.boxplot(data=df2, y='Frequency')
plt.title('Boxplot - Frequency')
plt.show()

# Graficar boxplot de Monetary Value
sns.boxplot(data=df2, y='Monetary Value')
plt.title('Boxplot - Monetary Value')
plt.show()


# In[214]:


df2.shape


# In[215]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Preparación de los datos
df2 = df2.dropna()  # Eliminando las filas con valores nulos

# Creando las variables RFM
rfm_df = df2.groupby('id_documento').agg({
    'Recency': 'min',
    'Frequency': 'count',
    'Monetary Value': 'sum'
})

# Normalizando las variables RFM
scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df)

# Determinando el número óptimo de clusters
wcss = []
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(rfm_df_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(rfm_df_scaled, kmeans.labels_))

# Gráfico del método del codo (WCSS)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
plt.title('Método del codo')

# Gráfico del Silhouette Score
plt.subplot(122)
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')

plt.tight_layout()
plt.show()

# Aplicando el algoritmo K-Means con el número óptimo de clusters determinado por el método del codo
optimal_clusters = 3  # Este valor debe ser determinado por el método del codo
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(rfm_df_scaled)

# Añadiendo los clusters al dataframe original
rfm_df['Cluster'] = clusters

# Interpretando los clusters
cluster_averages = rfm_df.groupby('Cluster').mean()
print(cluster_averages)

# Evaluación del modelo
silhouette_avg = silhouette_score(rfm_df_scaled, clusters)
davies_bouldin_avg = davies_bouldin_score(rfm_df_scaled, clusters)
wcss_cluster = kmeans.inertia_

print(f'Silhouette Score: {silhouette_avg}')
print(f'Davies-Bouldin Index: {davies_bouldin_avg}')
print(f'Within-Cluster Sum of Squares: {wcss_cluster}')


# In[216]:


pip install plotly


# In[217]:


df2.info()


# In[218]:


rfm_df.head(5)


# In[219]:


rfm_df.describe()


# In[220]:


rfm_df.shape


# In[221]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Preparación de los datos
df2 = df2.dropna()  # Eliminando las filas con valores nulos

# Creando las variables RFM
rfm_df = df2.groupby('id_documento').agg({
    'Recency': 'min',
    'Frequency': 'count',
    'Monetary Value': 'sum'
})

# Normalizando las variables RFM
scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df)

# Aplicando el algoritmo K-Means con el número óptimo de clusters determinado por el método del codo
optimal_clusters = 3  # Este valor debe ser determinado por el método del codo
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(rfm_df_scaled)

# Añadiendo los clusters al dataframe original
rfm_df['Cluster'] = clusters

# Visualización en 3D con Plotly
fig = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary Value',
                    color='Cluster', symbol='Cluster', opacity=0.8, size_max=10)
fig.update_layout(scene=dict(xaxis_title='Recency', yaxis_title='Frequency', zaxis_title='Monetary Value'))
fig.show()


# In[232]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.offline as offline

# Preparación de los datos
df2 = df2.dropna()  # Eliminando las filas con valores nulos

# Creando las variables RFM
rfm_df = df2.groupby('id_documento').agg({
    'Recency': 'min',
    'Frequency': 'count',
    'Monetary Value': 'sum'
})

# Normalizando las variables RFM
scaler = StandardScaler()
rfm_df_scaled = scaler.fit_transform(rfm_df)

# Aplicando el algoritmo K-Means con el número óptimo de clusters determinado por el método del codo
optimal_clusters = 3  # Este valor debe ser determinado por el método del codo
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(rfm_df_scaled)

# Añadiendo los clusters al dataframe original
rfm_df['Cluster'] = clusters

# Graficar en 3D con Plotly
scatter = go.Scatter3d(
    x=rfm_df['Recency'],
    y=rfm_df['Frequency'],
    z=rfm_df['Monetary Value'],
    mode='markers',
    marker=dict(
        size=3.5,
        color=rfm_df['Cluster'],
        colorscale='Viridis',
        opacity=0.8
    )
)

layout = go.Layout(
    title='Segmentación de Suscriptores',
    scene=dict(
        xaxis=dict(title='Recency'),
        yaxis=dict(title='Frequency'),
        zaxis=dict(title='Monetary Value')
    )
)

fig = go.Figure(data=[scatter], layout=layout)
offline.plot(fig, filename='segmentation_3d.html')


# In[ ]:





# In[ ]:





# # umaps

# In[ ]:





# anterior

# In[ ]:


from scipy import stats
#df2["Monetary Value"] = stats.boxcox(df2['Monetary Value'])[0]
#df2["Frequency"] = stats.boxcox(df2['Frequency'])[0]
#df2["Recency"] = pd.Series(np.cbrt(df2['Recency'])).values


# In[ ]:


# Mostrar los últimos 10 registros de df2
df2.head(10)


# In[ ]:


quintiles = df2[['Recency', 'Frequency', 'Monetary Value']].quantile([.2, .4, .6, .8]).to_dict()


# In[ ]:


def r_score(x):
    if x <= quintiles['Recency'][.2]:
        return 5
    elif x <= quintiles['Recency'][.4]:
        return 4
    elif x <= quintiles['Recency'][.6]:
        return 3
    elif x <= quintiles['Recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5  


# In[ ]:


df2['R'] = df2['Recency'].apply(lambda x: r_score(x))
df2['F'] = df2['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
df2['M'] = df2['Monetary Value'].apply(lambda x: fm_score(x, 'Monetary Value'))


# In[ ]:


df2['RFM Score'] = df2['R'].map(str) + df2['F'].map(str) + df2['M'].map(str)


# In[ ]:


segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

df2['Segment'] = df2['R'].map(str) + df2['F'].map(str)
df2['Segment'] = df2['Segment'].replace(segt_map, regex=True)
df2.head()


# grafico 3D

# In[ ]:


# Importando las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Seleccionando los registros únicos de 'id_documento'
df2 = df2.drop_duplicates(subset=['id_documento'])

# Seleccionando las columnas de interés y aplicando la transformación de escala
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df2[['Recency', 'Frequency', 'Monetary Value']]), 
                         columns=['Recency', 'Frequency', 'Monetary Value'])

# Creando una figura de Matplotlib
fig = plt.figure()

# Agregando un subplot tridimensional
ax = fig.add_subplot(111, projection='3d')

# Agregando los puntos al gráfico
ax.scatter(df_scaled['Recency'], df_scaled['Frequency'], df_scaled['Monetary Value'], depthshade=True)

# Agregando etiquetas a los ejes
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary Value')

# Agregando un título al gráfico
ax.set_title('Gráfico 3D interactivo de Recency, Frequency y Monetary Value')

# Mostrando el gráfico
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Asegurarse de que df2 está definido y tiene las columnas correctas.
assert 'df2' in globals(), "df2 no está definido"
assert 'Recency' in df2.columns, "df2 no tiene una columna 'Recency'"
assert 'Frequency' in df2.columns, "df2 no tiene una columna 'Frequency'"
assert 'Monetary Value' in df2.columns, "df2 no tiene una columna 'Monetary Value'"
assert 'id_documento' in df2.columns, "df2 no tiene una columna 'id_documento'"

# Eliminar duplicados de id_documento
df2_unique = df2.drop_duplicates(subset='id_documento')

# Escalar las variables
scaler = StandardScaler()
df2_scaled = scaler.fit_transform(df2_unique[['Recency', 'Frequency', 'Monetary Value']])

# Crear un nuevo dataframe con los datos escalados
df2_scaled = pd.DataFrame(df2_scaled, columns=['Recency', 'Frequency', 'Monetary Value'])

# Eliminar los valores extremos usando los percentiles
low = .05
high = .95
quant_df = df2_scaled.quantile([low, high])
df2_filtered = df2_scaled.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                    (x < quant_df.loc[high,x.name])], axis=0)

df2_filtered = df2_filtered.dropna()

# Crear un gráfico interactivo de 3 dimensiones
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = df2_filtered['Recency']
y = df2_filtered['Frequency']
z = df2_filtered['Monetary Value']

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary Value')

plt.show()


# In[ ]:


df2.head(10)


# In[ ]:


df.info()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

# Asumimos que df2 es tu DataFrame y que ya ha sido definido
# df2 = pd.read_csv("ruta_a_tu_archivo.csv") # descomenta esta línea si necesitas cargar los datos desde un archivo CSV

# Eliminamos las filas duplicadas basadas en la columna 'id_documento_unico'
df2 = df2.drop_duplicates(subset=['id_documento'], keep='first')

# Creamos una copia del dataframe para no alterar los datos originales
df_scaled = df2.copy()

# Definimos las columnas a escalar
cols_to_scale = ['R', 'F', 'M']

# Creamos el escalador
scaler = MinMaxScaler()

# Ajustamos y transformamos las columnas seleccionadas y las reemplazamos en nuestro dataframe
df_scaled[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

# Ahora creamos el gráfico 3D
fig = px.scatter_3d(df_scaled, x='R', y='F', z='M', color='id_documento')

# Finalmente, mostramos el gráfico
fig.show()


# In[ ]:


# Contar la cantidad de id_documentos únicos en df2
unique_id_documentos = df2['id_documento'].nunique()

print(f"El número de id_documentos únicos en df2 es: {unique_id_documentos}")


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

def customerSegmentPlot(df2, id_field, title):
    # Agrupa por segmento y cuenta los valores únicos de id_documento
    segments_counts = df2.groupby("Segment")[id_field].nunique().sort_values(ascending=True)

    fig, ax = plt.subplots()

    bars = ax.barh(range(len(segments_counts)),
                  segments_counts,
                  color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False,
                   bottom=False,
                   labelbottom=False)
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)
    ax.set_title(title, fontsize=15)
    for i, bar in enumerate(bars):
            value = bar.get_width()
            if segments_counts.index[i] in ['champions', 'loyal customers']:
                bar.set_color('#7FC6DC')
            elif segments_counts.index[i] in ['at risk', 'about to sleep', 'need attention', "can't loose"]:
                bar.set_color('#004173')
            elif segments_counts.index[i] in ['potential loyalists', 'new customers', 'promising']:
                bar.set_color('lightgray')
            else:
                bar.set_color('darkgray')
            ax.text(value,
                    bar.get_y() + bar.get_height()/2,
                    '{:,} ({:}%)'.format(int(value),
                                       round(value*100/162200)),
                    va='center',
                    ha='left'
                   )

    plt.show()

# Llama a la función modificada con el campo de ID del documento
customerSegmentPlot(df2, 'id_documento', title="Customer category")


# In[ ]:


def assign_group(segment):
    if segment in ['champions', 'loyal customers']:
        return 'alto_valor'
    elif segment in ['potential loyalists', 'new customers', 'promising']:
        return 'potencial_de_crecimiento'
    elif segment in ['at risk', 'about to sleep', 'need attention', "can't loose"]:
        return 'riesgo_de_perdida'
    elif segment in ['hibernating']:
        return 'clientes_inactivos'
    else:
        return 'otros'

df2['Grupo'] = df2['Segment'].apply(assign_group)


# In[ ]:


df2.info()


# Creo las bases por grupo

# In[ ]:



# Contar la cantidad de 'id_documento' único en la columna 'Segment' para cada DataFrame
g1_counts = g1.groupby('Segment')['id_documento'].nunique()
g2_counts = g2.groupby('Segment')['id_documento'].nunique()
g3_counts = g3.groupby('Segment')['id_documento'].nunique()
g4_counts = g4.groupby('Segment')['id_documento'].nunique()

# Función para crear y mostrar el gráfico de barras
def plot_bar_chart(data, title):
    bar_color = '#7FC6DC'
    ax = data.plot(kind='bar', color=bar_color, figsize=(10, 6))
    ax.set_title(title)
    ax.set_ylabel('Cantidad')
    ax.set_xlabel('Categorías de Segment')
    plt.show()

# Crear y mostrar los gráficos de barras para cada DataFrame
plot_bar_chart(g1_counts, 'Distribución de Segment en g1')
plot_bar_chart(g2_counts, 'Distribución de Segment en g2')
plot_bar_chart(g3_counts, 'Distribución de Segment en g3')
plot_bar_chart(g4_counts, 'Distribución de Segment en g4')


# visualizar

# In[ ]:


def add_cant_beneficios_column(df):
    beneficios_count = df.groupby('id_documento')['NombreCuenta'].nunique()
    beneficios_count.name = 'cant_beneficios'
    df = df.join(beneficios_count, on='id_documento')
    return df

# Reemplazar g1, g2, g3, g4 con los dataframes correspondientes
dataframes = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

for group, df in dataframes.items():
    df_with_cant_beneficios = add_cant_beneficios_column(df)
    dataframes[group] = df_with_cant_beneficios
    print(f"Dataframe {group} con columna 'cant_beneficios':")
    print(dataframes[group][['id_documento', 'NombreCuenta', 'cant_beneficios']].head(20))
    print("\n")


# In[ ]:


import matplotlib.pyplot as plt

def add_cant_beneficios_column(df):
    beneficios_count = df.groupby('id_documento')['NombreCuenta'].nunique()
    beneficios_count.name = 'cant_beneficios'
    df = df.join(beneficios_count, on='id_documento')
    return df

def plot_cant_beneficios(df, title):
    cant_beneficios_counts = df['cant_beneficios'].value_counts().sort_index()
    ax = cant_beneficios_counts.plot(kind='bar', figsize=(10, 6), rot=0)
    ax.set_xlabel('Cantidad de Beneficios')
    ax.set_ylabel('Frecuencia')
    ax.set_title(title)

# Reemplazar g1, g2, g3, g4 con los dataframes correspondientes
dataframes = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

# Agregar la columna 'cant_beneficios' a cada dataframe
for group, df in dataframes.items():
    dataframes[group] = add_cant_beneficios_column(df)

# Graficar la columna 'cant_beneficios' en cada dataframe
for group, df in dataframes.items():
    plt.figure()
    plot_cant_beneficios(df, f"Dataframe {group}: Distribución de 'cant_beneficios'")
    plt.show()


# In[ ]:


g1.info()


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que los dataframes g1, g2, g3 y g4 ya están cargados

# Combinar todos los dataframes en una lista
dataframes = [g1, g2, g3, g4]

# Campos seleccionados
selected_fields = [
    "porc_descuento",
    "rubro", "edad", "rango_etario",
    "Recency", "Frequency", "Monetary Value"
]

# Crear subplots para cada campo
fig, axes = plt.subplots(nrows=len(selected_fields), ncols=len(dataframes), figsize=(20, 60))
plt.subplots_adjust(hspace=0.5)

# Establecer tonos azules para los gráficos
palette = sns.color_palette("Blues")

# Iterar sobre los campos seleccionados y dataframes para generar gráficos
for row, field in enumerate(selected_fields):
    for col, df in enumerate(dataframes):
        # Obtener valores únicos del campo id_documento
        unique_df = df.drop_duplicates(subset='id_documento', keep='first')
        
        if unique_df[field].dtype == np.dtype('O'):
            order = unique_df[field].value_counts().index
            sns.countplot(data=unique_df, x=field, ax=axes[row, col], palette=palette, order=order)
        elif unique_df[field].dtype == np.dtype('<M8[ns]'):
            sns.histplot(data=unique_df, x=field, ax=axes[row, col], color=palette[-1], bins=20)
        else:
            sns.histplot(data=unique_df, x=field, ax=axes[row, col], color=palette[-1], bins=20, kde=True)

        axes[row, col].set_title(f'G{col+1}: {field}')
        axes[row, col].tick_params(axis='x', labelrotation=45, labelsize=8)

plt.show()


# In[ ]:


unique_id_documentos = g1['id_documento'].nunique()
print(f'Hay {unique_id_documentos} id_documentos únicos en el dataframe g1.')


# In[ ]:


unique_id_documentos = g2['id_documento'].nunique()
print(f'Hay {unique_id_documentos} id_documentos únicos en el dataframe g1.')


# In[ ]:


unique_id_documentos = g3['id_documento'].nunique()
print(f'Hay {unique_id_documentos} id_documentos únicos en el dataframe g1.')


# In[ ]:


unique_id_documentos = g4['id_documento'].nunique()
print(f'Hay {unique_id_documentos} id_documentos únicos en el dataframe g1.')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que los dataframes g1, g2, g3 y g4 ya están cargados

# Combinar todos los dataframes en una lista
dataframes = [g1, g2, g3, g4]

# Campos seleccionados
selected_fields = [
    "rubro", "subrubro", "edad", "rango_etario",
    "Recency", "Frequency", "Monetary Value"
]

# Calcular los límites de los ejes para cada campo
axis_limits = {}
combined_df = pd.concat(dataframes).drop_duplicates(subset='id_documento', keep='first')
for field in selected_fields:
    if combined_df[field].dtype != np.dtype('O'):
        min_val = combined_df[field].min()
        max_val = combined_df[field].max()
        axis_limits[field] = (min_val, max_val)

# Crear subplots para cada campo
fig, axes = plt.subplots(nrows=len(selected_fields), ncols=len(dataframes), figsize=(30, 70))
plt.subplots_adjust(hspace=0.5)

# Establecer tonos azules para los gráficos
palette = sns.color_palette("Blues")

# Iterar sobre los campos seleccionados y dataframes para generar gráficos
for row, field in enumerate(selected_fields):
    for col, df in enumerate(dataframes):
        # Obtener valores únicos del campo id_documento
        unique_df = df.drop_duplicates(subset='id_documento', keep='first')
        
        if unique_df[field].dtype == np.dtype('O'):
            if field == 'subrubro':
                order = unique_df[field].value_counts().iloc[:10].index
            else:
                order = unique_df[field].value_counts().index
            sns.countplot(data=unique_df, x=field, ax=axes[row, col], palette=palette, order=order)
        elif unique_df[field].dtype == np.dtype('<M8[ns]'):
            sns.histplot(data=unique_df, x=field, ax=axes[row, col], color=palette[-1], bins=20)
        else:
            sns.histplot(data=unique_df, x=field, ax=axes[row, col], color=palette[-1], bins=20, kde=True)
            axes[row, col].set_xlim(axis_limits[field])

        axes[row, col].set_title(f'G{col+1}: {field}')
        axes[row, col].tick_params(axis='x', labelrotation=45, labelsize=8)

plt.show()


# In[ ]:


import pandas as pd

def top_10_zones(df):
    top_10_suscriptor = df['zona_partido_suscriptor'].value_counts().nlargest(10)
    top_10_suscriptor_percentage = (top_10_suscriptor / df['zona_partido_suscriptor'].count()) * 100

    top_10_comercio = df['zona_partido_comercio'].value_counts().nlargest(10)
    top_10_comercio_percentage = (top_10_comercio / df['zona_partido_comercio'].count()) * 100

    result_suscriptor = pd.concat([top_10_suscriptor, top_10_suscriptor_percentage], axis=1)
    result_suscriptor.columns = ['Count', 'Percentage']
    
    result_comercio = pd.concat([top_10_comercio, top_10_comercio_percentage], axis=1)
    result_comercio.columns = ['Count', 'Percentage']

    return result_suscriptor, result_comercio

# Reemplazar g1, g2, g3, g4 con los dataframes correspondientes
dataframes = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

for group, df in dataframes.items():
    suscriptor, comercio = top_10_zones(df)
    print(f"Top 10 zona_partido_suscriptor para {group}:")
    print(suscriptor)
    print(f"\nTop 10 zona_partido_comercio para {group}:")
    print(comercio)
    print("\n")


# In[ ]:


import pandas as pd

def top_10_zones_without_caba(df):
    df_no_caba = df[(df['zona_partido_suscriptor'] != 'CABA') & (df['zona_partido_comercio'] != 'CABA')]

    top_10_suscriptor = df_no_caba['zona_partido_suscriptor'].value_counts().nlargest(10)
    top_10_suscriptor_percentage = (top_10_suscriptor / df_no_caba['zona_partido_suscriptor'].count()) * 100

    top_10_comercio = df_no_caba['zona_partido_comercio'].value_counts().nlargest(10)
    top_10_comercio_percentage = (top_10_comercio / df_no_caba['zona_partido_comercio'].count()) * 100

    result_suscriptor = pd.concat([top_10_suscriptor, top_10_suscriptor_percentage], axis=1)
    result_suscriptor.columns = ['Count', 'Percentage']
    
    result_comercio = pd.concat([top_10_comercio, top_10_comercio_percentage], axis=1)
    result_comercio.columns = ['Count', 'Percentage']

    return result_suscriptor, result_comercio

# Reemplazar g1, g2, g3, g4 con los dataframes correspondientes
dataframes = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

for group, df in dataframes.items():
    suscriptor, comercio = top_10_zones_without_caba(df)
    print(f"Top 10 zona_partido_suscriptor sin CABA para {group}:")
    print(suscriptor)
    print(f"\nTop 10 zona_partido_comercio sin CABA para {group}:")
    print(comercio)
    print("\n")


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Asumiendo que los dataframes g1, g2, g3 y g4 ya están cargados

# Combinar todos los dataframes en una lista
dataframes = [g1, g2, g3, g4]

# Campos seleccionados
selected_fields = [
    "Recency", "Frequency", "Monetary Value"
]

# Crear subplots para cada campo
fig, axes = plt.subplots(nrows=len(selected_fields), ncols=len(dataframes), figsize=(20, 15))
plt.subplots_adjust(hspace=0.5)

# Establecer tonos azules para los gráficos
palette = sns.color_palette("Blues")

# Iterar sobre los campos seleccionados y dataframes para generar gráficos
for row, field in enumerate(selected_fields):
    for col, df in enumerate(dataframes):
        # Obtener valores únicos del campo id_documento
        unique_df = df.drop_duplicates(subset='id_documento', keep='first')
        
        # Crear histograma para cada campo
        sns.histplot(data=unique_df, x=field, ax=axes[row, col], color=palette[-1], bins=20, kde=True)

        axes[row, col].set_title(f'G{col+1}: {field}')
        axes[row, col].tick_params(axis='x', labelrotation=45, labelsize=8)

plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")
sns.set_palette("Blues_r")


# In[ ]:


def plot_dataframe(df, title):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='cat_credencial', hue='condicion')
    plt.title('Categoría de Credencial y Condición')
    
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='pago_socio', kde=True)
    plt.title('Pago Socio')
    
    plt.subplot(2, 2, 3)
    sns.histplot(data=df, x='porc_descuento', kde=True)
    plt.title('Porcentaje de Descuento')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='rubro', y='edad')
    plt.xticks(rotation=90)
    plt.title('Edad por Rubro')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.show()

    sns.countplot(data=df, x='rango_etario')
    plt.xticks(rotation=90)
    plt.title('Rango Etario')
    plt.show()

    sns.scatterplot(data=df, x='Recency', y='Frequency', size='Monetary Value', alpha=0.5)
    plt.title('Recency vs Frequency (Tamaño por Monetary Value)')
    plt.show()


# In[ ]:


dataframes = {'g1': g1, 'g2': g2, 'g3': g3, 'g4': g4}

for name, df in dataframes.items():
    unique_df = df.drop_duplicates(subset='id_documento')
    plot_dataframe(unique_df, f'Visualización del dataframe {name}')

