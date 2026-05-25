#################################################################
#						 		#
#  PROYECTO XAI 2026				 		#
#  TITULO: XAI PARA SERIES DE TIEMPO DE TSUNAMI  		#
#  INTEGRANTES:					 		#
#    - MATIAS SIFON ANDALAFT			 		#
#    - TOMÁS MERCADO JEFF   				 	#
#    - IGNACIO MUÑOZ ULLOA					#
#  PROFESORA: RAQUEL PEZOA RIVERA		 		#
#  ULTIMA ACTUALIZACIÓN: MAYO 2026	 	 		#
#						 		#
#################################################################

Proyecto desarrollado en el marco de la asignatura "Introducción a la Inteligencia Artificial Explicable" año 2026, para implementar métodos de XAI a series de tiempo de tsunami. Para información conceptual, revisar el informe en formato pdf. Para ejecutar:

Proyecto desarrollado con Python 3.12.13, usando un entorno virtual de Conda.

La estructura de los archivos de Jupyter Notebook (*.ipynb) es serieda de la siguiente forma:
- 0X_nombre.ipynb : Archivos introductorios y de orden de datos y modelos
- 1X_nombre.ipynb : Archivos para implementar el método CONFETTI (Cetina et al. 2026)
- 2X_nombre.ipynb : Archivos para implementar el método ts-MULE (Schlegel et al. 2021)
- 3X_nombre.ipynb : Archivos para análisis conjunto usando ambos métodos

El repositorio tiene la siguiente estructura:
|--Repo
	|-- DATA/: Set de datos
	|-- FIGS/: Figuras de desempeño del modelo
	|-- METHODS/
		|-- confetti/: Repositorio CONFETTI
		|-- ts-mule/: Repositorio ts-MULE
	|-- MODELS/: Archivos de modelos *.keras, *.h5 y pesos para CAM (usado por CONFETTI)
	|-- REFS/: Archivos de bibliografía
	|-- RESULTADOS_COMPARACION/: Archivos y gráficos obtenidos de J. Notebooks 3X_nombre.ipynb
	|-- RESUTLADOS CONFETTI/: Archivos y gráficos obtenidos de J. Notebooks 1X_nombre.ipynb
	|-- RESULTADOS_TSMULE/: Archivos y gráficos obtenidos de J. Notebooks 2X_nombre.ipynb
	|-- xai_utils/: códigos de python para analizar y gráficar aplicación de métodos
	|-- requirements.txt : Requerimientos de Python
	|-- README.txt : lo estás leyendo :)


¡¡ Es necesario instalar CONFETTI !!

### PyPI Installation (bash)
pip install confetti-ts

### Development Installation (bash)
git clone https://github.com/serval-uni-lu/confetti.git
cd confetti


1. Obtención de los archivos y paquetes:
Copiar el repositorio de GitHub, con el comando:
git clone XXXX (nuestro repositorio)


2. Configuración del entorno virtual:
Crear el entorno con el comando: 

> conda create --name <env> --file requirements.txt

3. Ahora se puede ejecutar cualquiera de los otros notebook
