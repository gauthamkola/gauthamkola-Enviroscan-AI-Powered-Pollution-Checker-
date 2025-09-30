{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\csgray\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh13040\viewkind0
\deftab720
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # Pollution Source Prediction System\
A machine learning-based web application built with Streamlit to monitor air pollution, predict pollution sources (e.g., Vehicular, Industrial, Agricultural, Burning, Natural), and visualize data on maps and charts. Data is fetched from OpenAQ API, enriched with weather data and OSMnx features, and models are trained using scikit-learn and XGBoost.\
## Features\
- **Data Collection**: Fetches real air quality data (PM2.5, PM10, NO2, SO2, O3, CO) from OpenAQ for major Indian cities.\
- **Feature Engineering**: Adds weather data, seasonal indicators, OSMnx-based industrial and road features.\
- **ML Models**: Trains Random Forest, XGBoost, and Decision Tree classifiers to predict pollution sources. Saves the best model.\
- **Streamlit Dashboard**: Interactive map, analytics, alerts, and a prediction interface.\
- **Deployment**: Uses ngrok for public URL exposure (for development/testing).\
## Tech Stack\
- Python 3.x\
- Libraries: Streamlit, Folium, Plotly, scikit-learn, XGBoost, OSMnx, Geopandas, Requests, Joblib, Pyngrok\
- APIs: OpenAQ (for air quality), OpenWeather (placeholder, uses static monthly data)\
- Data Sources: OpenAQ, OSMnx for geospatial features\
## Installation\
1. Clone the repo:\
# Pollution Source Prediction System\
A machine learning-based web application built with Streamlit to monitor air pollution, predict pollution sources (e.g., Vehicular, Industrial, Agricultural, Burning, Natural), and visualize data on maps and charts. Data is fetched from OpenAQ API, enriched with weather data and OSMnx features, and models are trained using scikit-learn and XGBoost.\
## Features\
- **Data Collection**: Fetches real air quality data (PM2.5, PM10, NO2, SO2, O3, CO) from OpenAQ for major Indian cities.\
- **Feature Engineering**: Adds weather data, seasonal indicators, OSMnx-based industrial and road features.\
- **ML Models**: Trains Random Forest, XGBoost, and Decision Tree classifiers to predict pollution sources. Saves the best model.\
- **Streamlit Dashboard**: Interactive map, analytics, alerts, and a prediction interface.\
- **Deployment**: Uses ngrok for public URL exposure (for development/testing).\
## Tech Stack\
- Python 3.x\
- Libraries: Streamlit, Folium, Plotly, scikit-learn, XGBoost, OSMnx, Geopandas, Requests, Joblib, Pyngrok\
- APIs: OpenAQ (for air quality), OpenWeather (placeholder, uses static monthly data)\
- Data Sources: OpenAQ, OSMnx for geospatial features\
## Installation\
1. Clone the repo:\
# Pollution Source Prediction System\
A machine learning-based web application built with Streamlit to monitor air pollution, predict pollution sources (e.g., Vehicular, Industrial, Agricultural, Burning, Natural), and visualize data on maps and charts. Data is fetched from OpenAQ API, enriched with weather data and OSMnx features, and models are trained using scikit-learn and XGBoost.\
## Features\
- **Data Collection**: Fetches real air quality data (PM2.5, PM10, NO2, SO2, O3, CO) from OpenAQ for major Indian cities.\
- **Feature Engineering**: Adds weather data, seasonal indicators, OSMnx-based industrial and road features.\
- **ML Models**: Trains Random Forest, XGBoost, and Decision Tree classifiers to predict pollution sources. Saves the best model.\
- **Streamlit Dashboard**: Interactive map, analytics, alerts, and a prediction interface.\
- **Deployment**: Uses ngrok for public URL exposure (for development/testing).\
## Tech Stack\
- Python 3.x\
- Libraries: Streamlit, Folium, Plotly, scikit-learn, XGBoost, OSMnx, Geopandas, Requests, Joblib, Pyngrok\
- APIs: OpenAQ (for air quality), OpenWeather (placeholder, uses static monthly data)\
- Data Sources: OpenAQ, OSMnx for geospatial features\
## Installation\
1. Clone the repo:\
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f1\fs22 \cf3 \kerning1\expnd0\expndtw0 \CocoaLigature0 \outl0\strokewidth0 git clone https://github.com/gauthamkola/Enviroscan-AI-Powered-Pollution-Checker-.git\
cd Enviroscan-AI-Powered-Pollution-Checker-\
\
2. Install dependencies:\
\pard\pardeftab720\sa240\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\CocoaLigature1 \outl0\strokewidth0 \strokec2 pip install -r requirements.txt\
3. Set up API keys (in `generate_data_and_train.py`):\
- OpenAQ API Key: Replace the placeholder.\
- Ngrok Auth Token: Replace in the script if using ngrok.\
## Usage\
1. Generate data and train models:\
python generate_data_and_train.py\
- This creates `data/labeled_data.csv` and saves models in `models/`.\
2. Run the Streamlit app:\
streamlit run app.py\
- For public access, the script includes ngrok setup. Run it in a Colab or local environment with ngrok.\
\
3. Access the dashboard:\
- Local: http://localhost:8501\
- Public (via ngrok): Check the console output for the URL.\
\
## Project Structure\
- `app.py`: Streamlit application code.\
- `generate_data_and_train.py`: Script to fetch data, engineer features, train models.\
- `data/`: Stores generated CSV data (e.g., `labeled_data.csv`).\
- `models/`: Stores trained ML models (e.g., `model.joblib`).\
- `requirements.txt`: Dependencies.\
- `.gitignore`: Ignores unnecessary files.\
\
## Data Sources and Notes\
- **OpenAQ API**: Used for real-time/historical air quality data. Limit requests to avoid rate limits.\
- **Weather Data**: Static monthly averages for cities (can be extended with real API calls).\
- **OSMnx**: Fetches industrial areas and road lengths within 2km radius.\
- **Models**: Predicts sources based on pollutants, weather, and geospatial features.\
- **Limitations**: Data is sampled to 500 rows for efficiency. Extend for production.\
\
## Contributing\
Feel free to fork and submit pull requests. Issues welcome!\
\
## License\
MIT License }