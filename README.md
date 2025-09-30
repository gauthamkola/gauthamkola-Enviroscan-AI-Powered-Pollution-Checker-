# Pollution Source Prediction System

A machine learning-based web application built with Streamlit to monitor air pollution, predict pollution sources (e.g., Vehicular, Industrial, Agricultural, Burning, Natural), and visualize data on maps and charts. Data is fetched from OpenAQ API, enriched with weather data and OSMnx features, and models are trained using scikit-learn and XGBoost.

## Features
- *Data Collection*: Fetches real air quality data (PM2.5, PM10, NO2, SO2, O3, CO) from OpenAQ for major Indian cities.
- *Feature Engineering*: Adds weather data, seasonal indicators, OSMnx-based industrial and road features.
- *ML Models*: Trains Random Forest, XGBoost, and Decision Tree classifiers to predict pollution sources. Saves the best model.
- *Streamlit Dashboard*: Interactive map, analytics, alerts, and a prediction interface.
- *Deployment*: Uses ngrok for public URL exposure (for development/testing).

## Tech Stack
- Python 3.x
- Libraries: Streamlit, Folium, Plotly, scikit-learn, XGBoost, OSMnx, Geopandas, Requests, Joblib, Pyngrok
- APIs: OpenAQ (for air quality), OpenWeather (placeholder, uses static monthly data)
- Data Sources: OpenAQ, OSMnx for geospatial features

## Installation
1. Clone the repo:
git clone https://github.com/gauthamkola/gauthamkola-Enviroscan-AI-Powered-Pollution-Checker-.git
- cd gauthamkola-Enviroscan-AI-Powered-Pollution-Checker-
2. Install dependencies:
pip install -r requirements.txt
3. Set up API keys (in generate_data_and_train.py):
- OpenAQ API Key: Replace the placeholder.
- Ngrok Auth Token: Replace in the script if using ngrok.
- OpenWeatherApi Key: Replace in the placeholder.

## Usage
1. Generate data and train models:
python generate_data_and_train.py
- This creates data/labeled_data.csv and saves models in models/.

2. Run the Streamlit app:
- For public access, the script includes ngrok setup. Run it in a Colab or local environment with ngrok.

3. Access the dashboard:
- Local: http://localhost:8501
- Public (via ngrok): Check the console output for the URL.

## Project Structure
- app.py: Streamlit application code.
- generate_data_and_train.py: Script to fetch data, engineer features, train models.
- data/: Stores generated CSV data (e.g., labeled_data.csv).
- models/: Stores trained ML models (e.g., model.joblib).
- requirements.txt: Dependencies.
- .gitignore: Ignores unnecessary files.
## Dashboard Pictures
<img width="1470" height="791" alt="Screenshot 2025-10-01 at 12 58 09 AM" src="https://github.com/user-attachments/assets/8dbd59f2-5a78-4637-845c-bb4ea8d28aad" />
<img width="1470" height="752" alt="Screenshot 2025-10-01 at 12 58 34 AM" src="https://github.com/user-attachments/assets/76609915-205d-4979-97f8-d3c92d20592f" />
<img width="1453" height="762" alt="Screenshot 2025-10-01 at 12 58 43 AM" src="https://github.com/user-attachments/assets/c84a6ca8-a5fa-4c07-80b0-a2168fb64733" />
<img width="1470" height="702" alt="Screenshot 2025-10-01 at 12 58 48 AM" src="https://github.com/user-attachments/assets/6f0bbe07-f984-4627-ab7a-9afc12f155ba" />

## Data Sources and Notes
- *OpenAQ API*: Used for real-time/historical air quality data. Limit requests to avoid rate limits.
- *Weather Data*: Static monthly averages for cities (can be extended with real API calls).
- *OSMnx*: Fetches industrial areas and road lengths within 2km radius.
- *Models*: Predicts sources based on pollutants, weather, and geospatial features.
- *Limitations*: Data is sampled to 500 rows for efficiency. Extend for production.
  
## Project Done By:
- Gautham K
- Kushal P Hiremath

## Contributing
Feel free to fork and submit pull requests. Issues welcome!

## License
MIT License.
