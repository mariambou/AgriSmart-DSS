# AgriSmart-DSS

AgriSmart-DSS is a data-driven decision support system designed to help farmers and agricultural advisors make informed decisions about crop selection. By combining soil characteristics, climate conditions, and market economics, the system predicts crop yield and market price, then recommends the most profitable crop for a given farm.

*This project was developed as a final year thesis in Data Engineering.


## Intelligent Agricultural Decision Support System

~ Overview :
Facing climate challenges, water scarcity, and market volatility, farmers need intelligent tools to optimize their agricultural decisions. AgriSmart-DSS addresses this need by:
- Analyzing relationships between soil, climate, and agricultural production
- Building machine learning models to predict crop yield and market price
- Developing an interactive web application for real-time decision support
- Providing crop recommendations based on profitability

~ Objectives :
- Predict crop yield using agronomic and climate variables
- Estimate market price using economic indicators
- Calculate profitability (yield × price) for each crop
- Recommend the most profitable crop (Rice, Wheat, Corn, Soybean)
- Deliver insights through an accessible, interactive dashboard

~ Key Features :
**Yield Prediction** : Estimates crop yield (tons) using Random Forest models 
**Price Prediction** : Estimates market price ($/ton) using Gradient Boosting models 
**Profitability Calculation** : Computes expected profit = yield × price 
**Crop Recommendation** : Suggests the most profitable crop (Rice, Wheat, Corn, Soybean) 
**Interactive Visualizations** : Bar charts, radar charts, scatter plots using Plotly 
**Agronomic Advice** : Provides best practices for the recommended crop 


~ Models Used :
#### Yield Prediction (Random Forest Regressor)
- Trained separately for each crop (Rice, Wheat, Corn, Soybean)
- Captures non-linear relationships between agronomic variables

#### Price Prediction (Gradient Boosting Regressor)
- Trained separately for each crop
- Integrates economic indicators and market dynamics

#### Performance Metrics
For Yield > Random Forest : R²= ~0.00 | RMSE= ~2.62 
For Price > Gradient Boosting : R²= ~0.00 | RMSE= ~117.33 

> **Note**: Low R² values indicate weak linear correlations in the dataset. Models provide approximate estimates suitable for decision support, not precise predictions.

~ How to Use the Application :
1.Enter farm parameters in the sidebar:
- Soil pH and moisture
- Temperature and rainfall
- Fertilizer and pesticide usage
- Market price and demand index
2. Click "Analyze" to generate estimates
3. View results:
- Estimated yield for each crop
- Estimated market price
- Calculated profitability
4. Get recommendation:
- The system highlights the most profitable crop
- Displays agronomic advice for that crop
5. Explore visualizations:
- Compare crops using bar charts
- Analyze trade-offs with radar charts

~ Limitations :
- Low model performance (R² near 0) due to weak correlations in the dataset
- Cross-sectional data : no temporal dynamics (seasonal or yearly trends)
- Missing variables : seed quality, irrigation practices, pest pressure
- Static dataset : does not update with real-time weather or market data
Important: Predictions are indicative only. Always consult a local agricultural advisor before making major decisions.

~ Future Improvements :
- Integrate real-time weather APIs and live market data
- Add more features (soil composition, historical yields, disease risk)
- Test deep learning models
- Add explainable AI to interpret predictions
- Deploy as a cloud-based web service


Author
Mariam Bourchouk – Data Engineering student
Institution: Department of Computer Engineering – Data Engineering Program
Academic Year: 2025–2026


### Repository Files
- README.md	: Project overview, objectives, features, and instructions
- data/raw/ : Original datasets (soil/climate and market data)
- data/processed/	: Cleaned and merged final dataset for modeling
- notebooks/ : Jupyter notebooks for EDA, feature engineering, and modeling
- app/ : Streamlit application code
- images/ : Screenshots of the application and visualizations
- docs/	: project report included)
- requirements.txt : List of Python dependencies required to run the project
- .gitignore : Specifies intentionally untracked files to ignore
