import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go

# Load the predictions
predictions_df = pd.read_csv('data/predicted_prices.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Crypto Price Prediction Comparison"),
    
    # Graph for predicted prices
    dcc.Graph(id='price-predictions-graph'),
    
    # Graph for MSE comparison
    html.Div([
        html.H2("Mean Squared Error (MSE) Comparison"),
        dcc.Graph(id='mse-graph')
    ])
])

# Callback to update the price predictions graph
@app.callback(
    Output('price-predictions-graph', 'figure'),
    Input('price-predictions-graph', 'id')
)
def update_price_predictions_graph(_):
    fig = go.Figure()
    
    # Plot predicted prices for each model
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['LSTM_Predicted_Price'],
                             mode='lines+markers', name='LSTM'))
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['GRU_Predicted_Price'],
                             mode='lines+markers', name='GRU'))
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['CNN_Predicted_Price'],
                             mode='lines+markers', name='CNN'))
    
    fig.update_layout(title="Predicted Crypto Prices for Next Week",
                      xaxis_title="Date", yaxis_title="Price")
    return fig

# Callback to update the MSE graph
@app.callback(
    Output('mse-graph', 'figure'),
    Input('mse-graph', 'id')
)
def update_mse_graph(_):
    # Dummy MSE values; replace with actual values calculated from model training
    mse_data = {
        'Model': ['LSTM', 'GRU', 'CNN'],
        'MSE': [
            predictions_df['LSTM_Predicted_Price'].mean(),  # Replace with actual MSE values
            predictions_df['GRU_Predicted_Price'].mean(),   # Replace with actual MSE values
            predictions_df['CNN_Predicted_Price'].mean()    # Replace with actual MSE values
        ]
    }
    mse_df = pd.DataFrame(mse_data)
    
    fig = go.Figure([go.Bar(x=mse_df['Model'], y=mse_df['MSE'])])
    fig.update_layout(title="Mean Squared Error Comparison",
                      xaxis_title="Model", yaxis_title="MSE")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
