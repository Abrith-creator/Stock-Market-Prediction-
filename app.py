import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

page_bg_img = """
<style>
[data-testid="stHeader"]{
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7towp_9J68WCYtjfFqX4Dq2JnvdGH2-Nf5ddMELjIisv1won3ZnpOq0-qt7tJjGHkwn4&usqp=CAU");

}
[data-testid="stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-vector/abstract-background-with-flowing-dots_1048-11254.jpg?size=626&ext=jpg&ga=GA1.1.2116175301.1717632000&semt=ais_user");
background-size:cover;

}
[data-testid="stSidebarUserContent"]{
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7towp_9J68WCYtjfFqX4Dq2JnvdGH2-Nf5ddMELjIisv1won3ZnpOq0-qt7tJjGHkwn4&usqp=CAU");

}
[data-testid="stSidebarContent"]{
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7towp_9J68WCYtjfFqX4Dq2JnvdGH2-Nf5ddMELjIisv1won3ZnpOq0-qt7tJjGHkwn4&usqp=CAU");

}
[data-testid="stSidebarHeader"]{
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT7towp_9J68WCYtjfFqX4Dq2JnvdGH2-Nf5ddMELjIisv1won3ZnpOq0-qt7tJjGHkwn4&usqp=CAU");

}
<style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Set app title and description
st.title("Stock Price Predictor App")
st.markdown("""
    This app predicts stock prices using a pre-trained machine learning model.
    Enter the stock symbol and select the date range in the sidebar, then explore the predictions.
""")

# Sidebar for user input
st.sidebar.title("Stock Symbol Input")
stock = st.sidebar.text_input("Enter the Stock Symbol", "GOOG")

# Date input
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp.now() - pd.DateOffset(years=20))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.now())

# Fetch historical stock data
try:
    google_data = yf.download(stock, start=start_date, end=end_date)
    if google_data.empty:
        st.error("Failed to retrieve data for the given stock symbol. Please check the symbol and try again.")
    else:
        # Calculate moving averages
        google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
        google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
        google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()

        # Load pre-trained model
        model = load_model("Latest_stock_price_model.keras")

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Stock Data", 
            "Price vs MA 100", 
            "MA 100 vs MA 200", 
            "MA 200 vs MA 250",
            "Original vs Predicted Prices"
        ])

        with tab1:
            st.subheader("Stock Data")
            st.write("Displaying the last 500 days of data.")
            st.dataframe(google_data.tail(500))  # Display the last 500 rows of the data

        with tab2:
            st.subheader('Price vs MA 100')
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(google_data['Close'], label='Close Price')
            ax.plot(google_data['MA_for_100_days'], label='MA for 100 days')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

        with tab3:
            st.subheader('MA 100 vs MA 200')
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(google_data['MA_for_100_days'], label='MA for 100 days')
            ax.plot(google_data['MA_for_200_days'], label='MA for 200 days')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

        with tab4:
            st.subheader('MA 200 vs MA 250')
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(google_data['MA_for_200_days'], label='MA for 200 days')
            ax.plot(google_data['MA_for_250_days'], label='MA for 250 days')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

        with tab5:
            # Data preprocessing
            splitting_len = int(len(google_data) * 0.7)
            x_test = pd.DataFrame(google_data['Close'][splitting_len:])
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(x_test[['Close']])

            # Generate predictions
            x_data = []
            y_data = []

            for i in range(100, len(scaled_data)):
                x_data.append(scaled_data[i-100:i])
                y_data.append(scaled_data[i])

            x_data, y_data = np.array(x_data), np.array(y_data)
            predictions = model.predict(x_data)
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)

            # Display original vs predicted values
            plotting_data = pd.DataFrame({
                'Original Close Price': inv_y_test.reshape(-1),
                'Predicted Close Price': inv_pre.reshape(-1)
            }, index=google_data.index[splitting_len+100:])
            st.subheader("Original vs Predicted Close Prices")
            st.write(plotting_data.tail(10))  # Display the last 10 rows of the data

            # Plot original vs predicted close price
            st.subheader('Original Close Price vs Predicted Close Price')
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(google_data['Close'][:splitting_len+100], label='Original Close Price')
            ax.plot(plotting_data['Predicted Close Price'], label='Predicted Close Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
