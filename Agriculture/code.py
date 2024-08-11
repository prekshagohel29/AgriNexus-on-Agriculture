import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import csv  # Import the csv module

# Path to the CSV files
PREDICT_DATA_FILE = "predict_data.csv"
CALCULATOR_DATA_FILE = "calculator_data.csv"
CSV_FILE = "user_data.csv"

# Load the calculator data
calculator_df = pd.read_csv(CALCULATOR_DATA_FILE)

# Set up the page layout and title
st.set_page_config(layout="wide", page_title="AgriNexus")

# Sidebar menu
st.sidebar.title("Navigation")
menu_selection = st.sidebar.selectbox(
    "Main Menu",
    ["Home", "Price Predictions", "Cost & Profit Calculator"],
    index=0
)

if menu_selection == "Home":
    st.markdown("<h1 class='title'>🌾 AgriNexus: Bridging Agriculture with Technology 🌾</h1>", unsafe_allow_html=True)

    # Display the introduction details in a styled box
    st.markdown("""
    <div class='intro-box'>
        <h2>Welcome to AgriNexus!</h2>
        <p>
            <strong>🌱 Challenge:</strong> Farmers face daunting challenges such as unpredictable crop diseases and fluctuating market prices, making it difficult to sustain their livelihoods.
        </p>
        <p>
            <strong>🌟 Our Solution:</strong> AgriNexus empowers farmers with cutting-edge tools that predict crop diseases and provide real-time market prices, fostering informed decision-making and boosting agricultural productivity.
        </p>
        <p>
            <strong>🎯 Our Goal:</strong> We aim to revolutionize the agricultural industry by delivering actionable insights, enabling farmers to thrive in a technology-driven world.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create the login form with enhanced styling
    with st.form(key="login_form"):
        st.markdown("<h2 style='text-align: center;'>🔐 Login to Your Account</h2>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username")
        email = st.text_input("Email Address", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", placeholder="Enter your password")

        submit_button = st.form_submit_button(label="Login")

    # Handle form submission
    if submit_button:
        if username and email and password:
            # Save to CSV
            file_exists = os.path.isfile(CSV_FILE)

            try:
                with open(CSV_FILE, 'a', newline='') as csvfile:
                    fieldnames = ['Username', 'Email Address', 'Password']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    # Write header if file is empty
                    if not file_exists:
                        writer.writeheader()

                    writer.writerow({'Username': username, 'Email Address': email, 'Password': password})

                # Confirm successful saving
                st.success(f"🎉 Welcome, {username}! Your data has been saved successfully.")

            except Exception as e:
                st.error(f"⚠️ An error occurred while saving data: {e}")
        else:
            st.warning("⚠️ Please fill out all fields.")


elif menu_selection == "Cost & Profit Calculator":
    st.title("💰 Cost and Profit Margin Calculator")

    # Input for land area in hectares
    land_area = st.number_input("Enter Land in Hectares (1 Hectare = 4,047 square meters)", min_value=0.0, value=1.0)

    # Load the CSV file with historical data for commodity prediction
    try:
        df = pd.read_csv(PREDICT_DATA_FILE)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.set_index('Commodity')

        # Display the commodity selection dropdown
        commodity = st.selectbox("Select Commodity for Profit Calculation", df.index)

        if commodity:
            # Fetch seed and fertilizer information from the calculator data
            calculator_row = calculator_df[calculator_df['Commodity'] == commodity]

            if not calculator_row.empty:
                seed_per_hectare = calculator_row['Seed per Hectare (kg)'].values[0]
                seed_cost_per_kg = calculator_row['Seed Cost per kg (INR)'].values[0]
                fertilizer_per_hectare = calculator_row['Fertilizer per Hectare (kg)'].values[0]
                fertilizer_cost_per_kg = calculator_row['Fertilizer Cost per kg (INR)'].values[0]

                # Calculate total costs based on land area
                seed_cost = seed_per_hectare * seed_cost_per_kg * land_area
                fertilizer_cost = fertilizer_per_hectare * fertilizer_cost_per_kg * land_area

                st.write(f"**Seed Cost:** ₹ {seed_cost}")
                st.write(f"**Fertilizer Cost:** ₹ {fertilizer_cost}")

                # Input for other costs (labor, other expenses)
                labor_cost = st.number_input("Labor Cost", min_value=0.0, value=0.0)
                other_costs = st.number_input("Other Costs", min_value=0.0, value=0.0)

                # Calculate total cost
                total_cost = seed_cost + fertilizer_cost + labor_cost + other_costs
                st.write(f"**Total Cost:** ₹ {total_cost}")

                # Predict selling price based on commodity prediction
                future_years = ['2025-26', '2026-27']  # Future years to predict
                X = np.arange(len(df.columns)).reshape(-1, 1)
                y = df.loc[commodity].values

                # Scale and train model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = LinearRegression()
                model.fit(X_scaled, y)

                # Predict prices for future years
                years_to_predict = np.array([[len(df.columns)], [len(df.columns) + 1]])
                years_to_predict_scaled = scaler.transform(years_to_predict)
                predicted_prices = model.predict(years_to_predict_scaled)

                selling_price = st.number_input("Expected Selling Price per Unit", min_value=0.0, value=predicted_prices[0])

                # Calculate profit margin
                units_sold = st.number_input("Units Expected to Sell", min_value=0.0, value=1.0)
                profit_margin = (selling_price * units_sold) - total_cost

                st.write(f"**Expected Profit Margin:** ₹ {profit_margin}")
            else:
                st.warning("⚠️ No data available for the selected commodity.")
        else:
            st.warning("⚠️ Please select a commodity.")

    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the predictions: {e}")

elif menu_selection == "Price Predictions":
    st.title("📈 Commodity Price Predictions")

    # Load the CSV file with historical data
    try:
        df = pd.read_csv(PREDICT_DATA_FILE)

        # Check for duplicate columns and remove them
        df = df.loc[:, ~df.columns.duplicated()]

        # Extract years from the columns and set 'Commodity' as index
        years = df.columns[1:]  # Extract years from columns
        df = df.set_index('Commodity')

        # Display the commodity selection dropdown
        commodity = st.selectbox("Select Commodity", df.index)

        # Prepare for scaling and model training
        future_years = ['2025-26', '2026-27']  # Future years to predict

        if commodity:
            # Prepare data
            X = np.arange(len(df.columns)).reshape(-1, 1)  # years as features
            y = df.loc[commodity].values  # prices as target

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train the linear regression model
            model = LinearRegression()
            model.fit(X_scaled, y)

            # Predict for 2025-26 and 2026-27
            years_to_predict = np.array([[len(df.columns)], [len(df.columns) + 1]])  # 2025-26 and 2026-27
            years_to_predict_scaled = scaler.transform(years_to_predict)
            predicted_prices = model.predict(years_to_predict_scaled)

            # Prepare predictions DataFrame
            predictions = pd.DataFrame(
                {
                    'Year': future_years,
                    'Predicted Price': predicted_prices
                }
            )

            # Display the DataFrame
            st.write(f"Predictions for {commodity}:")
            st.dataframe(predictions)

            # Option to download the predictions as a CSV file
            st.download_button(
                label="Download Predictions",
                data=predictions.to_csv(index=False),
                file_name=f"{commodity}_predictions_2025_2026.csv"
            )
        else:
            st.warning("⚠️ Please select a commodity.")
    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the predictions: {e}")
