import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Đọc dữ liệu lịch sử
df_hist = pd.read_csv('techco_financials_with_variables.csv')
features = ['Labor Cost', 'Production Cost', 'Other Costs', 'CPI (%)', 'Material Cost (%)']
target_revenue = 'Revenue (M USD)'
target_profit = 'Profit'

# Huấn luyện mô hình dự báo doanh thu
X = df_hist[features]
y_revenue = df_hist[target_revenue]
model_revenue = LinearRegression()
model_revenue.fit(X, y_revenue)

# Huấn luyện mô hình dự báo lợi nhuận
y_profit = df_hist[target_profit]
model_profit = LinearRegression()
model_profit.fit(X, y_profit)

st.title("Dự báo doanh thu theo CPI và Giá nguyên liệu")

# Nhập các biến đầu vào
labor = st.number_input("Labor Cost", value=0.53)
prod = st.number_input("Production Cost", value=0.86)
other = st.number_input("Other Costs", value=0.24)
cpi = st.number_input("CPI (%)", value=106.0)
material = st.number_input("Material Cost (%)", value=109.0)

if st.button("Dự báo doanh thu và lợi nhuận"):
    X_new = pd.DataFrame([[labor, prod, other, cpi, material]], columns=features)
    revenue_pred = model_revenue.predict(X_new)[0]
    profit_pred = model_profit.predict(X_new)[0]
    st.success(f"Doanh thu dự báo: {revenue_pred:.2f} triệu USD")
    st.success(f"Lợi nhuận dự báo: {profit_pred:.2f} triệu USD")

    # Biểu đồ tròn các chi phí
    st.subheader("Biểu đồ tròn tỷ trọng các chi phí")
    cost_labels = ['Labor Cost', 'Production Cost', 'Other Costs']
    cost_values = [labor, prod, other]
    fig1, ax1 = plt.subplots()
    ax1.pie(cost_values, labels=cost_labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Biểu đồ cột doanh thu và lợi nhuận lịch sử
st.subheader("Biểu đồ cột doanh thu và lợi nhuận lịch sử")
df_bar = df_hist[['Date', 'Revenue (M USD)', 'Profit']].copy()
df_bar['Date'] = pd.to_datetime(df_bar['Date'])
df_bar = df_bar.sort_values('Date')
df_bar.set_index('Date', inplace=True)
st.bar_chart(df_bar)