import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="Transaction Review Co-Pilot", layout="wide")

if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
if 'reviewed_ids' not in st.session_state:
    st.session_state.reviewed_ids = []
if 'manual_review_ids' not in st.session_state:
    st.session_state.manual_review_ids = []
if 'confirmed_fraud_ids' not in st.session_state:
    st.session_state.confirmed_fraud_ids = []
if 'show_shap_for_id' not in st.session_state:
    st.session_state.show_shap_for_id = None


@st.cache_data
def get_processed_data():
    """Loads data, runs models, and returns a fully processed DataFrame."""
    iso_forest_model = joblib.load('models/isolation_forest_model.joblib')
    features = joblib.load('models/features.joblib')
    # df = pd.read_csv('transactions_sample.csv')
    df = pd.read_csv('data/transactions_mini.csv')

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Transaction ID'}, inplace=True)
    
    df_processed = pd.get_dummies(df.copy(), columns=['type'], prefix='type')
    

    for col in features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    X = df_processed[features]
    
    df['anomaly_score'] = iso_forest_model.decision_function(X)
    min_score, max_score = df['anomaly_score'].min(), df['anomaly_score'].max()
    df['risk_score'] = (1 - (df['anomaly_score'] - min_score) / (max_score - min_score)) * 100
    
    return df, X

@st.cache_resource
def load_explainer():
    """Loads the SHAP explainer."""
    xgb_surrogate_model = joblib.load('models/xgb_surrogate_model.joblib')
    explainer = shap.TreeExplainer(xgb_surrogate_model)
    return explainer

def categorize_anomaly(row, user_avg_amount):

    avg_amount = user_avg_amount.get(row['nameOrig'], row['amount'])
    if row['oldbalanceOrg'] == 0 and row['newbalanceOrig'] == 0 and row['amount'] > 0:
        return "Suspicious Balance"
    if row['amount'] > (avg_amount * 10) and row['amount'] > 10000:
        return "Amount Anomaly"
    if row['type'] in ['TRANSFER', 'CASH_OUT']:
        return "High-Risk Type"
    return "General Anomaly"
    
def get_category_display(category):
    if category == "High-Risk Type": return f":red[{category} ]"
    if category in ["Suspicious Balance", "Amount Anomaly"]: return f":orange[{category} ]"
    return category

def to_excel(df_export):
    """Converts a DataFrame to an in-memory Excel file."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Report')
    processed_data = output.getvalue()
    return processed_data


df, X = get_processed_data()
explainer = load_explainer()
user_avg_amount = df.groupby('nameOrig')['amount'].mean().to_dict() # Fast enough to not need caching


st.title("Transaction Review Co-Pilot ")
st.subheader("High-Risk Transaction Review Queue")

risk_threshold = 90
review_queue = df[(df['risk_score'] >= risk_threshold) & (~df['Transaction ID'].isin(st.session_state.reviewed_ids))].sort_values(by='risk_score', ascending=False)

items_per_page = 10
total_items = len(review_queue)
total_pages = max(1, (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0))
st.session_state.page_number = max(0, min(st.session_state.page_number, total_pages - 1))
start_idx = st.session_state.page_number * items_per_page
end_idx = start_idx + items_per_page
paginated_queue = review_queue.iloc[start_idx:end_idx]

st.write(f"Displaying {len(paginated_queue)} of {total_items} transactions for review (Page {st.session_state.page_number + 1} of {total_pages}).")

col_headers = st.columns((0.5, 1, 1.5, 3))
col_headers[0].write("**ID**"); col_headers[1].write("**Amount**"); col_headers[2].write("**Category**"); col_headers[3].write("**Actions**")
st.divider()


for index, row in paginated_queue.iterrows():
    category = categorize_anomaly(row, user_avg_amount)
    col1, col2, col3, col4 = st.columns((0.5, 1, 1.5, 3))
    
    col1.write(f"`{row['Transaction ID']}`")
    col2.write(f"₹{row['amount']:,.2f}")
    col3.markdown(get_category_display(category))
        
    with col4:
        action_cols = st.columns(4)
        if action_cols[0].button("See SHAP", key=f"shap_{row['Transaction ID']}"):
            st.session_state.show_shap_for_id = row['Transaction ID']
        if action_cols[1].button("Safe", key=f"safe_{row['Transaction ID']}"):
            st.session_state.reviewed_ids.append(row['Transaction ID'])
            st.rerun()
        if action_cols[2].button("Fraud", type="primary", key=f"fraud_{row['Transaction ID']}"):
            st.session_state.reviewed_ids.append(row['Transaction ID'])
            st.session_state.confirmed_fraud_ids.append(row['Transaction ID'])
            st.rerun()
        if action_cols[3].button("Review", key=f"manual_{row['Transaction ID']}"):
            st.session_state.reviewed_ids.append(row['Transaction ID'])
            st.session_state.manual_review_ids.append(row['Transaction ID'])
            st.rerun()

    if st.session_state.show_shap_for_id == row['Transaction ID']:
        transaction_features = X.loc[index:index]
        shap_values = explainer(transaction_features)
        
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150) 
        shap.plots.waterfall(shap_values[0], max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        if st.button("Close Explanation", key=f"close_{row['Transaction ID']}"):
            st.session_state.show_shap_for_id = None
            st.rerun()
    st.divider()


nav_cols = st.columns(3)
if nav_cols[0].button("⬅️ Previous", disabled=(st.session_state.page_number == 0)):
    st.session_state.page_number -= 1; st.session_state.show_shap_for_id = None; st.rerun()
if nav_cols[2].button("Next ➡️", disabled=(st.session_state.page_number >= total_pages - 1)):
    st.session_state.page_number += 1; st.session_state.show_shap_for_id = None; st.rerun()
    

st.sidebar.title("Reports")
if st.session_state.confirmed_fraud_ids:
    fraud_df = df[df['Transaction ID'].isin(st.session_state.confirmed_fraud_ids)]
    st.sidebar.download_button(label="Download Fraud Report", data=to_excel(fraud_df), file_name="confirmed_fraud_report.xlsx")
if st.session_state.manual_review_ids:
    manual_df = df[df['Transaction ID'].isin(st.session_state.manual_review_ids)]
    st.sidebar.download_button(label="Download Manual Review Report", data=to_excel(manual_df), file_name="manual_review_report.xlsx")