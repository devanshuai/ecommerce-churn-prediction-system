import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Churn Prediction System", layout="wide")
st.title("📊 E-commerce Customer Churn Prediction System")

# -------------------------
# LOAD LOTTIE
# -------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

# -------------------------
# LOAD MODEL + FEATURES
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📂 Upload Customer Dataset (CSV)", type=["csv"])

# =========================================================
# ===================== MAIN LOGIC =========================
# =========================================================

if uploaded_file is not None:

# -------------------------
# FILE READ
# -------------------------
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    # -------------------------
    # 📘 DATASET FORMAT GUIDE
    # -------------------------
    with st.expander("📘 Dataset Format Guide"):
        st.write("Your dataset must include these columns:")
        st.write(feature_columns)

    # -------------------------
    # ✅ COLUMN VALIDATION
    # -------------------------
    missing_cols = [col for col in feature_columns if col not in df.columns]

    if missing_cols:
        st.error("❌ Invalid dataset format")
        st.markdown("### ❌ Missing Columns")
        st.dataframe(pd.DataFrame(missing_cols, columns=["Missing Columns"]))

        st.info("Please upload a dataset matching the required schema.")

        st.write("### 📋 Expected Columns:")
        st.dataframe(pd.DataFrame(feature_columns, columns=["Required Columns"]))

        st.stop()

    # -------------------------
    # DATA PREVIEW
    # -------------------------
    st.header("📌 Uploaded Data Preview")
    st.dataframe(df, height=400, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

    # -------------------------
    # PRESERVE CUSTOMER ID
    # -------------------------
    if 'Customer_ID' in df.columns:
        customer_ids = df['Customer_ID']
        df_model = df.drop(columns=['Customer_ID'])
    else:
        customer_ids = None
        df_model = df.copy()

    # -------------------------
    # ENCODING
    # -------------------------
    for col in df_model.select_dtypes(include='object').columns:
        df_model[col] = df_model[col].astype('category').cat.codes

    # -------------------------
    # MATCH TRAINING FEATURES
    # -------------------------
    df_model = df_model[feature_columns]

    # -------------------------
    # PREDICTION
    # -------------------------
    with st.spinner("🔄 Running churn analysis..."):
        preds = model.predict(df_model)
        probs = model.predict_proba(df_model)[:, 1]

    df['Churn_Prediction'] = preds
    df['Churn_Probability'] = probs
    df['Churn_Label'] = df['Churn_Prediction'].map({0: 'No Churn', 1: 'Churn'})
    total = len(df)

    # -------------------------
    # RISK CONSTANTS (IMPORTANT FIX)
    # -------------------------
    HIGH = "High Risk 🔴"
    MEDIUM = "Medium Risk 🟡"
    LOW = "Low Risk 🟢"

    # -------------------------
    # RISK SEGMENTATION
    # -------------------------
    def risk_level(p):
        if p > 0.65:
            return HIGH
        elif p > 0.35:
            return MEDIUM
        else:
            return LOW

    df['Risk_Level'] = df['Churn_Probability'].apply(risk_level)

    # -------------------------
    # 🔥 PERSONALIZED RECOMMENDATIONS
    # -------------------------
    def get_recommendation(row):
        actions = []

        if row['Risk_Level'] == HIGH:

            if row.get('days_since_last_purchase', 0) > 60:
                actions.append("📧 Win-back email (30% OFF)")

            if row.get('engagement_score', 0) < 3:
                actions.append("🎯 Personalized offers")

            if row.get('cart_abandonment_rate', 0) > 0.5:
                actions.append("🛒 Cart recovery discount")

            if row.get('satisfaction_score', 5) < 3:
                actions.append("📞 Feedback call")

        elif row['Risk_Level'] == MEDIUM:

            if row.get('discount_usage_rate', 0) < 0.2:
                actions.append("🎟️ Limited coupon (10–15%)")

            if row.get('browsing_frequency_per_week', 0) > 3:
                actions.append("📢 Product recommendations")

            if row.get('loyalty_member', 0) == 0:
                actions.append("💎 Join loyalty program")

        else:

            if row.get('avg_order_value', 0) > 100:
                actions.append("🛍️ Upsell premium products")

            if row.get('total_orders', 0) > 10:
                actions.append("🎁 VIP rewards")

            actions.append("🚀 Early sale access")

        if not actions:
            return "Maintain engagement"

        return " | ".join(actions)

    df['Recommendation'] = df.apply(get_recommendation, axis=1)

    # -------------------------
    # METRICS
    # -------------------------
    st.header("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk", (df['Risk_Level'] == HIGH).sum())
    col2.metric("📊 Total Customers", total)
    col3.metric("📉 Avg Churn", round(df['Churn_Probability'].mean(), 2))

    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

    # -------------------------
    # DASHBOARD
    # -------------------------
    st.header("📊 Insights Dashboard")

    col1, col2, col3 = st.columns(3)

    # PIE
    with col1:
        fig1, ax1 = plt.subplots()

        counts = df['Churn_Label'].value_counts()

        labels = counts.index
        values = counts.values

        color_map = {
            'No Churn': '#90EE90',
            'Churn': '#FF9999'
        }

        colors = [color_map[label] for label in labels]

        ax1.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops={'edgecolor': 'black'},
            shadow=True
        )

        ax1.set_title("Churn Distribution")

        st.pyplot(fig1)

    # BAR CHART
    with col2:
        fig2, ax2 = plt.subplots()

        counts = df['Risk_Level'].value_counts()

        ax2.bar(
            counts.index,
            counts.values,
            color='#1E90FF',   # bright blue
            edgecolor='black'
        )

        # Add value labels on top
        for i, v in enumerate(counts.values):
            ax2.text(i, v + 20, str(v), ha='center', fontsize=10)

        ax2.set_title("Customer Risk Distribution")
        ax2.set_xlabel("Risk Level")
        ax2.set_ylabel("Number of Customers")

        # Optional: rotate labels for clarity
        ax2.set_xticklabels(counts.index, rotation=20)

        st.pyplot(fig2)

    # LOTTIE
    with col3:
        st_lottie(load_lottie("https://assets1.lottiefiles.com/packages/lf20_qp1q7mct.json"), height=400)

    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

    # =========================================================
    # 🎯 FILTER + SEARCH UI
    # =========================================================
    st.markdown("## 🎯 Explore Customers")

    col1, col2 = st.columns([2, 1])

    with col1:
        filter_option = st.radio(
            "Filter by Risk Level:",
            ["All", HIGH, MEDIUM, LOW],
            horizontal=True
        )

    with col2:
        search_id = st.text_input("🔍 Search Customer ID")

    # =========================================================
    # APPLY FILTER LOGIC
    # =========================================================
    filtered_df = df.copy()

    # Risk filter
    if filter_option != "All":
        filtered_df = filtered_df[filtered_df['Risk_Level'] == filter_option]

    # Search filter
    if search_id and 'Customer_ID' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['Customer_ID'].astype(str).str.contains(search_id)
        ]

    # =========================================================
    # HIGHLIGHT SELECTION
    # =========================================================
    st.success(f"✅ Viewing: {filter_option}")

    # =========================================================
    # DISPLAY FILTERED DATA
    # =========================================================
    st.markdown(f"### 📊 Showing: {filter_option} Customers")
    st.caption(f"{len(filtered_df)} customers based on your selection")

    st.dataframe(
        filtered_df.sort_values(by="Churn_Probability", ascending=False),
        height=400,
        use_container_width=True
    )

    st.markdown("---")

    

    
    # ------------------------
    # CUSTOMER DRILL DOWN 
    # ------------------------

    
    st.header("🔍 Customer Deep Dive")
    if 'Customer_ID' in filtered_df.columns:
        selected_id = st.selectbox(
            "Select Customer ID",
            filtered_df['Customer_ID'].astype(str)
        )

        selected_customer = filtered_df[
            filtered_df['Customer_ID'].astype(str) == selected_id
        ].iloc[0]

    st.markdown("### 📌 Customer Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Churn Status", selected_customer['Churn_Label'])
    col2.metric("⚠️ Risk Level", selected_customer['Risk_Level'])
    col3.metric("📉 Churn Probability", round(selected_customer['Churn_Probability'], 2))

    # -------------------------
    # WHY CHURN (IMPROVED)
    # -------------------------
    def get_churn_reason(row, model, feature_columns):
        try:
            import pandas as pd
            
            importance = pd.Series(model.feature_importances_, index=feature_columns)
            top_features = importance.sort_values(ascending=False).head(5).index

            reasons = []

            for feature in top_features:
                if feature in row:
                    value = row[feature]

                    if "days_since" in feature and value > 30:
                        reasons.append("Inactive for long time")

                    elif "engagement" in feature and value < 3:
                        reasons.append("Low engagement")

                    elif "satisfaction" in feature and value < 3:
                        reasons.append("Low satisfaction")

                    elif "cart_abandonment" in feature and value > 0.5:
                        reasons.append("High cart abandonment")

                    elif "discount_usage" in feature and value < 0.2:
                        reasons.append("Not responding to offers")

            return reasons[:3] if reasons else ["No strong negative signals detected"]

        except:
            return ["Model explanation not available"]


    # -------------------------
    # DYNAMIC HEADING + LOGIC
    # -------------------------
    risk = selected_customer['Risk_Level']

    # Dynamic heading
    if "High Risk" in risk:
        st.markdown("### ❌ Why this customer is likely to churn")
        st.error("⚠️ Critical churn risk detected")

    elif "Medium Risk" in risk:
        st.markdown("### 🚨 Potential churn signals")
        st.warning("⚠️ This customer needs attention")

    else:
        st.markdown("### ✅ No Worries")


    # Content logic (FIXED)
    if "Low Risk" in risk:
        st.success("✅ Customer is stable. No churn risk indicators detected.")

    else:
        reasons = get_churn_reason(selected_customer, model, feature_columns)

        for r in reasons:
            st.write(f"⚠️ {r}")


    # -------------------------
    # RECOMMENDED ACTIONS
    # -------------------------
    st.markdown("### 🎯 Recommended Actions")

    actions = selected_customer['Recommendation'].split(" | ")

    for a in actions:
        st.write(f"👉 {a}")


    # -------------------------
    # INSIGHTS + LOTTIE
    # -------------------------

    st.markdown("---")
    col1, col2 = st.columns([2,1])

    with col1:
        st.header("🧠 Key Insights")

        high = (df['Risk_Level']=="High Risk 🔴").sum()
        medium = (df['Risk_Level']=="Medium Risk 🟡").sum()
        low = (df['Risk_Level']=="Low Risk 🟢").sum()

        st.markdown(f"""
### 🔴 High Risk ({high})
- 🎁 Offer personalized discounts (e.g., *Flat ₹100 off*)
- 🛒 Cart recovery offers (*Shop above ₹500 → 20% OFF*)
- 📞 Direct outreach via email/SMS 

### 🟡 Medium Risk ({medium})
- 🎯 Loyalty points / rewards
- 🎟️ Limited-time coupons (*10% OFF weekend*)
- 📢 Product recommendations

### 🟢 Low Risk ({low})
- 💎 Loyalty programs
- 🛍️ Upsell premium products
- 📦 Early access to sales  
""")

    with col2:
        st_lottie(load_lottie("https://assets4.lottiefiles.com/packages/lf20_49rdyysj.json"), height=400)

    # -------------------------
    # DOWNLOAD
    # -------------------------
    st.markdown("---")

    st.success("✅ Analysis Complete!")

    # -------------------------
    # FILTER UI
    # -------------------------
    st.markdown("### 🎯 Select Data to Download")

    selected_filter = st.radio(
        "Choose Risk Segment:",
        ["All", "High Risk 🔴", "Medium Risk 🟡", "Low Risk 🟢"],
        horizontal=True
    )

    if selected_filter == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df['Risk_Level'] == selected_filter]


    # -------------------------
    # PREP CLEAN CSV (REMOVE EMOJIS)
    # -------------------------
    import re

    def remove_emojis(text):
        if isinstance(text, str):
            return re.sub(r'[^\x00-\x7F]+', '', text)
        return text

    csv = df.copy()
    csv['Churn_Probability'] = csv['Churn_Probability'].round(3)

    # Remove emojis from key columns
    csv['Risk_Level'] = csv['Risk_Level'].apply(remove_emojis)
    csv['Recommendation'] = csv['Recommendation'].apply(remove_emojis)

    # Encode properly for Excel compatibility
    csv_data = csv.to_csv(index=False).encode('utf-8-sig')


    # -------------------------
    # CSV DOWNLOAD (CLEAN)
    # -------------------------
    st.download_button(
        label="📥 Download Full Analysis (CSV)",
        data=csv_data,
        file_name="churn_analysis_results.csv",
        mime="text/csv"
    )


    # -------------------------
    # EXCEL DOWNLOAD (WITH EMOJIS)
    # -------------------------
    import io
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Customer Data', index=False)

        summary = pd.DataFrame({
            "Metric": ["Total Customers", "High Risk", "Medium Risk", "Low Risk"],
            "Value": [
                len(df),
                (df['Risk_Level'] == "High Risk 🔴").sum(),
                (df['Risk_Level'] == "Medium Risk 🟡").sum(),
                (df['Risk_Level'] == "Low Risk 🟢").sum()
            ]
        })

        summary.to_excel(writer, sheet_name='Summary Report', index=False)

    st.download_button(
        label="📊 Download Excel Report",
        data=buffer.getvalue(),
        file_name="churn_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


    # -------------------------
    # FILTERED CSV DOWNLOAD (CLEAN)
    # -------------------------
    filtered_csv = filtered_df.copy()

    filtered_csv['Churn_Probability'] = filtered_csv['Churn_Probability'].round(3)
    filtered_csv['Risk_Level'] = filtered_csv['Risk_Level'].apply(remove_emojis)
    filtered_csv['Recommendation'] = filtered_csv['Recommendation'].apply(remove_emojis)

    filtered_csv_data = filtered_csv.to_csv(index=False).encode('utf-8-sig')

    st.download_button(
        label=f"📥 Download {selected_filter} Data",
        data=filtered_csv_data,
        file_name=f"{selected_filter.replace(' ', '_').lower()}_customers.csv",
        mime="text/csv"
    )

# =========================================================
# ===================== HOMEPAGE ===========================
# =========================================================

else:
    lottie = load_lottie("https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json")

    col1, col2 = st.columns([1, 2])

    with col1:
        st_lottie(lottie, height=330)

    with col2:
        st.markdown("## 🚀 Welcome to the Churn Prediction System")
        st.markdown("""
        This intelligent system helps **e-commerce businesses** identify customers who are likely to churn.

        🔍 Upload your dataset and get:
        - Churn predictions  
        - Customer risk analysis  
        - Actionable recommendations  
        - Visual insights  
        """)

    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Predict Churn")
        st.markdown("Identify customers likely to stop purchasing.")

    with col2:
        st.markdown("### 📈 Visual Insights")
        st.markdown("Understand trends with interactive charts.")

    with col3:
        st.markdown("### 💡 Smart Recommendations")
        st.markdown("Get actionable strategies to retain customers.")

    st.info("⬆️ Upload your dataset above to get started")