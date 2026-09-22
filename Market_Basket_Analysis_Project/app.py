
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")

st.title("🛒 Market Basket Analysis")
st.caption("Discover products that are frequently purchased together and turn them into cross-selling opportunities.")

@st.cache_data
def load_data():
    return pd.read_csv("data/transactions.csv")

df = load_data()

st.sidebar.header("Analysis Settings")
min_support = st.sidebar.slider("Minimum Support", 0.05, 0.80, 0.15, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.10, 1.00, 0.50, 0.05)
top_n = st.sidebar.slider("Top Items / Itemsets", 5, 20, 10)

# Basket matrix
basket = (df.assign(value=1)
          .pivot_table(index="TransactionID", columns="Item", values="value", aggfunc="max", fill_value=0)
          .astype(bool))

freq = apriori(basket, min_support=min_support, use_colnames=True)
if not freq.empty:
    freq["length"] = freq["itemsets"].apply(len)
    freq["items"] = freq["itemsets"].apply(lambda x: ", ".join(sorted(x)))
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        rules["rule"] = rules["antecedents_str"] + " → " + rules["consequents_str"]
else:
    rules = pd.DataFrame()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", basket.shape[0])
c2.metric("Unique Products", basket.shape[1])
c3.metric("Frequent Itemsets", len(freq))
c4.metric("Association Rules", len(rules))

tab1, tab2, tab3, tab4 = st.tabs(["📊 Frequent Items", "🔗 Association Rules", "💡 Business Insights", "📁 Data"])

with tab1:
    st.subheader("Top Products by Transaction Frequency")
    item_counts = df.groupby("Item")["TransactionID"].nunique().sort_values(ascending=False).head(top_n)
    st.bar_chart(item_counts)

    st.subheader("Frequent Itemsets")
    if freq.empty:
        st.warning("No frequent itemsets found. Lower the minimum support.")
    else:
        show = freq.sort_values(["support","length"], ascending=False).head(top_n)[["items","support","length"]].copy()
        show["support"] = (show["support"] * 100).round(2).astype(str) + "%"
        show.columns = ["Itemset", "Support", "Size"]
        st.dataframe(show, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Strong Association Rules")
    if rules.empty:
        st.warning("No rules found. Lower minimum confidence or support.")
    else:
        show = rules.sort_values(["lift","confidence"], ascending=False).head(top_n)[
            ["rule","support","confidence","lift"]
        ].copy()
        show["support"] = (show["support"] * 100).round(2).astype(str) + "%"
        show["confidence"] = (show["confidence"] * 100).round(2).astype(str) + "%"
        show["lift"] = show["lift"].round(2)
        show.columns = ["Rule","Support","Confidence","Lift"]
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.info("Lift > 1 indicates the products occur together more often than would be expected if they were independent.")

with tab3:
    st.subheader("Actionable Business Insights")
    if rules.empty:
        st.write("Run the analysis with lower thresholds to generate insights.")
    else:
        best = rules.sort_values(["lift","confidence"], ascending=False).head(5)
        for i, (_, r) in enumerate(best.iterrows(), 1):
            antecedent = ", ".join(sorted(r["antecedents"]))
            consequent = ", ".join(sorted(r["consequents"]))
            st.markdown(
                f"**{i}. Cross-sell {consequent} when customers buy {antecedent}.**  \n"
                f"Confidence: **{r['confidence']:.1%}** · Lift: **{r['lift']:.2f}** · Support: **{r['support']:.1%}**"
            )
        st.markdown("### Recommended Strategies")
        st.markdown("""
        - **Bundle offers:** Create combo packs for high-lift product pairs.
        - **Cart recommendations:** Show the consequent product during checkout.
        - **Store placement:** Place strongly associated products closer together.
        - **Promotions:** Offer a small discount on the recommended add-on.
        - **Personalization:** Use customer baskets to recommend complementary products.
        """)

with tab4:
    st.subheader("Transaction Data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "transactions.csv",
        "text/csv"
    )

st.divider()
st.caption("Algorithm: Apriori | Metrics: Support, Confidence, Lift | Dataset: Sample grocery transactions")
