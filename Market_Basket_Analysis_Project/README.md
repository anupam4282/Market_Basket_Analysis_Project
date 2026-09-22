# Market Basket Analysis

A beginner-friendly Data Analytics / Machine Learning project that identifies products frequently purchased together.

## Objective
- Find frequent item combinations.
- Generate association rules using the Apriori algorithm.
- Evaluate rules using Support, Confidence and Lift.
- Provide business recommendations for cross-selling.

## Tech Stack
- Python
- Pandas
- Streamlit
- MLxtend
- Matplotlib
- Seaborn

## Project Structure
```
market_basket_analysis/
├── app.py
├── requirements.txt
├── README.md
└── data/
    └── transactions.csv
```

## How to Run

1. Open terminal in the project folder.
2. Install packages:
```bash
pip install -r requirements.txt
```
3. Start the dashboard:
```bash
streamlit run app.py
```
4. Open the local URL shown by Streamlit.

## Algorithm
### Apriori
Apriori finds itemsets that occur frequently in transactions and then creates association rules.

### Support
Support tells how often an itemset appears:
`Support(A) = Transactions containing A / Total transactions`

### Confidence
Confidence tells how often B is purchased when A is purchased:
`Confidence(A → B) = Support(A ∪ B) / Support(A)`

### Lift
Lift measures how much stronger an association is compared with random chance:
`Lift(A → B) = Confidence(A → B) / Support(B)`

A lift greater than 1 generally indicates a useful positive association.

## Business Use
The results can be used for:
- Cross-selling
- Product bundling
- Checkout recommendations
- Store/product placement
- Promotional campaigns
- Personalized recommendations
