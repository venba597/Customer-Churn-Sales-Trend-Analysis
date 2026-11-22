
# ==========================================================
# PROJECT 03: CUSTOMER CHURN / SALES TREND ANALYSIS
# Data Science and AI - InternCircle
# Tools: Python, Pandas, NumPy, Matplotlib, Seaborn
# Dataset: Simulated E-commerce Orders
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. SET VISUALIZATION STYLE
# ----------------------------------------------------------

sns.set_theme(style="whitegrid", palette="colorblind")

print("Libraries imported successfully!")

# ----------------------------------------------------------
# 2. GENERATE SAMPLE E-COMMERCE DATASET
# ----------------------------------------------------------

print("\nGenerating E-commerce Dataset...")

rng = np.random.default_rng(42)

n = 2000

orders = pd.DataFrame({
    "order_date": (
        pd.to_datetime("2025-01-01")
        + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    ),
    "customer_id": rng.integers(1, 301, n),
    "category": rng.choice(
        ["Electronics", "Fashion", "Home", "Books"],
        n,
        p=[0.30, 0.35, 0.20, 0.15]
    ),
    "amount": rng.gamma(2.0, 600, n).round(2)
})

orders["order_id"] = range(1, n + 1)

print("\nDataset Generated Successfully!")

print("\nFirst 5 Records:")
print(orders.head())

print("\nDataset Shape:")
print(orders.shape)

print("\nDataset Information:")
orders.info()

print("\nStatistical Description:")
print(orders.describe())

# ----------------------------------------------------------
# 3. DATA INSPECTION
# ----------------------------------------------------------

print("\n========== DATA INSPECTION ==========")

print("\nMissing Values:")
print(orders.isnull().sum())

print("\nDuplicate Records:")
print(orders.duplicated().sum())

# ----------------------------------------------------------
# 4. FEATURE ENGINEERING - DATE EXTRACTION
# ----------------------------------------------------------

print("\n========== FEATURE ENGINEERING ==========")

orders["order_date"] = pd.to_datetime(orders["order_date"])

orders["month"] = orders["order_date"].dt.to_period("M")

orders["month_name"] = orders["order_date"].dt.month_name()

orders["weekday"] = orders["order_date"].dt.day_name()

orders["quarter"] = orders["order_date"].dt.quarter

print("\nDate Features Created Successfully!")

print(orders.head())

# ----------------------------------------------------------
# 5. MONTHLY REVENUE ANALYSIS
# ----------------------------------------------------------

print("\n========== MONTHLY REVENUE ANALYSIS ==========")

monthly_revenue = orders.groupby("month")["amount"].sum()

monthly_growth = monthly_revenue.pct_change() * 100

print("\nMonthly Revenue:")
print(monthly_revenue.round(2))

print("\nMonth-over-Month Growth (%):")
print(monthly_growth.round(2))

best_month = monthly_revenue.idxmax()

worst_month = monthly_revenue.idxmin()

print("\nBest Revenue Month:", best_month)
print("Worst Revenue Month:", worst_month)

# ----------------------------------------------------------
# 6. CATEGORY ANALYSIS
# ----------------------------------------------------------

print("\n========== CATEGORY ANALYSIS ==========")

category_analysis = orders.groupby("category")["amount"].agg(
    Total_Revenue="sum",
    Average_Order_Value="mean",
    Number_of_Orders="count"
).sort_values("Total_Revenue", ascending=False)

print("\nCategory Performance:")
print(category_analysis.round(2))

best_category = category_analysis["Total_Revenue"].idxmax()

print("\nBest Performing Category:", best_category)

# ----------------------------------------------------------
# 7. PIVOT TABLE - MONTH VS CATEGORY
# ----------------------------------------------------------

print("\n========== PIVOT TABLE ==========")

pivot_table = pd.pivot_table(
    orders,
    values="amount",
    index="month",
    columns="category",
    aggfunc="sum",
    fill_value=0
)

print("\nMonthly Category Revenue:")
print(pivot_table.round(2))

# ----------------------------------------------------------
# 8. TOP CUSTOMER ANALYSIS - PARETO IDEA
# ----------------------------------------------------------

print("\n========== CUSTOMER ANALYSIS ==========")

customer_spending = (
    orders.groupby("customer_id")["amount"]
    .sum()
    .sort_values(ascending=False)
)

total_revenue = customer_spending.sum()

number_of_customers = len(customer_spending)

top_10_percent_count = max(
    1,
    int(np.ceil(number_of_customers * 0.10))
)

top_customers = customer_spending.head(top_10_percent_count)

top_customer_revenue = top_customers.sum()

top_customer_share = (
    top_customer_revenue / total_revenue
) * 100

print("\nTotal Customers:", number_of_customers)

print("\nTop 10% Customers:")
print(top_customers)

print("\nTop 10% Customer Revenue:", round(top_customer_revenue, 2))

print("Total Revenue:", round(total_revenue, 2))

print(
    "Top 10% Customers Revenue Contribution:",
    round(top_customer_share, 2),
    "%"
)

# ----------------------------------------------------------
# 9. CREATE DASHBOARD
# ----------------------------------------------------------

print("\n========== CREATING DASHBOARD ==========")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# GRAPH 1: MONTHLY REVENUE TREND

sns.lineplot(
    x=monthly_revenue.index.astype(str),
    y=monthly_revenue.values,
    marker="o",
    linewidth=2,
    color="steelblue",
    ax=axes[0, 0]
)

axes[0, 0].set_title("Monthly Revenue Trend")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Revenue")
axes[0, 0].tick_params(axis="x", rotation=45)

# GRAPH 2: CATEGORY REVENUE

sns.barplot(
    x=category_analysis.index,
    y=category_analysis["Total_Revenue"],
    hue=category_analysis.index,
    palette="Set2",
    legend=False,
    ax=axes[0, 1]
)

axes[0, 1].set_title("Revenue by Product Category")
axes[0, 1].set_xlabel("Category")
axes[0, 1].set_ylabel("Total Revenue")
axes[0, 1].tick_params(axis="x", rotation=20)

# GRAPH 3: MONTHLY CATEGORY HEATMAP

sns.heatmap(
    pivot_table,
    annot=True,
    fmt=".0f",
    cmap="YlGnBu",
    linewidths=0.5,
    ax=axes[1, 0]
)

axes[1, 0].set_title("Monthly Revenue by Category")
axes[1, 0].set_xlabel("Category")
axes[1, 0].set_ylabel("Month")

# GRAPH 4: TOP 10 CUSTOMERS

top_customers.head(10).sort_values().plot(
    kind="barh",
    color="teal",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Top 10 Customers by Revenue")
axes[1, 1].set_xlabel("Total Spending")
axes[1, 1].set_ylabel("Customer ID")

# DASHBOARD TITLE

plt.suptitle(
    "E-COMMERCE SALES TREND ANALYSIS DASHBOARD",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

# SAVE DASHBOARD

plt.savefig(
    "PROJECT03_Dashboard.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nDashboard saved successfully!")

# ----------------------------------------------------------
# 10. BUSINESS FINDINGS
# ----------------------------------------------------------

print("\n========== KEY BUSINESS FINDINGS ==========")

print(
    f"1. Best revenue month: {best_month} "
    f"with revenue of {monthly_revenue.max():.2f}"
)

print(
    f"2. Best product category: {best_category} "
    f"with revenue of {category_analysis.loc[best_category, 'Total_Revenue']:.2f}"
)

print(
    f"3. Top 10% of customers contribute "
    f"{top_customer_share:.2f}% of total revenue."
)

# ----------------------------------------------------------
# 11. BUSINESS RECOMMENDATIONS
# ----------------------------------------------------------

print("\n========== BUSINESS RECOMMENDATIONS ==========")

print(
    f"1. Focus marketing campaigns on {best_category}, "
    "the highest-revenue category."
)

print(
    f"2. Introduce loyalty rewards for the top 10% of customers, "
    f"who contribute {top_customer_share:.2f}% of revenue."
)

print(
    f"3. Study sales activities during {best_month} "
    "and plan promotions and inventory around high-revenue periods."
)

# ----------------------------------------------------------
# 12. SAVE ANALYSIS RESULTS
# ----------------------------------------------------------

orders.to_csv("PROJECT03_Orders.csv", index=False)

category_analysis.to_csv("PROJECT03_Category_Analysis.csv")

monthly_revenue.to_csv("PROJECT03_Monthly_Revenue.csv")

print("\nDataset and analysis files saved successfully!")

print("\n==========================================")
print("PROJECT 03 COMPLETED SUCCESSFULLY!")
print("==========================================")
