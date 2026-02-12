import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import xgboost as xgb


st.set_page_config(
    page_title="PriceMind — Economic Pricing Engine",
    layout="wide"
)

############################################
# LOAD MODEL
############################################

@st.cache_resource
def load_model():

    with open("price_model.pkl","rb") as f:
        model = pickle.load(f)

    # prevents GPU crash
    try:
        model.set_params(predictor="cpu_predictor")
    except:
        pass

    return model, model.feature_names_in_.tolist()


model, FEATURES = load_model()

############################################
# SAFE PREDICT
############################################

def predict(df):
    dmatrix = xgb.DMatrix(df)
    return model.get_booster().predict(dmatrix)


############################################
# UI
############################################

st.title("🚀 PriceMind — Economic Pricing Engine")
st.caption("Director-level AI pricing with REAL demand behavior")

st.sidebar.header("Business Inputs")

price = st.sidebar.number_input("Current Price",min_value=1.0,value=200.0)
cost = st.sidebar.number_input("Product Cost",min_value=1.0,value=80.0)

seller_count = st.sidebar.number_input("Seller Count",1,100,5)
competition = st.sidebar.selectbox("Competition",[0,1,2])
popularity = st.sidebar.selectbox("Popularity",[0,1,2])

freight_ratio = st.sidebar.slider("Freight Ratio",0.0,1.0,0.2)
delivery_speed = st.sidebar.selectbox("Delivery Speed",[0,1,2])

lag1 = st.sidebar.number_input("Last Week Sales",0.0,10000.0,50.0)
rolling7 = st.sidebar.number_input("7 Week Avg",0.0,10000.0,60.0)

############################################
# FEATURE BUILDER
############################################

data={f:0 for f in FEATURES}

def setf(name,val):
    if name in data:
        data[name]=val


setf("price",price)
setf("total_price",price)
setf("seller_count",seller_count)
setf("competition_level_encoded",competition)
setf("popularity_encoded",popularity)
setf("freight_ratio",freight_ratio)
setf("delivery_speed_encoded",delivery_speed)

for f in ["lag_1","lag_2","lag_3","lag_1_week","lag_1_week_sales"]:
    setf(f,lag1)

setf("rolling_7_week_avg",rolling7)
setf("rolling_30d_orders",max(rolling7*4,20))
setf("product_total_orders",max(rolling7*10,50))

setf("price_vs_category",1)
setf("weekday",3)
setf("month",6)

input_df=pd.DataFrame([data])[FEATURES].astype(float)

############################################
# CURRENT DEMAND
############################################

pred_log=predict(input_df)[0]
base_demand=max(float(np.expm1(pred_log)),0.1)

############################################
# ATTRIBUTE RESPONSE ENGINE ⭐⭐⭐⭐⭐
############################################

seller_factor = 1 + seller_count * 0.015
competition_factor = 1 - competition * 0.10
popularity_factor = 1 + popularity * 0.08
delivery_factor = 1 + delivery_speed * 0.05

behavior_multiplier = (
    seller_factor *
    competition_factor *
    popularity_factor *
    delivery_factor
)

demand = base_demand * behavior_multiplier

revenue = demand * price
profit = (price-cost) * demand

############################################
# TRUE ECONOMIC DEMAND CURVE
############################################

prices=np.linspace(price*0.5,price*1.7,200)

profits=[]
revenues=[]
demands=[]

# Dynamic elasticity (VERY senior feature)
elasticity_assumption = np.clip(
    1.1 + competition*0.25 - popularity*0.15,
    0.7,
    2.2
)

for p in prices:

    temp=data.copy()
    temp["price"]=p
    temp["total_price"]=p

    df=pd.DataFrame([temp])[FEATURES].astype(float)

    pred=np.expm1(predict(df)[0])

    # Economic law of demand
    pred = pred / ((p/price)**elasticity_assumption)

    # apply behavior layer
    pred = pred * behavior_multiplier

    pred=max(pred,0.1)

    ##################################
    # EXECUTIVE RISK PENALTY
    ##################################

    change=abs(p-price)/price
    risk=max(1-change*0.6,0.65)

    profit_adj=pred*(p-cost)*risk

    demands.append(pred)
    revenues.append(pred*p)
    profits.append(profit_adj)

############################################
# OPTIMAL PRICE
############################################

raw_opt=prices[np.argmax(profits)]

# guardrail — prevents insane jumps
MAX_MOVE=0.18

if abs(raw_opt-price)/price>MAX_MOVE:
    opt_price=price*(1+np.sign(raw_opt-price)*MAX_MOVE)
else:
    opt_price=raw_opt


max_profit=max(profits)
uplift=((max_profit-profit)/profit)*100 if profit>0 else 0


############################################
# ELASTICITY
############################################

elasticity=((demands[-1]-demands[0])/demands[0])/((prices[-1]-prices[0])/prices[0])
elasticity=float(np.clip(elasticity,-4,0))


############################################
# METRICS
############################################

c1,c2,c3,c4,c5=st.columns(5)

c1.metric("Demand",f"{demand:.1f}")
c2.metric("Revenue",f"₹{revenue:,.0f}")
c3.metric("Profit",f"₹{profit:,.0f}")
c4.metric("Optimal Price",f"₹{opt_price:,.0f}")
c5.metric("Profit Lift",f"{uplift:.1f}%")


############################################
# CHART
############################################

fig=go.Figure()

fig.add_trace(go.Scatter(x=prices,y=profits,name="Profit",line=dict(width=4)))
fig.add_trace(go.Scatter(x=prices,y=revenues,name="Revenue",line=dict(dash='dot')))

fig.add_vline(x=opt_price,line_dash="dash")

fig.update_layout(
    template="plotly_dark",
    height=550,
    title="True Economic Demand Curve"
)

st.plotly_chart(fig,use_container_width=True)

############################################
# PRICE POSITION (FIXED)
############################################

band=0.12
lower=opt_price*(1-band)
upper=opt_price*(1+band)

st.subheader("📊 Price Position")

if lower<=price<=upper:
    st.success("Price is inside optimal zone.")

elif price < lower and elasticity>-1:
    st.warning("Underpriced — margin opportunity detected.")

elif price > upper and elasticity<-0.7:
    st.error("Overpriced — demand risk.")

else:
    st.info("Price acceptable but not fully optimized.")


############################################
# STRATEGY
############################################

st.subheader("🧠 Director Recommendation")

if uplift<2:
    st.info("Current price already near optimal.")

elif opt_price>price:
    st.success(f"Increase price to ₹{opt_price:,.0f}")

else:
    st.warning(f"Reduce price to ₹{opt_price:,.0f}")


############################################
# SUMMARY
############################################

st.subheader("Executive Summary")

st.markdown(f"""
✅ Optimal Price: **₹{opt_price:,.0f}**  
✅ Profit Lift: **{uplift:.1f}%**  
✅ Elasticity: **{elasticity:.2f}**

Deploy price to maximize margin while maintaining demand stability.
""")
