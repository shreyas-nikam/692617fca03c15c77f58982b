import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# --- Utility Functions ---
@st.cache_data(ttl="2h")
def initialize_investment_environment(initial_cash: float, stock_symbols: list, initial_prices: dict, volatility: float) -> dict:
    market_data = {symbol: {"price": initial_prices.get(symbol, 100.0), "volatility": volatility} for symbol in stock_symbols}
    return {
        "cash": initial_cash,
        "portfolio": {symbol: 0 for symbol in stock_symbols}, # Initial holdings are zero
        "market_data": market_data,
        "current_prices": {symbol: initial_prices.get(symbol, 100.0) for symbol in stock_symbols},
        "volatility_factor": volatility
    }

@st.cache_data(ttl="2h")
def simulate_market_movement(current_market_data: dict, volatility_factor: float) -> dict:
    new_market_data = {}
    for symbol, data in current_market_data.items():
        current_price = data["price"]
        # Simulate a random walk with some volatility
        change = np.random.normal(0, volatility_factor)
        new_price = max(1.0, current_price * (1 + change)) # Price cannot go below 1.0
        new_market_data[symbol] = {"price": new_price, "volatility": volatility_factor}
    return new_market_data

def execute_trade(portfolio, stock_id, quantity, price, trade_type):
    trade_status = "SUCCESS"
    reason = ""
    if trade_type == "BUY":
        cost = quantity * price
        if portfolio["cash"] >= cost:
            portfolio["cash"] -= cost
            portfolio["holdings"][stock_id] = portfolio["holdings"].get(stock_id, 0) + quantity
            reason = f"Bought {quantity} of {stock_id} at ${price:.2f}"
        else:
            trade_status = "FAILED"
            reason = f"Insufficient cash to buy {quantity} of {stock_id} at ${price:.2f}"
    elif trade_type == "SELL":
        if portfolio["holdings"].get(stock_id, 0) >= quantity:
            portfolio["cash"] += quantity * price
            portfolio["holdings"][stock_id] -= quantity
            if portfolio["holdings"][stock_id] == 0:
                del portfolio["holdings"][stock_id]
            reason = f"Sold {quantity} of {stock_id} at ${price:.2f}"
        else:
            trade_status = "FAILED"
            reason = f"Insufficient holdings to sell {quantity} of {stock_id}"
    else:
        trade_status = "FAILED"
        reason = f"Unknown trade type: {trade_type}"

    return portfolio, {"type": trade_type, "stock_id": stock_id, "quantity": quantity, "price": price, "status": trade_status, "reason": reason}

@st.cache_data(ttl="2h")
def calculate_portfolio_value(portfolio, prices):
    value = portfolio["cash"]
    for stock_id, quantity in portfolio["holdings"].items():
        if stock_id in prices:
            value += quantity * prices[stock_id]
    return value

def calculate_reward(portfolio_value: float, initial_portfolio_value: float, risk_taken: float, target_returns: float, risk_aversion: float) -> float:
    profit = portfolio_value - initial_portfolio_value # Profit for the current step/period
    risk_penalty = risk_aversion * (risk_taken ** 2) # Penalize higher risk, square for increasing penalty
    return profit - risk_penalty

# --- Agent Classes ---
class PlannerAgent:
    def __init__(self, objective: str, risk_tolerance: str, initial_prices: dict):
        self.objective = objective
        self.risk_tolerance = risk_tolerance
        self.initial_prices = initial_prices
        st.session_state.setdefault("planner_plan_history", [])
        self.original_objective = objective # Store original objective for reset or comparison

    def plan(self, current_state: dict, mis_specify_goal: bool = False) -> list:
        plan_actions = []
        current_cash = current_state["cash"]
        current_holdings = current_state["portfolio"]
        market_prices = current_state["current_prices"]

        # Adjust objective if mis-specification is active
        effective_objective = self.objective
        if mis_specify_goal:
            if self.objective == "Maximize returns":
                effective_objective = "Invest aggressively in volatile assets"
            elif self.objective == "Safely grow portfolio":
                effective_objective = "Take high risks for short-term gains"
            elif self.objective == "Execute high volume trades regardless of risk": # If agent misperception fault changed it
                 effective_objective = "Execute high volume trades regardless of risk"
            else:
                effective_objective = "Maximize short-term trading volume"
        
        st.session_state.current_objective_display = effective_objective # To show in UI
        
        # Strategy based on effective_objective and risk_tolerance
        if "volatile assets" in effective_objective.lower() or "high risks" in effective_objective.lower() or "high volume trades" in effective_objective.lower():
            # Aggressive strategy
            for symbol, price_data in market_prices.items():
                if current_cash > price_data: # Enough cash to buy at least one
                    quantity_to_buy = int((current_cash * 0.2) / price_data) # Buy 20% of available cash value
                    if quantity_to_buy > 0:
                        plan_actions.append({"action": "BUY", "stock_id": symbol, "quantity": quantity_to_buy, "price": price_data})
                        current_cash -= quantity_to_buy * price_data
            if current_holdings:
                for symbol, quantity in list(current_holdings.items()): # Iterate on a copy
                    if np.random.rand() > 0.7: # Randomly sell some high-risk assets
                        quantity_to_sell = int(quantity * 0.3)
                        if quantity_to_sell > 0:
                            plan_actions.append({"action": "SELL", "stock_id": symbol, "quantity": quantity_to_sell, "price": market_prices[symbol]})

        elif "maximize returns" in effective_objective.lower():
            # Moderate strategy
            for symbol, price_data in market_prices.items():
                # Check if current price is lower than initial price, suggesting a buying opportunity (oversimplified)
                if price_data < self.initial_prices.get(symbol, price_data * 1.05) and current_cash > price_data:
                    quantity_to_buy = int((current_cash * 0.1) / price_data) # Buy 10% of available cash value
                    if quantity_to_buy > 0:
                        plan_actions.append({"action": "BUY", "stock_id": symbol, "quantity": quantity_to_buy, "price": price_data})
                        current_cash -= quantity_to_buy * price_data
            # Consider selling if price significantly higher than initial
            for symbol, quantity in list(current_holdings.items()): # Iterate on a copy
                if market_prices[symbol] > self.initial_prices.get(symbol, market_prices[symbol] * 0.95) * 1.1: # 10% profit
                    quantity_to_sell = int(quantity * 0.2)
                    if quantity_to_sell > 0:
                        plan_actions.append({"action": "SELL", "stock_id": symbol, "quantity": quantity_to_sell, "price": market_prices[symbol]})

        elif "safely grow portfolio" in effective_objective.lower():
            # Conservative strategy
            for symbol, price_data in market_prices.items():
                if current_cash > price_data: # Ensure there's cash
                    quantity_to_buy = int((current_cash * 0.05) / price_data) # Buy 5% of available cash value
                    if quantity_to_buy > 0:
                        plan_actions.append({"action": "BUY", "stock_id": symbol, "quantity": quantity_to_buy, "price": price_data})
                        current_cash -= quantity_to_buy * price_data
            # Conservative agents typically don't sell aggressively unless significant loss

        st.session_state.planner_plan_history.append({"step": st.session_state.current_step, "plan": plan_actions, "objective": effective_objective})
        return plan_actions

class ExecutorAgent:
    def __init__(self):
        st.session_state.setdefault("trade_log", [])

    def execute(self, action_details: dict, environment_state: dict) -> (dict, dict):
        portfolio = st.session_state.portfolio # Get current portfolio from session state
        
        stock_id = action_details["stock_id"]
        quantity = action_details["quantity"]
        trade_type = action_details["action"]
        price = environment_state["current_prices"].get(stock_id, 0) # Get actual current price

        updated_portfolio, trade_record = execute_trade(portfolio, stock_id, quantity, price, trade_type)
        
        # Update session state with new portfolio and trade log
        st.session_state.portfolio = updated_portfolio
        
        # The trade_record already has the step from the executor call, ensure it's there
        # No, trade_record comes from execute_trade, doesn't have step. Add it here.
        trade_record_with_step = {"step": st.session_state.current_step, **trade_record}
        st.session_state.trade_log.append(trade_record_with_step)
        
        # Update environment_state's cash and portfolio (which is a reference to st.session_state.portfolio)
        environment_state["cash"] = st.session_state.portfolio["cash"]
        environment_state["portfolio"] = st.session_state.portfolio["holdings"] # Only holdings here

        return st.session_state.portfolio, trade_record_with_step

class CriticAgent:
    def __init__(self, target_metrics: dict):
        self.target_metrics = target_metrics
        st.session_state.setdefault("critic_feedback_history", [])

    def evaluate(self, environment_state: dict, current_objective: str, previous_portfolio_value: float, risk_tolerance: str) -> dict:
        current_portfolio_value = calculate_portfolio_value(st.session_state.portfolio, environment_state["current_prices"])
        
        profit = current_portfolio_value - previous_portfolio_value # Profit for this step
        total_profit_since_init = current_portfolio_value - st.session_state.initial_cash # Total profit
        
        # Simplified risk calculation: higher risk if many trades or if portfolio deviates significantly
        num_trades_this_step = len([t for t in st.session_state.trade_log if t["step"] == st.session_state.current_step])
        
        # Determine risk_taken based on current actions and risk tolerance
        # A more sophisticated risk model would use historical volatility, VaR, etc.
        risk_taken = 0.0
        if num_trades_this_step > 0:
            # Example: Risk increases with the square of trade count and risk tolerance setting
            risk_multiplier = {"safe": 0.01, "moderate": 0.03, "aggressive": 0.06}.get(risk_tolerance.lower(), 0.03)
            risk_taken = num_trades_this_step * risk_multiplier

        # Add penalty if current actions contradict the effective objective (mis-specification)
        if ("high risks" in current_objective.lower() or "aggressive" in current_objective.lower() or "high volume trades" in current_objective.lower()) and risk_tolerance == "safe":
            risk_taken += 0.1 # Significant penalty for mis-specified goal leading to risky actions
        elif "safely grow portfolio" in current_objective.lower() and risk_tolerance == "aggressive":
            risk_taken += 0.05 # Smaller penalty for conservative agent being forced to be aggressive


        risk_penalty = self.target_metrics["risk_aversion"] * (risk_taken ** 2)
        
        reward = calculate_reward(current_portfolio_value, previous_portfolio_value, risk_taken, self.target_metrics["target_returns"], self.target_metrics["risk_aversion"])
        
        feedback = {
            "step": st.session_state.current_step,
            "portfolio_value": current_portfolio_value,
            "profit_component": profit,
            "risk_taken": risk_taken,
            "risk_penalty": risk_penalty,
            "calculated_reward": reward,
            "message": "Portfolio updated."
        }

        if current_portfolio_value < previous_portfolio_value * (1 - self.target_metrics["max_risk"]) and previous_portfolio_value != 0: # Avoid division by zero
            feedback["message"] = f"WARNING: Portfolio value dropped significantly beyond max risk ({self.target_metrics['max_risk']*100:.2f}%)."
            feedback["status"] = "warning"
        elif total_profit_since_init < st.session_state.initial_cash * self.target_metrics["target_returns"] and st.session_state.current_step > 0:
            feedback["message"] = f"INFO: Below target returns of {self.target_metrics['target_returns']*100:.2f}% (current total profit: ${total_profit_since_init:,.2f})."
            feedback["status"] = "info"
        else:
            feedback["status"] = "success"

        st.session_state.critic_feedback_history.append(feedback)
        return feedback

# --- Main app.py content starts here ---
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Agentic AI Systems in Finance")
st.divider()

# Initialize session state for all pages
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.current_page = "Introduction" # Default page
    # Scenario Builder Defaults
    st.session_state.objective = "Maximize returns"
    st.session_state.risk_tolerance = "moderate"
    st.session_state.initial_cash = 100000.0
    st.session_state.stock_symbols = ["AAPL", "GOOG"]
    st.session_state.initial_prices_input = {"AAPL": 150.0, "GOOG": 120.0} # For input widgets
    st.session_state.volatility = 0.01
    st.session_state.target_returns = 0.05
    st.session_state.max_risk = 0.02
    st.session_state.risk_aversion = 0.5
    st.session_state.mis_specify_goal = False
    st.session_state.inject_fault_at_step = 5
    st.session_state.fault_type = "Market Anomaly"

    # Simulation State
    st.session_state.environment_state = None
    st.session_state.portfolio = {"cash": st.session_state.initial_cash, "holdings": {}}
    st.session_state.portfolio_history = [] # For plotting [{'step': 0, 'value': X, 'returns': Y}]
    st.session_state.trade_log = []
    st.session_state.current_step = 0
    st.session_state.planner_agent = None
    st.session_state.executor_agent = None
    st.session_state.critic_agent = None
    st.session_state.planner_plan = []
    st.session_state.executor_action = {}
    st.session_state.critic_feedback = {}
    st.session_state.previous_portfolio_value = st.session_state.initial_cash # For critic
    st.session_state.current_objective_display = st.session_state.objective # To display effective objective in simulation

    # Add initial portfolio value to history
    st.session_state.portfolio_history.append({"step": 0, "value": st.session_state.initial_cash, "returns": 0.0})


# Your code starts here
st.markdown("""
In this lab, we will explore **Agentic AI Systems** and their application in finance. We will delve into their architectural patterns, understand how they make decisions, manage risks, and the crucial role of human oversight. This interactive application will allow you to configure and simulate investment scenarios, visualize agent behavior, and identify emergent risks in a synthetic market environment.

We will focus on understanding:
*   The fundamental concepts and characteristics of agentic AI systems.
*   Key architectural patterns like the Planner-Executor-Critic (P-E-C) loop and ReAct chains.
*   How to configure custom investment scenarios and observe AI agent behavior.
*   Emergent risks such as goal mis-specification, autonomy creep, and cascading error propagation.
*   The importance of robust design and human-in-the-loop interventions.
*   The conceptual mathematical foundations for agentic systems.
""")

page = st.sidebar.selectbox(
    label="Navigation",
    options=["Introduction", "Scenario Builder", "Simulation & Results", "Risk Analysis"],
    index=["Introduction", "Scenario Builder", "Simulation & Results", "Risk Analysis"].index(st.session_state.get("current_page", "Introduction"))
)
st.session_state.current_page = page

if page == "Introduction":
    from application_pages.introduction import main
    main()
elif page == "Scenario Builder":
    from application_pages.scenario_builder import main
    main(initialize_investment_environment, PlannerAgent, ExecutorAgent, CriticAgent)
elif page == "Simulation & Results":
    from application_pages.simulation_results import main
    main(simulate_market_movement, execute_trade, calculate_portfolio_value, calculate_reward)
elif page == "Risk Analysis":
    from application_pages.risk_analysis import main
    main()
# Your code ends here
