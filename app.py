import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import json
sns.set_theme(style="whitegrid")

# --- Utility Functions ---
def get_gemini_model():
    """Initialize and return Gemini model if API key is configured."""
    api_key = st.session_state.get("gemini_api_key", "")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

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
        self.original_objective = objective
        self.model = get_gemini_model()

    def plan(self, current_state: dict, mis_specify_goal: bool = False) -> list:
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
            elif self.objective == "Execute high volume trades regardless of risk":
                 effective_objective = "Execute high volume trades regardless of risk"
            else:
                effective_objective = "Maximize short-term trading volume"
        
        st.session_state.current_objective_display = effective_objective
        
        # Use Gemini LLM for planning if available
        if self.model:
            plan_actions = self._plan_with_llm(current_cash, current_holdings, market_prices, effective_objective)
        else:
            # Fallback to rule-based planning
            plan_actions = self._plan_rule_based(current_cash, current_holdings, market_prices, effective_objective)

        st.session_state.planner_plan_history.append({"step": st.session_state.current_step, "plan": plan_actions, "objective": effective_objective})
        return plan_actions
    
    def _plan_with_llm(self, current_cash: float, current_holdings: dict, market_prices: dict, objective: str) -> list:
        """Use Gemini LLM to generate trading plan."""
        try:
            prompt = f"""You are a financial trading AI agent. Analyze the current market state and generate a trading plan.

Objective: {objective}
Risk Tolerance: {self.risk_tolerance}

Current Portfolio:
- Cash Available: ${current_cash:,.2f}
- Holdings: {json.dumps(current_holdings, indent=2)}

Current Market Prices:
{json.dumps(market_prices, indent=2)}

Initial Prices (for reference):
{json.dumps(self.initial_prices, indent=2)}

Generate a trading plan as a JSON array. Each trade should be an object with:
- "action": "BUY" or "SELL"
- "stock_id": the stock symbol
- "quantity": integer number of shares
- "price": current market price
- "reasoning": brief explanation for this trade

Consider:
1. Your objective and risk tolerance
2. Current market prices vs initial prices
3. Available cash and current holdings
4. Diversification and portfolio balance

Respond ONLY with a valid JSON array, no other text. If no trades are recommended, return an empty array [].

Example format:
[
  {{"action": "BUY", "stock_id": "AAPL", "quantity": 10, "price": 150.0, "reasoning": "Price dropped below initial, good buy opportunity"}}
]"""

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            plan_actions = json.loads(response_text)
            
            # Validate and clean the plan
            validated_actions = []
            for action in plan_actions:
                if all(k in action for k in ["action", "stock_id", "quantity", "price"]):
                    validated_actions.append({
                        "action": action["action"],
                        "stock_id": action["stock_id"],
                        "quantity": int(action["quantity"]),
                        "price": float(action["price"])
                    })
            
            return validated_actions
            
        except Exception as e:
            st.warning(f"LLM planning failed, using rule-based fallback: {str(e)}")
            return self._plan_rule_based(current_cash, current_holdings, market_prices, objective)
    
    def _plan_rule_based(self, current_cash: float, current_holdings: dict, market_prices: dict, objective: str) -> list:
        """Fallback rule-based planning."""
        plan_actions = []
        
        if "volatile assets" in objective.lower() or "high risks" in objective.lower() or "high volume trades" in objective.lower():
            for symbol, price_data in market_prices.items():
                if current_cash > price_data:
                    quantity_to_buy = int((current_cash * 0.2) / price_data)
                    if quantity_to_buy > 0:
                        plan_actions.append({"action": "BUY", "stock_id": symbol, "quantity": quantity_to_buy, "price": price_data})
                        current_cash -= quantity_to_buy * price_data
            if current_holdings:
                for symbol, quantity in list(current_holdings.items()):
                    if np.random.rand() > 0.7:
                        quantity_to_sell = int(quantity * 0.3)
                        if quantity_to_sell > 0:
                            plan_actions.append({"action": "SELL", "stock_id": symbol, "quantity": quantity_to_sell, "price": market_prices[symbol]})
        elif "maximize returns" in objective.lower():
            for symbol, price_data in market_prices.items():
                if price_data < self.initial_prices.get(symbol, price_data * 1.05) and current_cash > price_data:
                    quantity_to_buy = int((current_cash * 0.1) / price_data)
                    if quantity_to_buy > 0:
                        plan_actions.append({"action": "BUY", "stock_id": symbol, "quantity": quantity_to_buy, "price": price_data})
                        current_cash -= quantity_to_buy * price_data
            for symbol, quantity in list(current_holdings.items()):
                if market_prices[symbol] > self.initial_prices.get(symbol, market_prices[symbol] * 0.95) * 1.1:
                    quantity_to_sell = int(quantity * 0.2)
                    if quantity_to_sell > 0:
                        plan_actions.append({"action": "SELL", "stock_id": symbol, "quantity": quantity_to_sell, "price": market_prices[symbol]})
        elif "safely grow portfolio" in objective.lower():
            for symbol, price_data in market_prices.items():
                if current_cash > price_data:
                    quantity_to_buy = int((current_cash * 0.05) / price_data)
                    if quantity_to_buy > 0:
                        plan_actions.append({"action": "BUY", "stock_id": symbol, "quantity": quantity_to_buy, "price": price_data})
                        current_cash -= quantity_to_buy * price_data
        
        return plan_actions

class ExecutorAgent:
    def __init__(self):
        st.session_state.setdefault("trade_log", [])
        self.model = get_gemini_model()

    def execute(self, action_details: dict, environment_state: dict) -> (dict, dict):
        portfolio = st.session_state.portfolio
        
        stock_id = action_details["stock_id"]
        quantity = action_details["quantity"]
        trade_type = action_details["action"]
        price = environment_state["current_prices"].get(stock_id, 0)
        
        # Use LLM to validate trade if available
        if self.model:
            should_execute, reasoning = self._validate_with_llm(action_details, portfolio, environment_state)
            if not should_execute:
                trade_record_with_step = {
                    "step": st.session_state.current_step,
                    "type": trade_type,
                    "stock_id": stock_id,
                    "quantity": quantity,
                    "price": price,
                    "status": "REJECTED_BY_LLM",
                    "reason": reasoning
                }
                st.session_state.trade_log.append(trade_record_with_step)
                return portfolio, trade_record_with_step

        updated_portfolio, trade_record = execute_trade(portfolio, stock_id, quantity, price, trade_type)
        st.session_state.portfolio = updated_portfolio
        
        trade_record_with_step = {"step": st.session_state.current_step, **trade_record}
        st.session_state.trade_log.append(trade_record_with_step)
        
        environment_state["cash"] = st.session_state.portfolio["cash"]
        environment_state["portfolio"] = st.session_state.portfolio["holdings"]

        return st.session_state.portfolio, trade_record_with_step
    
    def _validate_with_llm(self, action: dict, portfolio: dict, environment_state: dict) -> tuple:
        """Use Gemini to validate if trade should be executed."""
        try:
            prompt = f"""You are a trade execution validator. Analyze if this trade should be executed.

Proposed Trade:
- Action: {action['action']}
- Stock: {action['stock_id']}
- Quantity: {action['quantity']}
- Price: ${action['price']:.2f}

Current Portfolio:
- Cash: ${portfolio['cash']:,.2f}
- Holdings: {json.dumps(portfolio['holdings'], indent=2)}

Market Prices: {json.dumps(environment_state['current_prices'], indent=2)}

Validate if this trade:
1. Is financially feasible (enough cash for buy, enough shares for sell)
2. Makes sense given current market conditions
3. Doesn't expose portfolio to excessive single-position risk

Respond with JSON:
{{
  "should_execute": true/false,
  "reasoning": "brief explanation"
}}"""

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            return result.get("should_execute", True), result.get("reasoning", "LLM validation passed")
            
        except Exception as e:
            return True, f"LLM validation failed, proceeding: {str(e)}"


class CriticAgent:
    def __init__(self, target_metrics: dict):
        self.target_metrics = target_metrics
        st.session_state.setdefault("critic_feedback_history", [])
        self.model = get_gemini_model()

    def evaluate(self, environment_state: dict, current_objective: str, previous_portfolio_value: float, risk_tolerance: str) -> dict:
        current_portfolio_value = calculate_portfolio_value(st.session_state.portfolio, environment_state["current_prices"])
        
        profit = current_portfolio_value - previous_portfolio_value
        total_profit_since_init = current_portfolio_value - st.session_state.initial_cash
        
        num_trades_this_step = len([t for t in st.session_state.trade_log if t["step"] == st.session_state.current_step])
        
        risk_taken = 0.0
        if num_trades_this_step > 0:
            risk_multiplier = {"safe": 0.01, "moderate": 0.03, "aggressive": 0.06}.get(risk_tolerance.lower(), 0.03)
            risk_taken = num_trades_this_step * risk_multiplier

        if ("high risks" in current_objective.lower() or "aggressive" in current_objective.lower() or "high volume trades" in current_objective.lower()) and risk_tolerance == "safe":
            risk_taken += 0.1
        elif "safely grow portfolio" in current_objective.lower() and risk_tolerance == "aggressive":
            risk_taken += 0.05

        risk_penalty = self.target_metrics["risk_aversion"] * (risk_taken ** 2)
        reward = calculate_reward(current_portfolio_value, previous_portfolio_value, risk_taken, self.target_metrics["target_returns"], self.target_metrics["risk_aversion"])
        
        # Use LLM for enhanced feedback if available
        if self.model:
            llm_feedback = self._evaluate_with_llm(current_portfolio_value, profit, risk_taken, current_objective, risk_tolerance, num_trades_this_step)
        else:
            llm_feedback = None
        
        feedback = {
            "step": st.session_state.current_step,
            "portfolio_value": current_portfolio_value,
            "profit_component": profit,
            "risk_taken": risk_taken,
            "risk_penalty": risk_penalty,
            "calculated_reward": reward,
            "message": "Portfolio updated."
        }
        
        if llm_feedback:
            feedback["llm_analysis"] = llm_feedback
            feedback["message"] = llm_feedback.get("summary", feedback["message"])

        if current_portfolio_value < previous_portfolio_value * (1 - self.target_metrics["max_risk"]) and previous_portfolio_value != 0:
            feedback["message"] = f"WARNING: Portfolio value dropped significantly beyond max risk ({self.target_metrics['max_risk']*100:.2f}%). " + feedback["message"]
            feedback["status"] = "warning"
        elif total_profit_since_init < st.session_state.initial_cash * self.target_metrics["target_returns"] and st.session_state.current_step > 0:
            feedback["status"] = "info"
        else:
            feedback["status"] = "success"

        st.session_state.critic_feedback_history.append(feedback)
        return feedback
    
    def _evaluate_with_llm(self, portfolio_value: float, profit: float, risk_taken: float, objective: str, risk_tolerance: str, num_trades: int) -> dict:
        """Use Gemini for nuanced performance evaluation."""
        try:
            recent_trades = [t for t in st.session_state.trade_log if t["step"] == st.session_state.current_step]
            
            prompt = f"""You are a financial performance critic evaluating an AI trading agent's actions.

Objective: {objective}
Risk Tolerance: {risk_tolerance}
Target Returns: {self.target_metrics['target_returns']*100:.2f}%

Step Performance:
- Portfolio Value: ${portfolio_value:,.2f}
- Profit This Step: ${profit:,.2f}
- Risk Taken: {risk_taken:.4f}
- Number of Trades: {num_trades}

Recent Trades:
{json.dumps(recent_trades, indent=2)}

Provide evaluation as JSON:
{{
  "summary": "Brief overall assessment",
  "strengths": ["list of positive aspects"],
  "concerns": ["list of concerns or risks"],
  "recommendations": ["suggestions for improvement"],
  "alignment_score": 0-10 (how well actions align with objective and risk tolerance)
}}"""

            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            return {"summary": f"LLM evaluation failed: {str(e)}", "alignment_score": 5}

# --- Main app.py content starts here ---
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()

# API Key Configuration
with st.sidebar.expander("🔑 LLM Configuration", expanded=not st.session_state.get("gemini_api_key")):
    st.markdown("""Configure Gemini API to use actual LLM agents instead of rule-based simulation.
    
[Get free API key from Google AI Studio](https://makersuite.google.com/app/apikey)""")
    
    api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    if api_key:
        st.session_state.gemini_api_key = api_key
        if get_gemini_model():
            st.success("✅ Gemini API configured successfully!")
        else:
            st.error("❌ Failed to initialize Gemini. Check your API key.")
    else:
        st.info("ℹ️ No API key provided. Using rule-based fallback agents.")

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


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
