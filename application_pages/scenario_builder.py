import streamlit as st
import numpy as np # For random initial prices, if needed

def main(initialize_investment_environment_func, PlannerAgent_cls, ExecutorAgent_cls, CriticAgent_cls):
    st.header("Section 3: Setting Up the Simulated Investment Environment")
    st.markdown("""
    ### Creating a Synthetic Investment Portfolio Environment

    This section allows you to configure a synthetic market environment and define the objectives and constraints for your agentic AI system.

    **Key Features:**
    *   Define the agent's primary objective and risk tolerance.
    *   Set up initial market parameters like cash, stock symbols, and their starting prices.
    *   Introduce volatility to simulate market fluctuations.
    *   Configure the Critic's target metrics for evaluating performance.
    *   Optionally inject faults to observe emergent risks.
    """)

    st.subheader("Agent Objective & Risk Tolerance")
    st.session_state.objective = st.text_input(
        "Agent Objective",
        value=st.session_state.get("objective", "Maximize returns"),
        help="Define the primary goal for your AI investment agent (e.g., 'Maximize returns', 'Safely grow portfolio')."
    )
    st.session_state.risk_tolerance = st.radio(
        "Risk Tolerance",
        options=["safe", "moderate", "aggressive"],
        index=["safe", "moderate", "aggressive"].index(st.session_state.get("risk_tolerance", "moderate")),
        horizontal=True,
        help="Set the risk appetite for the agent. 'Safe' aims for minimal volatility, 'aggressive' for high growth with higher risk."
    )

    st.subheader("Environment Parameters")
    st.session_state.initial_cash = st.number_input(
        "Initial Cash ($")",
        min_value=1000.0,
        value=st.session_state.get("initial_cash", 100000.0),
        step=1000.0,
        help="Starting cash available for the agent's portfolio."
    )
    
    available_symbols = ["AAPL", "GOOG", "MSFT", "TSLA", "AMZN"]
    st.session_state.stock_symbols = st.multiselect(
        "Stock Symbols",
        options=available_symbols,
        default=st.session_state.get("stock_symbols", ["AAPL", "GOOG"]),
        help="Select the stock symbols available in the simulated market."
    )

    st.markdown("#### Initial Stock Prices")
    col_prices = st.columns(len(st.session_state.stock_symbols) if st.session_state.stock_symbols else 1)
    for i, symbol in enumerate(st.session_state.stock_symbols):
        with col_prices[i]:
            # Ensure session_state.initial_prices_input is initialized for new symbols
            if symbol not in st.session_state.initial_prices_input:
                st.session_state.initial_prices_input[symbol] = round(np.random.uniform(50, 200), 2)
            
            st.session_state.initial_prices_input[symbol] = st.number_input(
                f"Initial Price for {symbol} ($")",
                min_value=1.0,
                value=st.session_state.initial_prices_input[symbol],
                step=1.0,
                key=f"initial_price_{symbol}",
                help=f"Set the starting price for {symbol}."
            )

    st.session_state.volatility = st.number_input(
        "Market Volatility (e.g., 0.01 for 1%)",
        min_value=0.001,
        max_value=0.1,
        value=st.session_state.get("volatility", 0.01),
        step=0.001,
        format="%.3f",
        help="Determines the magnitude of random price fluctuations in the simulated market."
    )

    st.subheader("Critic Target Metrics")
    st.session_state.target_returns = st.number_input(
        "Target Returns (e.g., 0.05 for 5%)",
        min_value=0.0,
        max_value=0.5,
        value=st.session_state.get("target_returns", 0.05),
        step=0.01,
        format="%.2f",
        help="The target percentage return the Critic aims for the portfolio to achieve."
    )
    st.session_state.max_risk = st.number_input(
        "Max Permissible Risk (e.g., 0.02 for 2%)",
        min_value=0.0,
        max_value=0.1,
        value=st.session_state.get("max_risk", 0.02),
        step=0.005,
        format="%.3f",
        help="The maximum percentage drop in portfolio value the Critic tolerates before flagging a warning."
    )
    st.session_state.risk_aversion = st.number_input(
        "Risk Aversion Factor",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state.get("risk_aversion", 0.5),
        step=0.1,
        format="%.1f",
        help="A coefficient used by the Critic to penalize risk. Higher values mean stronger penalties for riskier actions."
    )

    st.subheader("Fault Injection (to observe emergent risks)")
    st.session_state.mis_specify_goal = st.checkbox(
        "Mis-specify Goal",
        value=st.session_state.get("mis_specify_goal", False),
        help="Activate to intentionally introduce a misalignment between the agent's objective and actual desired outcome, demonstrating 'Goal Mis-specification'."
    )
    st.session_state.inject_fault_at_step = st.number_input(
        "Inject Fault at Step",
        min_value=1,
        value=st.session_state.get("inject_fault_at_step", 5),
        step=1,
        help="Specify the simulation step at which to inject a fault into the environment or agent's perception."
    )
    st.session_state.fault_type = st.selectbox(
        "Fault Type",
        options=["Market Anomaly", "Agent Misperception"],
        index=["Market Anomaly", "Agent Misperception"].index(st.session_state.get("fault_type", "Market Anomaly")),
        help="Choose the type of fault to inject: 'Market Anomaly' (e.g., sudden price drop) or 'Agent Misperception' (agent misinterprets data)."
    )

    st.markdown("---")
    
    def initialize_simulation_callback():
        st.session_state.environment_state = initialize_investment_environment_func(
            st.session_state.initial_cash,
            st.session_state.stock_symbols,
            st.session_state.initial_prices_input,
            st.session_state.volatility
        )
        st.session_state.portfolio = {
            "cash": st.session_state.initial_cash,
            "holdings": {symbol: 0 for symbol in st.session_state.stock_symbols}
        }
        st.session_state.previous_portfolio_value = st.session_state.initial_cash
        st.session_state.portfolio_history = []
        st.session_state.trade_log = []
        st.session_state.critic_feedback_history = []
        st.session_state.planner_plan_history = []
        st.session_state.current_step = 0
        
        # Add initial portfolio value to history
        st.session_state.portfolio_history.append({"step": 0, "value": st.session_state.initial_cash, "returns": 0.0})

        
        # Instantiate agents
        st.session_state.planner_agent = PlannerAgent_cls(
            st.session_state.objective,
            st.session_state.risk_tolerance,
            st.session_state.initial_prices_input # Pass initial prices to planner
        )
        st.session_state.executor_agent = ExecutorAgent_cls()
        st.session_state.critic_agent = CriticAgent_cls(
            {"target_returns": st.session_state.target_returns,
             "max_risk": st.session_state.max_risk,
             "risk_aversion": st.session_state.risk_aversion}
        )
        
        st.success("Investment environment and agents initialized!")
        
    st.button("Initialize Environment", on_click=initialize_simulation_callback, help="Click to set up the market environment and prepare the AI agents for simulation.")

    if st.session_state.environment_state:
        st.markdown("#### Initial Environment State (after initialization):")
        st.json(st.session_state.environment_state)
