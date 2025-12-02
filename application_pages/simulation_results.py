import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # For random in planner

def main(simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func):
    st.header("Section 4: Simulation & Results")
    st.markdown("""
    This section allows you to run the agentic AI simulation, observe its behavior, and analyze the results.
    """)

    if not st.session_state.get("environment_state"):
        st.warning("Please configure the scenario and initialize the environment on the 'Scenario Builder' page first.")
        return

    st.subheader("Simulation Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("Run Single Step", on_click=lambda: run_simulation_step(simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func), help="Advance the simulation by one step.")
    with col2:
        st.button("Run Full Simulation (10 Steps)", on_click=lambda: run_full_simulation(10, simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func), help="Run the simulation for a predefined number of steps (e.g., 10-20 steps).")
    with col3:
        st.button("Reset Simulation", on_click=reset_simulation, help="Reset the simulation to its initial state.")

    st.markdown("---")

    st.subheader("P-E-C Loop Visualization")
    st.markdown(r"""
    During single-step runs, the currently active component of the P-E-C loop (Planner, Executor, or Critic) would be conceptually highlighted here.
    The Planner uses Decision Tree Logic: $$ \text{If Critique Result} = \text{Negative}, \text{then Plan Revision occurs} $$
    """)

    st.subheader("ReAct (Reasoning and Acting) Chains Visualization")
    st.markdown("""
    This flow depicts interleaved 'Thought' and 'Action' steps. In a live simulation, real-time highlighting would show the agent's current focus.
    """)
    st.markdown("---")
    
    st.subheader("Current Portfolio Metrics")
    if st.session_state.portfolio:
        current_prices = st.session_state.environment_state["current_prices"]
        current_portfolio_value = calculate_portfolio_value_func(st.session_state.portfolio, current_prices)
        total_profit = current_portfolio_value - st.session_state.initial_cash

        col_metrics_1, col_metrics_2, col_metrics_3, col_metrics_4 = st.columns(4)
        with col_metrics_1:
            st.metric("Current Portfolio Value", f"${current_portfolio_value:,.2f}")
        with col_metrics_2:
            st.metric("Cash Balance", f"${st.session_state.portfolio['cash']:,.2f}")
        with col_metrics_3:
            st.metric("Total Profit", f"${total_profit:,.2f}")
        with col_metrics_4:
            if st.session_state.critic_feedback_history:
                latest_reward = st.session_state.critic_feedback_history[-1].get("calculated_reward", 0)
                st.metric("Last Step Reward", f"{latest_reward:.4f}")
            else:
                st.metric("Last Step Reward", "N/A")
        
        st.markdown("#### Individual Stock Holdings")
        holdings_data = [{"Stock ID": s_id, "Quantity": qty, "Current Price": f"${current_prices.get(s_id, 0):,.2f}", "Value": f"${qty * current_prices.get(s_id, 0):,.2f}"}
                         for s_id, qty in st.session_state.portfolio["holdings"].items()]
        if holdings_data:
            st.dataframe(pd.DataFrame(holdings_data), width='stretch')
        else:
            st.info("No stock holdings yet.")
    else:
        st.info("Portfolio not initialized.")

    st.markdown("---")

    st.subheader("Portfolio Evolution Plot")
    if st.session_state.portfolio_history:
        df_history = pd.DataFrame(st.session_state.portfolio_history)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_history["step"], df_history["value"], label="Portfolio Value", marker='o')
        ax.set_title("Portfolio Value Over Simulation Steps")
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Returns plot
        fig_returns, ax_returns = plt.subplots(figsize=(10, 5))
        ax_returns.plot(df_history["step"], df_history["returns"] * 100, label="Returns (%)", color='green', marker='x')
        ax_returns.set_title("Portfolio Returns Over Simulation Steps")
        ax_returns.set_xlabel("Simulation Step")
        ax_returns.set_ylabel("Returns (%)")
        ax_returns.legend()
        ax_returns.grid(True)
        st.pyplot(fig_returns)
    else:
        st.info("Run the simulation to see portfolio evolution.")
        
    st.markdown("---")

    st.subheader("Reward Function Breakdown (Last Step)")
    if st.session_state.critic_feedback_history:
        latest_feedback = st.session_state.critic_feedback_history[-1]
        st.markdown(f"""
        **Step:** {latest_feedback["step"]}
        *   **Profit Component:** ${latest_feedback["profit_component"]:.2f}
        *   **Risk Taken (Normalized):** {latest_feedback["risk_taken"]:.4f}
        *   **Risk Penalty:** {latest_feedback["risk_penalty"]:.4f}
        *   **Calculated Reward:** {latest_feedback["calculated_reward"]:.4f}
        """)
        if latest_feedback.get("status") == "warning":
            st.warning(latest_feedback["message"])
        elif latest_feedback.get("status") == "info":
            st.info(latest_feedback["message"])
        else:
            st.success(latest_feedback["message"])
    else:
        st.info("Run the simulation to see reward breakdown.")

    st.markdown("---")

    st.subheader("Trade Log")
    if st.session_state.trade_log:
        df_trades = pd.DataFrame(st.session_state.trade_log)
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("No trades executed yet.")
        
    st.markdown("---")

    st.subheader("Agent Output Displays")
    st.markdown("#### Planner's Generated Plan (Last Step)")
    if st.session_state.planner_plan_history:
        latest_plan_entry = st.session_state.planner_plan_history[-1]
        st.json(latest_plan_entry["plan"])
        st.info(f"Planner's objective for this step: **{latest_plan_entry['objective']}**")
    else:
        st.info("No plan generated yet.")

    st.markdown("#### Executor's Action Details (Last Trades)")
    if st.session_state.trade_log:
        last_step_trades = [t for t in st.session_state.trade_log if t["step"] == st.session_state.current_step]
        if last_step_trades:
            st.json(last_step_trades)
        else:
            st.info("No trades by Executor in the last step.")
    else:
        st.info("No trades executed yet.")

    st.markdown("#### Critic's Feedback (Last Step)")
    if st.session_state.critic_feedback_history:
        latest_feedback = st.session_state.critic_feedback_history[-1]
        st.json(latest_feedback)
    else:
        st.info("No critic feedback yet.")

# --- Simulation Logic Functions ---
def run_simulation_step(simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func):
    if not st.session_state.get("environment_state") or not st.session_state.get("planner_agent"):
        st.error("Environment or agents not initialized. Please go to 'Scenario Builder' and initialize.")
        return

    st.session_state.current_step += 1
    st.info(f"Running simulation step {st.session_state.current_step}...")

    current_env = st.session_state.environment_state
    current_portfolio = st.session_state.portfolio
    current_prices = current_env["current_prices"]
    
    previous_portfolio_value = calculate_portfolio_value_func(current_portfolio, current_prices)
    st.session_state.previous_portfolio_value = previous_portfolio_value # Store for critic in next step
    
    # --- Fault Injection Logic ---
    if st.session_state.mis_specify_goal and st.session_state.current_step == st.session_state.inject_fault_at_step:
        st.warning(f"FAULT INJECTED at step {st.session_state.current_step}: Type - {st.session_state.fault_type}")
        if st.session_state.fault_type == "Market Anomaly":
            # Drastically drop prices for a symbol
            if st.session_state.stock_symbols:
                fault_symbol = st.session_state.stock_symbols[0]
                current_env["market_data"][fault_symbol]["price"] *= 0.5 # 50% drop
                current_env["current_prices"][fault_symbol] *= 0.5
                # If planner uses initial_prices for baseline, update that too
                if fault_symbol in st.session_state.planner_agent.initial_prices:
                    st.session_state.planner_agent.initial_prices[fault_symbol] *= 0.5 # Affect planner's baseline
                st.warning(f"Market Anomaly: Price of {fault_symbol} dropped by 50%!")
        elif st.session_state.fault_type == "Agent Misperception":
            # Planner might misinterpret a signal or act against risk tolerance
            st.session_state.planner_agent.objective = "Execute high volume trades regardless of risk" # Overrides initial obj
            st.warning(f"Agent Misperception: Planner's objective changed to '{st.session_state.planner_agent.objective}'!")

    # 1. Simulate Market Movement
    new_market_data = simulate_market_movement_func(current_env["market_data"], current_env["volatility_factor"])
    current_env["market_data"] = new_market_data
    current_env["current_prices"] = {symbol: data["price"] for symbol, data in new_market_data.items()}

    # 2. Planner plans
    planner_plan = st.session_state.planner_agent.plan(
        {"cash": current_portfolio["cash"],
         "portfolio": current_portfolio["holdings"],
         "current_prices": current_env["current_prices"]},
        st.session_state.mis_specify_goal and (st.session_state.current_step >= st.session_state.inject_fault_at_step)
    )
    st.session_state.planner_plan = planner_plan

    # 3. Executor executes
    executed_trades = []
    for action in planner_plan:
        updated_portfolio, trade_record = execute_trade_func(st.session_state.portfolio, action["stock_id"], action["quantity"], action["price"], action["action"])
        executed_trades.append(trade_record)
        
    st.session_state.portfolio = updated_portfolio # Update the global portfolio state
    st.session_state.executor_action = executed_trades # Store the last executed actions

    # 4. Critic evaluates
    critic_feedback = st.session_state.critic_agent.evaluate(
        current_env,
        st.session_state.get("current_objective_display", st.session_state.objective),
        previous_portfolio_value, # Use value before market movement/trades for profit calc
        st.session_state.risk_tolerance # Pass risk tolerance for more nuanced feedback
    )
    st.session_state.critic_feedback = critic_feedback
    
    # Update portfolio history
    current_portfolio_value_after_step = calculate_portfolio_value_func(st.session_state.portfolio, current_env["current_prices"])
    returns_this_step = (current_portfolio_value_after_step - st.session_state.portfolio_history[-1]["value"]) / st.session_state.portfolio_history[-1]["value"] if st.session_state.portfolio_history and st.session_state.portfolio_history[-1]["value"] != 0 else 0
    st.session_state.portfolio_history.append({
        "step": st.session_state.current_step,
        "value": current_portfolio_value_after_step,
        "returns": returns_this_step
    })
    
    st.rerun() # Rerun to update the display


@st.fragment
def run_full_simulation(num_steps: int, simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func):
    if not st.session_state.get("environment_state") or not st.session_state.get("planner_agent"):
        st.error("Environment or agents not initialized. Please go to 'Scenario Builder' and initialize.")
        return

    st.info(f"Running full simulation for {num_steps} steps...")
    progress_bar = st.progress(0)

    for i in range(num_steps):
        run_simulation_step_silent(simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func)
        progress_bar.progress((i + 1) / num_steps)
    
    st.success("Full simulation completed!")
    st.rerun() # Rerun to update all displays


def run_simulation_step_silent(simulate_market_movement_func, execute_trade_func, calculate_portfolio_value_func, calculate_reward_func):
    # This is a duplicate of run_simulation_step but without `st.info` and `st.rerun()` to avoid spamming UI during full run
    st.session_state.current_step += 1

    current_env = st.session_state.environment_state
    current_portfolio = st.session_state.portfolio
    current_prices = current_env["current_prices"]
    
    previous_portfolio_value = calculate_portfolio_value_func(current_portfolio, current_prices)
    st.session_state.previous_portfolio_value = previous_portfolio_value # Store for critic in next step
    
    # Fault Injection Logic
    if st.session_state.mis_specify_goal and st.session_state.current_step == st.session_state.inject_fault_at_step:
        if st.session_state.fault_type == "Market Anomaly":
            if st.session_state.stock_symbols:
                fault_symbol = st.session_state.stock_symbols[0]
                current_env["market_data"][fault_symbol]["price"] *= 0.5
                current_env["current_prices"][fault_symbol] *= 0.5
                if fault_symbol in st.session_state.planner_agent.initial_prices:
                    st.session_state.planner_agent.initial_prices[fault_symbol] *= 0.5
        elif st.session_state.fault_type == "Agent Misperception":
            st.session_state.planner_agent.objective = "Execute high volume trades regardless of risk"

    # 1. Simulate Market Movement
    new_market_data = simulate_market_movement_func(current_env["market_data"], current_env["volatility_factor"])
    current_env["market_data"] = new_market_data
    current_env["current_prices"] = {symbol: data["price"] for symbol, data in new_market_data.items()}

    # 2. Planner plans
    planner_plan = st.session_state.planner_agent.plan(
        {"cash": current_portfolio["cash"],
         "portfolio": current_portfolio["holdings"],
         "current_prices": current_env["current_prices"]},
        st.session_state.mis_specify_goal and (st.session_state.current_step >= st.session_state.inject_fault_at_step)
    )
    st.session_state.planner_plan = planner_plan

    # 3. Executor executes
    executed_trades = []
    updated_portfolio = st.session_state.portfolio
    for action in planner_plan:
        updated_portfolio, trade_record = execute_trade_func(updated_portfolio, action["stock_id"], action["quantity"], action["price"], action["action"])
        executed_trades.append(trade_record)
    st.session_state.portfolio = updated_portfolio
    st.session_state.executor_action = executed_trades

    # 4. Critic evaluates
    critic_feedback = st.session_state.critic_agent.evaluate(
        current_env,
        st.session_state.get("current_objective_display", st.session_state.objective),
        previous_portfolio_value,
        st.session_state.risk_tolerance
    )
    st.session_state.critic_feedback = critic_feedback
    
    # Update portfolio history
    current_portfolio_value_after_step = calculate_portfolio_value_func(st.session_state.portfolio, current_env["current_prices"])
    returns_this_step = (current_portfolio_value_after_step - st.session_state.portfolio_history[-1]["value"]) / st.session_state.portfolio_history[-1]["value"] if st.session_state.portfolio_history and st.session_state.portfolio_history[-1]["value"] != 0 else 0
    st.session_state.portfolio_history.append({
        "step": st.session_state.current_step,
        "value": current_portfolio_value_after_step,
        "returns": returns_this_step
    })


def reset_simulation():
    # Reset all simulation-specific session state variables
    st.session_state.environment_state = None
    st.session_state.portfolio = {"cash": st.session_state.initial_cash, "holdings": {}}
    st.session_state.portfolio_history = []
    st.session_state.trade_log = []
    st.session_state.current_step = 0
    st.session_state.planner_agent = None
    st.session_state.executor_agent = None
    st.session_state.critic_agent = None
    st.session_state.planner_plan = []
    st.session_state.executor_action = {}
    st.session_state.critic_feedback = {}
    st.session_state.previous_portfolio_value = st.session_state.initial_cash
    st.session_state.current_objective_display = st.session_state.objective # Reset this as well
    # Re-add initial portfolio value to history after reset
    st.session_state.portfolio_history.append({"step": 0, "value": st.session_state.initial_cash, "returns": 0.0})

    st.success("Simulation reset to initial state.")
    st.rerun()
