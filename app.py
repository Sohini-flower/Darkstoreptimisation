
import streamlit as st
import pandas as pd
import numpy as np
import folium
import altair as alt
from streamlit_folium import st_folium
from ortools.sat.python import cp_model

st.set_page_config(page_title="Chennai Dark Store Optimization", layout="wide")
st.title("Chennai Dark Store Location Optimization")

st.sidebar.header("Model Inputs")
n_customers = st.sidebar.number_input("Number of customers", 100, 10000, 10000, step=500)
n_candidates = st.sidebar.number_input("Candidate store locations", 5, 50, 50, step=5)
max_distance = st.sidebar.slider("Maximum delivery distance (km)", 1.0, 15.0, 5.0, 0.5)
revenue = st.sidebar.number_input("Revenue per order (₹)", 1.0, 5000.0, 250.0)
variable_cost = st.sidebar.number_input("Variable cost per order (₹)", 0.0, 5000.0, 170.0)
fixed_cost = st.sidebar.number_input("Fixed cost per store per day (₹)", 0.0, 100000.0, 5000.0)
capacity = st.sidebar.number_input("Daily store capacity", 50, 5000, 1000, step=50)
distance_cost = st.sidebar.number_input("Additional cost per km per order (₹)", 0.0, 500.0, 5.0)
time_limit = st.sidebar.slider("Solver time limit per scenario (seconds)", 5, 120, 30, 5)

SCENARIOS = [2, 5, 10, 20, 25]
CENTER = [13.0827, 80.2707]


def create_data(n, m, seed=42):
    rng = np.random.default_rng(seed)

    # Approximate Chennai-area simulation boundaries.
    lat_c = np.clip(
        rng.normal(13.075, 0.045, n), 12.90, 13.15
    )
    lon_c = np.clip(
        rng.normal(80.245, 0.045, n), 80.15, 80.32
    )

    lat_s = rng.uniform(12.90, 13.15, m)
    lon_s = rng.uniform(80.15, 80.32, m)

    customers = pd.DataFrame({
        "customer_id": [f"C{i + 1}" for i in range(n)],
        "latitude": lat_c,
        "longitude": lon_c,
    })

    stores = pd.DataFrame({
        "store_id": [f"S{i + 1}" for i in range(m)],
        "latitude": lat_s,
        "longitude": lon_s,
    })

    return customers, stores


def calculate_distances(customers, stores):
    lat1 = np.radians(customers["latitude"].to_numpy())[:, None]
    lon1 = np.radians(customers["longitude"].to_numpy())[:, None]
    lat2 = np.radians(stores["latitude"].to_numpy())[None, :]
    lon2 = np.radians(stores["longitude"].to_numpy())[None, :]

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )
    a = np.clip(a, 0, 1)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def optimize(customers, stores, distances, k, dmax,
             revenue, variable_cost, fixed_cost, capacity,
             distance_cost, time_limit):
    n, m = distances.shape
    model = cp_model.CpModel()

    opened = [model.NewBoolVar(f"open_{j}") for j in range(m)]
    assignments = {}
    customer_vars = [[] for _ in range(n)]
    store_vars = [[] for _ in range(m)]
    objective = []

    # Scale rupee values to integer paise for CP-SAT.
    scale = 100
    margin = int(round((revenue - variable_cost) * scale))
    fixed = int(round(fixed_cost * scale))

    for i in range(n):
        for j in range(m):
            if distances[i, j] <= dmax:
                y = model.NewBoolVar(f"assign_{i}_{j}")
                assignments[i, j] = y
                customer_vars[i].append(y)
                store_vars[j].append(y)

                model.Add(y <= opened[j])

                travel_cost = int(round(
                    distances[i, j] * distance_cost * scale
                ))
                objective.append((margin - travel_cost) * y)

    # Each customer can be assigned to at most one store.
    # Unserved customers are permitted.
    for i in range(n):
        if customer_vars[i]:
            model.Add(sum(customer_vars[i]) <= 1)

    # Capacity constraints.
    for j in range(m):
        if store_vars[j]:
            model.Add(sum(store_vars[j]) <= capacity * opened[j])
        else:
            model.Add(opened[j] == 0)

    model.Add(sum(opened) <= min(k, m))

    for j in range(m):
        objective.append(-fixed * opened[j])

    model.Maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    result = {
        "limit": k,
        "status": solver.StatusName(status),
        "profit": None,
        "opened": [],
        "assignments": pd.DataFrame(),
        "served": 0,
        "unserved": n,
        "avg_distance": 0.0,
        "max_distance": 0.0,
        "utilization": 0.0,
    }

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return result

    open_indices = [
        j for j in range(m) if solver.Value(opened[j]) == 1
    ]

    rows = []
    for (i, j), var in assignments.items():
        if solver.Value(var) == 1:
            rows.append({
                "customer_index": i,
                "store_index": j,
                "customer_id": customers.iloc[i]["customer_id"],
                "store_id": stores.iloc[j]["store_id"],
                "distance_km": distances[i, j],
            })

    assignment_df = pd.DataFrame(rows)
    served = len(assignment_df)
    unserved = n - served

    if served:
        total_distance = assignment_df["distance_km"].sum()
        avg_distance = total_distance / served
        max_assigned = assignment_df["distance_km"].max()
    else:
        total_distance = 0.0
        avg_distance = 0.0
        max_assigned = 0.0

    actual_revenue = served * revenue
    actual_variable_cost = served * variable_cost
    actual_distance_cost = total_distance * distance_cost
    actual_fixed_cost = len(open_indices) * fixed_cost
    profit = (
        actual_revenue
        - actual_variable_cost
        - actual_distance_cost
        - actual_fixed_cost
    )

    total_capacity = len(open_indices) * capacity
    utilization = (
        100 * served / total_capacity if total_capacity else 0.0
    )

    result.update({
        "profit": profit,
        "opened": open_indices,
        "assignments": assignment_df,
        "served": served,
        "unserved": unserved,
        "avg_distance": avg_distance,
        "max_distance": max_assigned,
        "utilization": utilization,
        "revenue": actual_revenue,
        "variable_cost": actual_variable_cost,
        "distance_cost": actual_distance_cost,
        "fixed_cost": actual_fixed_cost,
    })
    return result


def draw_map(customers, stores, result, dmax, show_lines=True):
    fmap = folium.Map(location=CENTER, zoom_start=11)

    # Candidate locations.
    opened_set = set(result["opened"])

    for j, store in stores.iterrows():
        is_open = j in opened_set
        folium.CircleMarker(
            [store["latitude"], store["longitude"]],
            radius=7 if is_open else 4,
            color="green" if is_open else "gray",
            fill=True,
            fill_opacity=0.9,
            tooltip=(
                f"{store['store_id']} - "
                f"{'OPEN' if is_open else 'CANDIDATE'}"
            ),
        ).add_to(fmap)

    assigned = result["assignments"]
    assigned_by_customer = {}

    if not assigned.empty:
        assigned_by_customer = dict(zip(
            assigned["customer_index"], assigned["store_index"]
        ))

    colors = [
        "red", "blue", "purple", "orange", "darkgreen",
        "cadetblue", "darkred", "pink", "darkblue", "lightgreen",
    ]

    # Plot all customers, colored by assigned store or gray if unserved.
    for i, customer in customers.iterrows():
        if i in assigned_by_customer:
            j = assigned_by_customer[i]
            color = colors[j % len(colors)]
            tooltip = f"{customer['customer_id']} -> {stores.iloc[j]['store_id']}"
        else:
            color = "gray"
            tooltip = f"{customer['customer_id']} - Unserved"

        folium.CircleMarker(
            [customer["latitude"], customer["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.6,
            weight=1,
            tooltip=tooltip,
        ).add_to(fmap)

    # Draw at most 500 assignment lines to keep the map readable.
    if show_lines and not assigned.empty:
        for _, row in assigned.head(500).iterrows():
            c = customers.iloc[int(row["customer_index"])]
            s = stores.iloc[int(row["store_index"])]

            folium.PolyLine(
                [
                    [c["latitude"], c["longitude"]],
                    [s["latitude"], s["longitude"]],
                ],
                color=colors[int(row["store_index"]) % len(colors)],
                weight=1,
                opacity=0.3,
            ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap


if st.button("Run optimization", type="primary"):
    customers, stores = create_data(
        int(n_customers), int(n_candidates)
    )
    distances = calculate_distances(customers, stores)

    results = []

    with st.spinner("Optimizing dark store locations..."):
        for k in SCENARIOS:
            results.append(optimize(
                customers, stores, distances, k, max_distance,
                revenue, variable_cost, fixed_cost, int(capacity),
                distance_cost, int(time_limit),
            ))

    st.session_state["optimization_results"] = {
        "customers": customers,
        "stores": stores,
        "results": results,
        "max_distance": max_distance,
    }


data = st.session_state.get("optimization_results")

if data is None:
    st.info("Choose the inputs and click Run optimization.")
else:
    customers = data["customers"]
    stores = data["stores"]
    results = data["results"]

    st.header("1. KPI Results")

    kpi_rows = []
    for r in results:
        kpi_rows.append({
            "Store limit": r["limit"],
            "Solver status": r["status"],
            "Stores opened": len(r["opened"]) if r["profit"] is not None else None,
            "Customers served": r["served"] if r["profit"] is not None else None,
            "Customers unserved": r["unserved"] if r["profit"] is not None else None,
            "Revenue (₹)": r.get("revenue"),
            "Variable cost (₹)": r.get("variable_cost"),
            "Distance cost (₹)": r.get("distance_cost"),
            "Fixed cost (₹)": r.get("fixed_cost"),
            "Daily profit (₹)": r["profit"],
            "Average distance (km)": r["avg_distance"],
            "Maximum assigned distance (km)": r["max_distance"],
            "Capacity utilization (%)": r["utilization"],
        })

    kpi_df = pd.DataFrame(kpi_rows)
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download KPI CSV",
        kpi_df.to_csv(index=False).encode("utf-8"),
        file_name="dark_store_kpis.csv",
        mime="text/csv",
    )

    st.caption(
        "OPTIMAL means optimality was proven. FEASIBLE means a valid "
        "solution was found, but optimality was not proven before timeout."
    )

    st.header("2. Profit Comparison")

    chart_df = pd.DataFrame([
        {
            "Store limit": str(r["limit"]),
            "Daily profit (₹)": r["profit"],
            "Customers served": r["served"],
            "Status": r["status"],
        }
        for r in results if r["profit"] is not None
    ])

    if not chart_df.empty:
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(
                "Store limit:N",
                sort=["2", "5", "10", "20", "25"],
                title="Maximum stores allowed",
            ),
            y=alt.Y(
                "Daily profit (₹):Q",
                axis=alt.Axis(format=",.0f"),
                title="Daily operating profit (₹)",
            ),
            tooltip=[
                "Store limit",
                "Daily profit (₹)",
                "Customers served",
                "Status",
            ],
        ).properties(height=350)

        st.altair_chart(chart, use_container_width=True)

    st.header("3. Detailed Solution Map")

    available = [r for r in results if r["profit"] is not None]

    if available:
        selected_limit = st.selectbox(
            "Select maximum store limit",
            [r["limit"] for r in available],
            index=len(available) - 1,
        )
        selected = next(
            r for r in available if r["limit"] == selected_limit
        )

        show_lines = st.checkbox(
            "Show customer-to-store assignment lines", value=True
        )

        fmap = draw_map(
            customers, stores, selected, data["max_distance"], show_lines
        )
        st_folium(
            fmap,
            use_container_width=True,
            height=600,
            returned_objects=[],
        )

        st.subheader("Opened Store Details")

        opened_df = stores.iloc[selected["opened"]].copy()

        if not opened_df.empty:
            counts = selected["assignments"]["store_index"].value_counts()
            opened_df["orders_assigned"] = [
                int(counts.get(j, 0)) for j in selected["opened"]
            ]
            opened_df["utilization_pct"] = (
                100 * opened_df["orders_assigned"] / capacity
            ).round(2)

        st.dataframe(opened_df, use_container_width=True, hide_index=True)

        st.subheader("Customer Assignments")
        assignment_df = selected["assignments"].copy()
        st.dataframe(assignment_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download assignment CSV",
            assignment_df.to_csv(index=False).encode("utf-8"),
            file_name="customer_assignments.csv",
            mime="text/csv",
        )

        if selected["max_distance"] <= data["max_distance"] + 1e-6:
            st.success("All assigned customers satisfy the distance limit.")
        else:
            st.error("Distance constraint validation failed.")
    else:
        st.error(
            "No usable solution was found. Try increasing the distance "
            "limit or the solver time limit."
        )

st.caption(
    "Illustrative model: coordinates are simulated within an approximate "
    "Chennai area. Distances are straight-line distances, not road distances."
)
