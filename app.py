import streamlit as st
import sympy as sp
import numpy as np
from scipy.optimize import linprog


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Optimization Solver",
    page_icon="⚡",
    layout="wide"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 30px;
}

.card {
    padding: 22px;
    border-radius: 16px;
    background-color: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    background-color: #eef6ff;
    border: 1px solid #bfdbfe;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">⚡ Optimization Solver</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Linear Optimization'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SETTINGS
# =========================================================

st.markdown("### ⚙️ Problem Settings")

col1, col2 = st.columns(2)

with col1:

    n_variables = st.number_input(
        "🔢 Number of variables",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col2:

    optimization_type = st.radio(
        "🎯 Optimization type",
        ["Maximize", "Minimize"],
        horizontal=True
    )


# =========================================================
# VARIABLE NAMES
# =========================================================

variables = [
    sp.Symbol(f"x{i + 1}")
    for i in range(int(n_variables))
]

variable_names = [
    str(variable)
    for variable in variables
]


# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

st.markdown("### 🎯 Objective Function")

objective = st.text_input(
    "Enter the objective function",
    placeholder="Example: 3*x1 + 5*x2"
)

st.caption(
    "Use x1, x2, x3, ... and * for multiplication."
)


# =========================================================
# CONSTRAINTS
# =========================================================

st.markdown("### 🔒 Constraints")

n_constraints = st.number_input(
    "Number of constraints",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

constraints = []

for i in range(int(n_constraints)):

    constraint = st.text_input(
        f"Constraint {i + 1}",
        placeholder="Example: 2*x1 + x2 <= 10",
        key=f"constraint_{i}"
    )

    constraints.append(constraint)


# =========================================================
# VARIABLE RESTRICTIONS
# =========================================================

st.markdown("### 📐 Variable Restrictions")

non_negative = st.checkbox(
    "Assume all variables are non-negative (x ≥ 0)",
    value=True
)


# =========================================================
# SOLVER FUNCTION
# =========================================================

def solve_linear_problem(
    objective_text,
    constraint_texts,
    variables,
    optimization_type,
    non_negative
):

    # -----------------------------------------------------
    # Parse objective
    # -----------------------------------------------------

    objective_expr = sp.sympify(
        objective_text,
        locals={str(v): v for v in variables}
    )

    objective_coefficients = [
        float(objective_expr.coeff(v))
        for v in variables
    ]


    # -----------------------------------------------------
    # Prepare constraints
    # -----------------------------------------------------

    A_ub = []
    b_ub = []

    A_eq = []
    b_eq = []


    for constraint in constraint_texts:

        constraint = constraint.strip()

        if "<=" in constraint:

            left, right = constraint.split("<=", 1)

            relation = "<="

        elif ">=" in constraint:

            left, right = constraint.split(">=", 1)

            relation = ">="

        elif "=" in constraint:

            left, right = constraint.split("=", 1)

            relation = "="

        else:

            raise ValueError(
                f"Invalid constraint: {constraint}"
            )


        left_expr = sp.sympify(
            left,
            locals={str(v): v for v in variables}
        )

        right_expr = sp.sympify(
            right,
            locals={str(v): v for v in variables}
        )


        # Move everything to the left:
        # left - right <= 0

        expression = left_expr - right_expr

        coefficients = [
            float(expression.coeff(v))
            for v in variables
        ]

        constant = float(
            expression.subs(
                {v: 0 for v in variables}
            )
        )


        # -------------------------------------------------
        # <= constraint
        # -------------------------------------------------

        if relation == "<=":

            A_ub.append(coefficients)
            b_ub.append(-constant)


        # -------------------------------------------------
        # >= constraint
        # -------------------------------------------------

        elif relation == ">=":

            A_ub.append(
                [-c for c in coefficients]
            )

            b_ub.append(constant)


        # -------------------------------------------------
        # Equality
        # -------------------------------------------------

        elif relation == "=":

            A_eq.append(coefficients)
            b_eq.append(-constant)


    # -----------------------------------------------------
    # Maximize -> Minimize(-objective)
    # -----------------------------------------------------

    c = objective_coefficients.copy()

    if optimization_type == "Maximize":

        c = [-value for value in c]


    # -----------------------------------------------------
    # Variable bounds
    # -----------------------------------------------------

    if non_negative:

        bounds = [
            (0, None)
            for _ in variables
        ]

    else:

        bounds = [
            (None, None)
            for _ in variables
        ]


    # -----------------------------------------------------
    # Run Linear Programming Solver
    # -----------------------------------------------------

    result = linprog(
        c,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs"
    )


    # -----------------------------------------------------
    # Check result
    # -----------------------------------------------------

    if not result.success:

        return None, None, result.message


    solution = result.x

    optimal_value = float(
        np.dot(
            objective_coefficients,
            solution
        )
    )


    return solution, optimal_value, result.message


# =========================================================
# PROBLEM PREVIEW
# =========================================================

st.markdown("### 👀 Problem Preview")

if objective.strip():

    operator = optimization_type

    st.markdown(
        f"""
        <div class="card">

        <h3>{operator}</h3>

        <p>
        <b>Objective:</b>
        </p>

        <p style="font-size:20px;">
        {objective}
        </p>

        <p>
        <b>Subject to:</b>
        </p>

        """,
        unsafe_allow_html=True
    )

    for constraint in constraints:

        if constraint.strip():

            st.write(constraint)

    if non_negative:

        st.write(
            "x1, x2, ..., xn ≥ 0"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

else:

    st.info(
        "💡 Enter an objective function to preview your problem."
    )


# =========================================================
# SOLVE BUTTON
# =========================================================

st.markdown("### 🚀 Solve")

if st.button(
    "🚀 SOLVE OPTIMIZATION PROBLEM",
    use_container_width=True
):

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------

    if not objective.strip():

        st.error(
            "❌ Please enter an objective function."
        )

        st.stop()


    if any(
        not constraint.strip()
        for constraint in constraints
    ):

        st.error(
            "❌ Please fill in all constraints."
        )

        st.stop()


    # -----------------------------------------------------
    # Solve
    # -----------------------------------------------------

    try:

        with st.spinner(
            "🔄 Searching for the optimal solution..."
        ):

            solution, optimal_value, message = (
                solve_linear_problem(
                    objective,
                    constraints,
                    variables,
                    optimization_type,
                    non_negative
                )
            )


        # -------------------------------------------------
        # No solution
        # -------------------------------------------------

        if solution is None:

            st.error(
                "❌ No optimal solution was found."
            )

            st.warning(message)


        # -------------------------------------------------
        # Solution found
        # -------------------------------------------------

        else:

            st.success(
                "✅ Optimal solution found!"
            )

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "## 🎯 Optimal Solution"
            )


            # ---------------------------------------------
            # Display variables
            # ---------------------------------------------

            result_columns = st.columns(
                len(variables)
            )

            for i, variable in enumerate(variables):

                with result_columns[i]:

                    st.metric(
                        str(variable),
                        f"{solution[i]:.4f}"
                    )


            st.divider()


            # ---------------------------------------------
            # Optimal value
            # ---------------------------------------------

            if optimization_type == "Maximize":

                st.metric(
                    "🏆 Maximum Value",
                    f"{optimal_value:.4f}"
                )

            else:

                st.metric(
                    "🏆 Minimum Value",
                    f"{optimal_value:.4f}"
                )


            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


            # ---------------------------------------------
            # Technical information
            # ---------------------------------------------

            with st.expander(
                "🔎 Solver information"
            ):

                st.write(
                    "Optimization method: Linear Programming"
                )

                st.write(
                    "Algorithm: HiGHS"
                )

                st.write(
                    "Number of variables:",
                    len(variables)
                )

                st.write(
                    "Number of constraints:",
                    len(constraints)
                )
    except Exception as e:

        st.error(
            f"❌ An error occurred: {e}"
        )