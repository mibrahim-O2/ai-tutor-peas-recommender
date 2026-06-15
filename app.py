# ============================================================
# app.py
# ------------------------------------------------------------
# AI Tutor for Personalized Learning Recommendations
# Final Year BS CS -- AI Lab Project
#
# This is the main Streamlit entry point.
# All UI rendering happens here.
# All AI logic is imported from recommendation_engine.py
#
# Lab Guide Modules Covered:
#   [A] Problem Setup    -- Tab 1 input form + validation
#   [B] Core Logic       -- Tab 1 result via engine/model
#   [C] Visual UI        -- Tab 2 charts (bar/pie/line/radar)
#   [D] Explainability   -- Tab 3 reasoning + factors
#   [E] Evaluation       -- Tab 4 metrics + confusion matrix
#
# Run with:
#   streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from recommendation_engine import (
    load_data,
    train_model,
    run_rules,
    run_model,
    generate_explanation,
    get_tree_explanation,
    create_visuals,
    evaluate_model
)
from utils.helpers import (
    get_topic_list,
    validate_inputs,
    decode_recommendation,
    get_recommendation_color
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# Must be the very first Streamlit call in the script.
# Sets the browser tab title, icon, and layout width.
# ------------------------------------------------------------
st.set_page_config(
    page_title            = "AI Tutor -- Personalized Learning",
    page_icon             = "[AI]",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)


# ============================================================
# HELPER FUNCTION: Metric Card
# ------------------------------------------------------------
# Renders a styled card with a colored left border.
# Used to display recommendation outputs in a clean format.
# Lab Guide [C]: Result panel requirement.
# ============================================================

def metric_card(label, value, color="#1f77b4"):
    """
    Renders a styled metric card using st.markdown with inline CSS.

    Parameters:
        label (str): Small label text shown above the value.
        value (str): Main value text displayed in bold.
        color (str): Hex color for the left border and text.
    """
    st.markdown(
        f"""
        <div style="
            background-color : {color}18;
            border-left      : 5px solid {color};
            padding          : 14px 18px;
            border-radius    : 8px;
            margin-bottom    : 8px;
        ">
            <p style="margin:0; font-size:13px; color:#888;">
                {label}
            </p>
            <p style="margin:4px 0 0 0; font-size:22px;
                      font-weight:700; color:{color};">
                {value}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HELPER FUNCTION: Comparison Bar Chart
# ------------------------------------------------------------
# Builds a grouped bar chart comparing Rule-Based Engine vs
# Decision Tree across all four evaluation metrics.
# Lab Guide [E]: Compare at least two approaches.
# ============================================================

def comparison_chart(metrics):
    """
    Builds a grouped bar chart comparing Rule-Based Engine vs
    Decision Tree across Accuracy, Precision, Recall, F1.

    Parameters:
        metrics (dict): Output dictionary from evaluate_model().

    Returns:
        plotly.graph_objects.Figure
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    rules_vals = [
        metrics["rules_accuracy"],
        metrics["rules_precision"],
        metrics["rules_recall"],
        metrics["rules_f1"]
    ]
    tree_vals = [
        metrics["tree_accuracy"],
        metrics["tree_precision"],
        metrics["tree_recall"],
        metrics["tree_f1"]
    ]

    fig = go.Figure()

    # -- Rule-Based Engine bars (orange)
    fig.add_trace(go.Bar(
        name         = "Rule-Based Engine",
        x            = metric_names,
        y            = rules_vals,
        marker_color = "#FFA500",
        text         = [f"{v:.2%}" for v in rules_vals],
        textposition = "outside"
    ))

    # -- Decision Tree bars (blue)
    fig.add_trace(go.Bar(
        name         = "Decision Tree",
        x            = metric_names,
        y            = tree_vals,
        marker_color = "#1f77b4",
        text         = [f"{v:.2%}" for v in tree_vals],
        textposition = "outside"
    ))

    fig.update_layout(
        title            = "[COMPARE] Rule-Based Engine vs Decision Tree -- Metric Comparison",
        barmode          = "group",
        yaxis_range      = [0, 1.15],
        yaxis_tickformat = ".0%",
        xaxis_title      = "Metric",
        yaxis_title      = "Score",
        plot_bgcolor     = "rgba(0,0,0,0)",
        paper_bgcolor    = "rgba(0,0,0,0)",
        legend           = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = 1.02,
            xanchor     = "right",
            x           = 1
        )
    )

    return fig


# ============================================================
# HELPER FUNCTION: Confusion Matrix Heatmap
# ------------------------------------------------------------
# Renders an annotated heatmap from a numpy confusion matrix.
# Lab Guide [E]: Evaluation metrics requirement.
# ============================================================

def confusion_matrix_chart(cm, labels):
    """
    Builds an annotated confusion matrix heatmap using Plotly.

    Rows = Actual labels.
    Columns = Predicted labels.
    Diagonal = Correct predictions.

    Parameters:
        cm     (np.ndarray): Confusion matrix from evaluate_model().
        labels (list):       Ordered list of class label names.

    Returns:
        plotly.express figure (imshow).
    """
    fig = px.imshow(
        cm,
        labels                 = dict(x="Predicted", y="Actual", color="Count"),
        x                      = labels,
        y                      = labels,
        text_auto              = True,
        color_continuous_scale = "Blues",
        title                  = "[MATRIX] Confusion Matrix -- Decision Tree"
    )
    fig.update_layout(
        xaxis_title   = "Predicted Label",
        yaxis_title   = "Actual Label",
        paper_bgcolor = "rgba(0,0,0,0)"
    )
    return fig


# ============================================================
# MAIN RENDER FUNCTION
# ------------------------------------------------------------
# Single function that renders the entire Streamlit UI.
# Organized into 4 tabs mapping to Lab Guide modules.
#
# Tab 1 -- Get Recommendation  [A] Problem Setup + [B] Core Logic
# Tab 2 -- Charts              [C] Visual UI
# Tab 3 -- Explainability      [D] Explainability Module
# Tab 4 -- Evaluation          [E] Evaluation Module
# ============================================================

def render_ui():
    """
    Main UI render function.
    Called at module level so Streamlit can execute it on every rerun.
    """

    # ----------------------------------------------------------
    # APP HEADER
    # Shows project title and team info at the top of every page.
    # ----------------------------------------------------------
    st.markdown(
        """
        <div style="padding: 20px 0 10px 0;">
            <h1 style="margin:0;">
                [AI TUTOR] Personalized Learning Recommendations
            </h1>
            <p style="color:#888; margin:6px 0 0 0; font-size:16px;">
                Final Year BS CS -- AI Lab Project &nbsp;|&nbsp;
                Rule-Based Engine + Decision Tree Classifier
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    # ----------------------------------------------------------
    # SIDEBAR
    # Contains mode selector, model controls, and about info.
    # Lab Guide [C]: Controls (buttons, dropdowns, toggles).
    # ----------------------------------------------------------
    with st.sidebar:
        st.markdown("## [SETTINGS]")

        # -- AI mode selector
        # Switches between Rule-Based Engine and Decision Tree.
        mode = st.radio(
            label   = "Select AI Mode",
            options = ["Rule-Based Engine", "Decision Tree"],
            index   = 0,
            help    = (
                "Rule-Based: uses IF-THEN logic rules.\n"
                "Decision Tree: uses a trained ML model."
            )
        )

        st.divider()
        st.markdown("### [MODEL CONTROLS]")

        # -- Retrain button
        # Allows user to retrain the Decision Tree on demand.
        # Model is saved to models/decision_tree.pkl after training.
        retrain_btn = st.button(
            label               = "[>>] Retrain Decision Tree",
            use_container_width = True,
            help                = "Retrains the model on the dataset and saves it."
        )

        if retrain_btn:
            with st.spinner("Training Decision Tree... please wait."):
                df_train = load_data()
                if df_train is not None:
                    train_result = train_model(df_train)
                    if train_result["status"] == "success":
                        st.success(
                            f"[OK] Model retrained successfully.\n"
                            f"Accuracy: {train_result['accuracy']:.2%}"
                        )
                    else:
                        st.error(f"[ERROR] {train_result['message']}")
                else:
                    st.error(
                        "[ERROR] Dataset not found. "
                        "Run: python data/generate_dataset.py"
                    )

        st.divider()
        st.markdown("### [ABOUT]")
        st.markdown(
            "**Project:** AI Tutor for Personalized Learning\n\n"
            "**Team:**\n"
            "- Muhammad Ibrahim *(Leader)*\n"
            "- Arsal\n"
            "- Ali\n\n"
            "**Stack:** Python · Streamlit · scikit-learn · Plotly\n\n"
            "**AI Methods:** Rule-Based Engine + Decision Tree"
        )

    # ----------------------------------------------------------
    # LOAD DATASET
    # Loaded once here and shared across all tabs.
    # Avoids reading the CSV file multiple times per page load.
    # ----------------------------------------------------------
    df = load_data()

    # ----------------------------------------------------------
    # TAB LAYOUT
    # Four tabs map directly to Lab Guide required modules.
    # ----------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "[A+B] Get Recommendation",
        "[C] Charts",
        "[D] Explainability",
        "[E] Evaluation"
    ])


    # ==========================================================
    # TAB 1 -- GET RECOMMENDATION
    # Lab Guide Module [A]: Problem Setup
    # Lab Guide Module [B]: Core Logic
    # ==========================================================

    with tab1:

        st.subheader("[INPUT] Enter Your Quiz Results")
        st.caption(
            "Fill in your quiz details below and click "
            "**Get My Recommendation** to receive a personalized study plan."
        )

        # ------------------------------------------------------
        # INPUT FORM
        # Lab Guide [A]: Let user provide or select input data.
        # Three columns for compact layout.
        # ------------------------------------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            # Dropdown -- topic selection from fixed list
            topic = st.selectbox(
                label   = "Quiz Topic",
                options = get_topic_list(),
                index   = 0,
                help    = "Select the topic you were just tested on."
            )

        with col2:
            # Slider -- quiz score from 0 to 100
            score = st.slider(
                label     = "Quiz Score (%)",
                min_value = 0,
                max_value = 100,
                value     = 60,
                step      = 1,
                help      = "Your score on the quiz (0 to 100)."
            )

        with col3:
            # Radio -- self-reported confidence level
            confidence = st.radio(
                label   = "Confidence Level",
                options = ["High", "Medium", "Low"],
                index   = 1,
                help    = "How confident did you feel during the quiz?"
            )

        col4, col5 = st.columns(2)

        with col4:
            # Slider -- response time in seconds
            response_time = st.slider(
                label     = "Response Time (seconds)",
                min_value = 5,
                max_value = 180,
                value     = 60,
                step      = 5,
                help      = "Total time spent answering the quiz."
            )

        with col5:
            # Number input -- previous score for trend analysis
            prev_score = st.number_input(
                label     = "Previous Score on this Topic (%)",
                min_value = 0.0,
                max_value = 100.0,
                value     = 50.0,
                step      = 0.5,
                help      = "Your score last time you took a quiz on this topic."
            )

        st.divider()

        # ------------------------------------------------------
        # INPUT PREVIEW METRICS
        # Shows a live summary of what the user entered before
        # they click the submit button.
        # Lab Guide [C]: Status messages and result panel.
        # ------------------------------------------------------
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric("Topic",         topic)
        col_p2.metric("Your Score",    f"{score}%")
        col_p3.metric("Confidence",    confidence)
        col_p4.metric("Response Time", f"{response_time}s")

        st.divider()

        # ------------------------------------------------------
        # SUBMIT BUTTON
        # Primary action button triggers recommendation engine.
        # Lab Guide [C]: Controls -- buttons.
        # ------------------------------------------------------
        run_btn = st.button(
            label               = "[>>] Get My Recommendation",
            type                = "primary",
            use_container_width = True
        )

        if run_btn:

            # --------------------------------------------------
            # INPUT VALIDATION
            # Lab Guide [A]: Validate user input and show clear
            # error messages.
            # --------------------------------------------------
            is_valid, error_msg = validate_inputs(score, response_time, topic)
            if not is_valid:
                # Show error and stop execution for this run
                st.error(f"[ERROR] {error_msg}")
                st.stop()

            # --------------------------------------------------
            # RUN AI ENGINE
            # Lab Guide [B]: Implement main algorithm/model.
            # Mode selected in sidebar determines which engine runs.
            # --------------------------------------------------
            with st.spinner("[PROCESSING] Analysing your quiz performance..."):

                # Build input dictionary passed to both engines
                input_data = {
                    "score_pct":           score,
                    "response_time":       response_time,
                    "confidence":          confidence,
                    "prev_score":          prev_score,
                    "topic":               topic,
                    # Extra keys used by radar chart in Tab 2
                    "score_pct_input":     score,
                    "response_time_input": response_time,
                    "prev_score_input":    prev_score
                }

                if mode == "Rule-Based Engine":
                    # Run IF-THEN rule engine (Phase 3)
                    result = run_rules(
                        score         = score,
                        confidence    = confidence,
                        response_time = response_time,
                        topic         = topic,
                        prev_score    = prev_score
                    )
                else:
                    # Run trained Decision Tree model (Phase 4)
                    result = run_model(input_data)

            # --------------------------------------------------
            # SAVE TO SESSION STATE
            # Stores result so Tabs 3 and 4 can access it
            # without rerunning the engine.
            # --------------------------------------------------
            st.session_state["last_result"]  = result
            st.session_state["last_context"] = input_data
            st.session_state["last_mode"]    = mode

            # --------------------------------------------------
            # RECOMMENDATION BANNER
            # Color-coded banner showing the final output.
            # Red = Review Basics | Orange = Practice More
            # Green = Next Topic
            # Lab Guide [C]: Result panel requirement.
            # --------------------------------------------------
            st.divider()
            rec   = result.get("recommendation", "Practice More")
            color = get_recommendation_color(rec)
            disp  = decode_recommendation(rec)

            st.markdown(
                f"""
                <div style="
                    background  : linear-gradient(135deg, {color}22, {color}08);
                    border-left : 6px solid {color};
                    padding     : 24px 28px;
                    border-radius: 10px;
                    margin-bottom: 16px;
                ">
                    <h2 style="color:{color}; margin:0 0 8px 0;">
                        [RESULT] {disp}
                    </h2>
                    <p style="margin:0; color:#666; font-size:15px;">
                        Topic: <strong>{topic}</strong>
                        &nbsp;&middot;&nbsp;
                        Score: <strong>{score}%</strong>
                        &nbsp;&middot;&nbsp;
                        Mode: <strong>{mode}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # --------------------------------------------------
            # OUTPUT METRIC CARDS
            # Four cards showing key outputs at a glance.
            # Lab Guide [C]: Result panel.
            # --------------------------------------------------
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card("Recommendation", rec, color)

            with c2:
                metric_card(
                    "Next Topic",
                    result.get("next_topic", topic),
                    "#1f77b4"
                )

            with c3:
                metric_card(
                    "Practice Questions",
                    str(result.get("practice_questions", 5)),
                    "#9467bd"
                )

            with c4:
                # Red if revision needed, green if not
                revision_color = (
                    "#FF4B4B" if result.get("revision_needed")
                    else "#00C853"
                )
                metric_card(
                    "Revision Needed",
                    "[!] Yes" if result.get("revision_needed") else "[OK] No",
                    revision_color
                )

            # --------------------------------------------------
            # QUICK EXPLANATION
            # Short natural-language paragraph shown immediately
            # after the result. Full explanation is in Tab 3.
            # Lab Guide [D]: Natural-language explanation.
            # --------------------------------------------------
            st.divider()
            st.markdown("#### [WHY] Quick Explanation")
            explanation = generate_explanation(result, input_data)
            st.info(explanation)

            # Show which rule or model path fired
            rule_text = result.get("rule_triggered", "")
            if rule_text:
                st.caption(f"[PATH] {rule_text}")

            # Guide user to other tabs for full details
            st.success(
                "[OK] Result saved. Switch to [C] Charts, "
                "[D] Explainability, or [E] Evaluation tabs "
                "for full details."
            )


    # ==========================================================
    # TAB 2 -- CHARTS
    # Lab Guide Module [C]: Visual UI (compulsory)
    # Required: charts, tables, controls, result panel,
    # status messages.
    # ==========================================================

    with tab2:

        st.subheader("[CHARTS] Performance Visualizations")

        # Guard: dataset must exist before drawing charts
        if df is None:
            st.error(
                "[ERROR] Dataset not found. "
                "Run: python data/generate_dataset.py"
            )
            st.stop()

        # ------------------------------------------------------
        # DATASET OVERVIEW METRICS
        # Summary numbers shown above the charts as context.
        # ------------------------------------------------------
        st.markdown("#### [DATA] Dataset Overview")
        ov1, ov2, ov3, ov4 = st.columns(4)
        ov1.metric("Total Records",     len(df))
        ov2.metric("Topics Covered",    df["topic"].nunique())
        ov3.metric("Avg Score",         f"{df['score_pct'].mean():.1f}%")
        ov4.metric("Avg Response Time", f"{df['response_time'].mean():.0f}s")

        st.divider()

        # ------------------------------------------------------
        # GENERATE CHARTS
        # Passes dataset and last input context to create_visuals().
        # Returns dict of four Plotly figures.
        # ------------------------------------------------------
        result_for_charts = st.session_state.get("last_context", {})
        charts = create_visuals(df, result_for_charts)

        if not charts:
            st.warning("[WARN] Charts could not be generated.")
            st.stop()

        # ------------------------------------------------------
        # ROW 1: Bar Chart + Pie Chart side by side
        # Bar  -- average quiz score per topic
        # Pie  -- distribution of recommendation labels
        # Lab Guide [C]: Charts (bar/pie).
        # ------------------------------------------------------
        col_l, col_r = st.columns(2)

        with col_l:
            st.plotly_chart(
                charts["bar_chart"],
                use_container_width=True
            )

        with col_r:
            st.plotly_chart(
                charts["pie_chart"],
                use_container_width=True
            )

        st.divider()

        # ------------------------------------------------------
        # ROW 2: Line Chart (full width)
        # Shows score trend across first 40 student records.
        # Lab Guide [C]: Charts (line).
        # ------------------------------------------------------
        st.plotly_chart(
            charts["line_chart"],
            use_container_width=True
        )

        st.divider()

        # ------------------------------------------------------
        # ROW 3: Radar Chart
        # Compares current student inputs vs dataset averages.
        # Only shown after user runs a recommendation in Tab 1.
        # Lab Guide [C]: Charts (scatter/radar).
        # ------------------------------------------------------
        if "last_context" in st.session_state:
            st.plotly_chart(
                charts["radar_chart"],
                use_container_width=True
            )
        else:
            st.info(
                "[INFO] Run a recommendation in the "
                "[A+B] Get Recommendation tab to see "
                "your personal radar chart."
            )

        st.divider()

        # ------------------------------------------------------
        # RAW DATA TABLE
        # Shows first 20 rows of the dataset.
        # Lab Guide [C]: Tables with highlights.
        # ------------------------------------------------------
        st.markdown("#### [TABLE] Dataset Sample")
        st.caption("Showing first 20 rows of student_scores.csv")
        st.dataframe(
            df.head(20).copy(),
            use_container_width=True,
            hide_index=True
        )


    # ==========================================================
    # TAB 3 -- EXPLAINABILITY
    # Lab Guide Module [D]: Explainability
    # Required: show why output was produced, display key
    # factors/rules, add natural-language explanation.
    # ==========================================================

    with tab3:

        st.subheader("[EXPLAIN] Explainability Module")
        st.caption(
            "This section shows exactly how and why "
            "the AI produced its recommendation."
        )

        # Guard: user must run a recommendation in Tab 1 first
        if "last_result" not in st.session_state:
            st.info(
                "[INFO] Go to the [A+B] Get Recommendation tab, "
                "enter your quiz results and click "
                "[>>] Get My Recommendation first."
            )
            st.stop()

        # Retrieve stored result and context from session state
        result  = st.session_state["last_result"]
        context = st.session_state["last_context"]
        mode    = st.session_state["last_mode"]

        rec   = result.get("recommendation", "Practice More")
        color = get_recommendation_color(rec)

        # ------------------------------------------------------
        # RECOMMENDATION SUMMARY BANNER
        # Repeats the recommendation so user sees it in context
        # of the explanation below.
        # ------------------------------------------------------
        st.markdown("#### [RESULT] Final Recommendation")
        st.markdown(
            f"""
            <div style="
                background    : {color}18;
                border-left   : 5px solid {color};
                padding       : 16px 20px;
                border-radius : 8px;
            ">
                <h3 style="color:{color}; margin:0;">
                    [RESULT] {decode_recommendation(rec)}
                </h3>
                <p style="margin:6px 0 0 0; color:#555;">
                    AI Mode used: <strong>{mode}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # ------------------------------------------------------
        # KEY FACTORS
        # Shows each input value with a signal (High/Low/OK).
        # Lab Guide [D]: Display key factors/features/rules.
        # ------------------------------------------------------
        st.markdown("#### [FACTORS] Key Factors in This Decision")

        # Extract input values from stored context
        score         = context.get("score_pct",     0)
        response_time = context.get("response_time", 0)
        confidence    = context.get("confidence",    "Medium")
        topic         = context.get("topic",         "")
        prev_score    = context.get("prev_score",    None)

        kf1, kf2, kf3, kf4 = st.columns(4)

        # Score signal -- compared against thresholds
        score_signal = (
            "[LOW]"    if score < 40  else
            "[MID]"    if score < 70  else
            "[HIGH]"
        )
        kf1.metric("Quiz Score",    f"{score}%",       score_signal)

        # Response time signal -- above 80s is slow
        time_signal = (
            "[SLOW]" if response_time > 80 else "[FAST]"
        )
        kf2.metric("Response Time", f"{response_time}s", time_signal)

        # Confidence signal
        conf_signal = (
            "[LOW CONFIDENCE]"    if confidence == "Low"    else
            "[MID CONFIDENCE]"    if confidence == "Medium" else
            "[HIGH CONFIDENCE]"
        )
        kf3.metric("Confidence", confidence, conf_signal)

        # Score trend signal -- compares current vs previous
        if prev_score is not None:
            change       = score - prev_score
            trend_label  = f"{'(+)' if change >= 0 else '(-)'} {abs(change):.1f}%"
            trend_signal = "[IMPROVING]" if change >= 0 else "[DECLINING]"
            kf4.metric("Score Trend", trend_label, trend_signal)
        else:
            trend_signal = "N/A"
            kf4.metric("Score Trend", "N/A", "No previous score")

        st.divider()

        # ------------------------------------------------------
        # FULL NATURAL-LANGUAGE EXPLANATION
        # Plain-English paragraph generated by generate_explanation().
        # Lab Guide [D]: Add short natural-language explanation.
        # ------------------------------------------------------
        st.markdown("#### [TEXT] Full Explanation")
        explanation = generate_explanation(result, context)
        st.info(explanation)

        st.divider()

        # ------------------------------------------------------
        # AI DECISION PATH
        # Shows the exact rule or tree path that fired.
        # Differs based on which AI mode was used.
        # Lab Guide [D]: Show why app produced that output.
        # Lab Guide [B]: Show intermediate steps.
        # ------------------------------------------------------
        st.markdown("#### [PATH] AI Decision Path")

        if mode == "Rule-Based Engine":

            # Primary rule that matched the score
            st.markdown("**Primary Rule Triggered:**")
            rule = result.get("rule_triggered", "")
            st.code(rule, language=None)

            # Modifier rules applied on top of primary rule
            modifiers = result.get("modifiers_applied", [])
            if modifiers:
                st.markdown("**Modifier Rules Applied:**")
                for mod in modifiers:
                    st.code(mod, language=None)
            else:
                st.success("[OK] No modifier rules were triggered.")

            # Full step-by-step reasoning log (expandable)
            st.markdown("**Step-by-Step Reasoning Log:**")
            steps = result.get("reasoning_steps", [])
            with st.expander(
                "Show full reasoning log", expanded=False
            ):
                for step in steps:
                    st.text(step)

        else:
            # Decision Tree mode -- show prediction and split path

            st.markdown("**Decision Tree Prediction:**")
            st.code(
                result.get("rule_triggered", "Decision Tree prediction"),
                language=None
            )

            # Tree split path (which features were checked)
            st.markdown("**Tree Split Path:**")
            tree_exp = get_tree_explanation(context)
            with st.expander("Show tree split path", expanded=True):
                st.code(tree_exp, language=None)

            # Reasoning steps logged during prediction
            st.markdown("**Model Reasoning Steps:**")
            steps = result.get("reasoning_steps", [])
            with st.expander("Show reasoning steps", expanded=False):
                for step in steps:
                    st.text(step)

        st.divider()

        # ------------------------------------------------------
        # INPUT VS THRESHOLD SUMMARY TABLE
        # Side-by-side comparison of each input against its
        # decision threshold used by the AI.
        # Lab Guide [C]: Tables with highlights.
        # Lab Guide [D]: Display key factors/rules used.
        # ------------------------------------------------------
        st.markdown("#### [TABLE] Input vs Threshold Summary")

        thresholds_data = {
            "Feature": [
                "Quiz Score (%)",
                "Response Time (s)",
                "Confidence Level",
                "Score Trend"
            ],
            "Your Value": [
                f"{score}%",
                f"{response_time}s",
                confidence,
                f"{score - prev_score:+.1f}%"
                if prev_score is not None else "N/A"
            ],
            "Threshold": [
                "< 40 / 40-70 / >= 70",
                "<= 80s (fast) / > 80s (slow)",
                "High / Medium / Low",
                "Drop > 10% flags revision"
            ],
            "Signal": [
                score_signal,
                time_signal,
                conf_signal,
                trend_signal if prev_score is not None else "N/A"
            ]
        }

        st.dataframe(
            pd.DataFrame(thresholds_data),
            use_container_width=True,
            hide_index=True
        )


    # ==========================================================
    # TAB 4 -- EVALUATION
    # Lab Guide Module [E]: Evaluation
    # Required: 2-3 performance indicators, compare at least
    # two settings/approaches, confusion matrix.
    # ==========================================================

    with tab4:

        st.subheader("[EVAL] Evaluation Module")
        st.caption(
            "Compares Rule-Based Engine vs Decision Tree across "
            "Accuracy, Precision, Recall, F1-Score, and Confusion Matrix."
        )

        # Guard: dataset must exist
        if df is None:
            st.error("[ERROR] Dataset not found.")
            st.stop()

        # ------------------------------------------------------
        # RUN EVALUATION
        # Trains (or loads) Decision Tree and evaluates both
        # approaches on the same 20% test split.
        # Lab Guide [E]: Compare at least two approaches.
        # ------------------------------------------------------
        with st.spinner(
            "[PROCESSING] Running evaluation on test split..."
        ):
            metrics = evaluate_model(df)

        # Guard: evaluation must succeed
        if not metrics or metrics["test_size"] == 0:
            st.error(
                "[ERROR] Evaluation failed. "
                "Check dataset and model."
            )
            st.stop()

        # ------------------------------------------------------
        # DATASET SPLIT INFO
        # Shows how many records were used for training vs testing.
        # ------------------------------------------------------
        st.markdown("#### [DATA] Dataset Split (80% Train / 20% Test)")
        ds1, ds2, ds3 = st.columns(3)
        ds1.metric("Total Records", len(df))
        ds2.metric("Training Set",  metrics["train_size"])
        ds3.metric("Test Set",      metrics["test_size"])

        st.divider()

        # ------------------------------------------------------
        # SIDE-BY-SIDE METRIC CARDS
        # Rule-Based (orange) vs Decision Tree (blue).
        # Lab Guide [E]: Show 2-3 performance indicators.
        # ------------------------------------------------------
        st.markdown("#### [METRICS] Performance Metrics")

        col_rules, col_tree = st.columns(2)

        with col_rules:
            st.markdown(
                "<h4 style='color:#FFA500;'>"
                "[RULES] Rule-Based Engine"
                "</h4>",
                unsafe_allow_html=True
            )
            metric_card("Accuracy",  f"{metrics['rules_accuracy']:.2%}",  "#FFA500")
            metric_card("Precision", f"{metrics['rules_precision']:.2%}", "#FFA500")
            metric_card("Recall",    f"{metrics['rules_recall']:.2%}",    "#FFA500")
            metric_card("F1 Score",  f"{metrics['rules_f1']:.2%}",        "#FFA500")

        with col_tree:
            st.markdown(
                "<h4 style='color:#1f77b4;'>"
                "[TREE] Decision Tree"
                "</h4>",
                unsafe_allow_html=True
            )
            metric_card("Accuracy",  f"{metrics['tree_accuracy']:.2%}",  "#1f77b4")
            metric_card("Precision", f"{metrics['tree_precision']:.2%}", "#1f77b4")
            metric_card("Recall",    f"{metrics['tree_recall']:.2%}",    "#1f77b4")
            metric_card("F1 Score",  f"{metrics['tree_f1']:.2%}",        "#1f77b4")

        st.divider()

        # ------------------------------------------------------
        # COMPARISON BAR CHART
        # Grouped bar chart showing both approaches side by side.
        # Lab Guide [E]: Compare at least two approaches.
        # Lab Guide [C]: Charts (bar).
        # ------------------------------------------------------
        st.markdown("#### [COMPARE] Side-by-Side Approach Comparison")
        comp_fig = comparison_chart(metrics)
        st.plotly_chart(comp_fig, use_container_width=True)

        # -- Winner callout message
        if metrics["tree_accuracy"] > metrics["rules_accuracy"]:
            diff = metrics["tree_accuracy"] - metrics["rules_accuracy"]
            st.success(
                f"[RESULT] Decision Tree outperforms the Rule-Based Engine "
                f"by {diff:.2%} in accuracy on this test set."
            )
        elif metrics["rules_accuracy"] > metrics["tree_accuracy"]:
            diff = metrics["rules_accuracy"] - metrics["tree_accuracy"]
            st.info(
                f"[RESULT] Rule-Based Engine outperforms the Decision Tree "
                f"by {diff:.2%} in accuracy on this test set."
            )
        else:
            st.info(
                "[RESULT] Both approaches achieved equal accuracy "
                "on this test set."
            )

        st.divider()

        # ------------------------------------------------------
        # CONFUSION MATRIX HEATMAP
        # Shows where the Decision Tree makes correct vs incorrect
        # predictions for each class.
        # Lab Guide [E]: Performance indicators.
        # ------------------------------------------------------
        st.markdown("#### [MATRIX] Confusion Matrix (Decision Tree)")
        st.caption(
            "Rows = Actual labels. "
            "Columns = Predicted labels. "
            "Diagonal cells = Correct predictions."
        )

        if metrics["confusion_matrix"] is not None:
            cm_fig = confusion_matrix_chart(
                metrics["confusion_matrix"],
                metrics["labels"]
            )
            st.plotly_chart(cm_fig, use_container_width=True)
        else:
            st.warning("[WARN] Confusion matrix not available.")

        st.divider()

        # ------------------------------------------------------
        # FULL CLASSIFICATION REPORT
        # Per-class breakdown of Precision, Recall, F1, Support.
        # Placed in expander to keep the page clean.
        # ------------------------------------------------------
        st.markdown(
            "#### [REPORT] Full Classification Report (Decision Tree)"
        )
        st.caption(
            "Per-class Precision, Recall, F1-Score, and Support."
        )
        with st.expander("Show full report", expanded=False):
            st.code(metrics["report"], language=None)

        st.divider()

        # ------------------------------------------------------
        # METRIC INTERPRETATION TABLE
        # Explains what each metric means in plain English.
        # Helps evaluators and viva panel understand the output.
        # Lab Guide [D]: Natural-language explanation.
        # ------------------------------------------------------
        st.markdown("#### [GUIDE] How to Interpret These Results")

        st.markdown("""
        | Term | Meaning |
        |---|---|
        | **Accuracy** | Percentage of all predictions that were correct |
        | **Precision** | Of all students predicted as class X, how many actually were X |
        | **Recall** | Of all actual class X students, how many did the model identify |
        | **F1 Score** | Harmonic mean of Precision and Recall -- best single metric |
        | **Confusion Matrix** | Shows where the model confuses one class for another |
        """)

        st.markdown("""
        **Why compare two approaches?**

        The Rule-Based Engine uses hand-crafted IF-THEN logic --
        transparent, explainable, and requires no training data.

        The Decision Tree learns patterns automatically from labeled
        data -- adaptive and data-driven.

        Comparing both satisfies the Lab Guide requirement to evaluate
        *at least two settings or approaches* in the Evaluation Module.
        """)


# ============================================================
# ENTRY POINT
# ------------------------------------------------------------
# Streamlit imports app.py as a module, so render_ui() must
# be called at module level (not only inside if __name__).
# Both branches call the same function.
# ============================================================

if __name__ == "__main__":
    render_ui()
else:
    render_ui()