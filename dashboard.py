"""
ABSA Visualization Dashboard — Flask + Plotly.
Visualizes dataset statistics, model comparisons, training curves, and confusion matrices.

Usage:
    python dashboard.py
    Then open http://localhost:5000
"""

import os
import sys
import json
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from config import RESULTS_DIR, ID2LABEL, LABEL2ID
from utils.data_loader import parse_xml, load_dataset

app = Flask(__name__)


# ============================================================
# Data Helpers
# ============================================================
def get_dataset_stats():
    """Compute dataset statistics for both domains."""
    stats = {}
    for domain in ["laptops", "restaurants"]:
        try:
            train_df, test_df = load_dataset(domain)
            polarity_counts = train_df["polarity"].value_counts().to_dict()
            avg_sentence_len = train_df["sentence"].str.split().str.len().mean()
            avg_aspect_len = train_df["aspect_term"].str.split().str.len().mean()

            # Sentence length distribution
            sent_lengths = train_df["sentence"].str.split().str.len().tolist()

            stats[domain] = {
                "total_train": len(train_df),
                "total_test": len(test_df),
                "polarity_counts": polarity_counts,
                "avg_sentence_len": round(avg_sentence_len, 1),
                "avg_aspect_len": round(avg_aspect_len, 1),
                "sentence_lengths": sent_lengths,
                "unique_aspects": train_df["aspect_term"].nunique(),
            }
        except Exception as e:
            stats[domain] = {"error": str(e)}
    return stats


def load_model_results():
    """Load all model results from results/ directory."""
    results = {}
    for json_file in glob.glob(
        os.path.join(RESULTS_DIR, "**", "*.json"), recursive=True
    ):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            # Derive key from path
            parts = (
                json_file.replace(RESULTS_DIR, "").strip(os.sep).replace(os.sep, "/")
            )
            key = parts.replace(".json", "").replace("/", "_")
            data["_path"] = json_file
            data["_dir"] = os.path.dirname(json_file)
            results[key] = data
        except Exception:
            pass
    return results


# ============================================================
# Plot Generators
# ============================================================
def create_polarity_distribution_chart(stats):
    """Create polarity distribution bar chart for both domains."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Laptops — Polarity Distribution",
            "Restaurants — Polarity Distribution",
        ],
        horizontal_spacing=0.12,
    )

    colors = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral": "#f39c12",
        "conflict": "#9b59b6",
    }

    for i, domain in enumerate(["laptops", "restaurants"], 1):
        if domain in stats and "polarity_counts" not in stats[domain]:
            continue
        counts = stats[domain]["polarity_counts"]
        labels = list(counts.keys())
        values = list(counts.values())
        bar_colors = [colors.get(l, "#95a5a6") for l in labels]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                marker_color=bar_colors,
                text=values,
                textposition="outside",
                name=domain.capitalize(),
                showlegend=False,
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        margin=dict(t=60, b=40),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_sentence_length_chart(stats):
    """Create sentence length histograms."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Laptops — Sentence Lengths", "Restaurants — Sentence Lengths"],
        horizontal_spacing=0.12,
    )

    for i, domain in enumerate(["laptops", "restaurants"], 1):
        if domain in stats and "sentence_lengths" in stats[domain]:
            fig.add_trace(
                go.Histogram(
                    x=stats[domain]["sentence_lengths"],
                    nbinsx=30,
                    marker_color="#0f3460",
                    opacity=0.85,
                    name=domain.capitalize(),
                    showlegend=False,
                ),
                row=1,
                col=i,
            )

    fig.update_xaxes(title_text="Number of Words")
    fig.update_yaxes(title_text="Count")
    fig.update_layout(
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        margin=dict(t=60, b=40),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_model_comparison_chart(results):
    """Create grouped bar chart comparing all models."""
    if not results:
        return None

    models = []
    accuracies = []
    f1_scores = []
    colors_list = []

    color_map = {
        "bert": "#e94560",
        "lstm": "#0f3460",
        "svm": "#533483",
        "rf": "#e94560",
        "traditional": "#533483",
    }

    for key, data in sorted(results.items()):
        model_name = data.get("model", key)
        domain = data.get("domain", "")
        acc = data.get("accuracy")
        f1 = data.get("macro_f1")
        if acc is not None and f1 is not None:
            label = f"{model_name}<br>({domain})"
            models.append(label)
            accuracies.append(round(acc * 100, 2))
            f1_scores.append(round(f1 * 100, 2))
            # Pick color
            for ckey, cval in color_map.items():
                if ckey in key.lower():
                    colors_list.append(cval)
                    break
            else:
                colors_list.append("#95a5a6")

    if not models:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Accuracy (%)",
            x=models,
            y=accuracies,
            marker_color="#2ecc71",
            text=[f"{a}%" for a in accuracies],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Macro-F1 (%)",
            x=models,
            y=f1_scores,
            marker_color="#3498db",
            text=[f"{f}%" for f in f1_scores],
            textposition="outside",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis_title="Score (%)",
        height=450,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=60, b=60),
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def create_training_curves_chart(results):
    """Create training curves from model histories."""
    charts = {}
    for key, data in results.items():
        history = data.get("history")
        if not history:
            continue

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Loss", "Accuracy"],
            horizontal_spacing=0.12,
        )

        epochs = list(range(1, len(history["train_loss"]) + 1))

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_loss"],
                mode="lines+markers",
                name="Train Loss",
                line=dict(color="#e94560"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                mode="lines+markers",
                name="Val Loss",
                line=dict(color="#0f3460"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["train_acc"],
                mode="lines+markers",
                name="Train Acc",
                line=dict(color="#e94560"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_acc"],
                mode="lines+markers",
                name="Val Acc",
                line=dict(color="#0f3460"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        model_name = data.get("model", key)
        domain = data.get("domain", "")
        fig.update_layout(
            title=f"{model_name} ({domain}) — Training Curves",
            height=350,
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font=dict(color="#e0e0e0"),
            margin=dict(t=60, b=40),
        )
        charts[key] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return charts


def create_dataset_overview_table(stats):
    """Create an HTML table with dataset overview stats."""
    rows = []
    for domain in ["laptops", "restaurants"]:
        if domain in stats and "error" not in stats[domain]:
            s = stats[domain]
            rows.append(
                {
                    "Domain": domain.capitalize(),
                    "Train Samples": s["total_train"],
                    "Test Samples": s["total_test"],
                    "Unique Aspects": s["unique_aspects"],
                    "Avg Sentence Len": s["avg_sentence_len"],
                    "Avg Aspect Len": s["avg_aspect_len"],
                }
            )
    return rows


# ============================================================
# HTML Template
# ============================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABSA Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --accent-1: #e94560;
            --accent-2: #0f3460;
            --accent-3: #533483;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0b0;
            --text-accent: #e94560;
            --border: #2a2a4a;
            --glow: 0 0 20px rgba(233, 69, 96, 0.15);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* ── Animated Background ── */
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(233, 69, 96, 0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(15, 52, 96, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(83, 52, 131, 0.05) 0%, transparent 50%);
            z-index: 0;
            pointer-events: none;
        }

        /* ── Header ── */
        .header {
            position: relative;
            z-index: 1;
            padding: 2rem 3rem;
            border-bottom: 1px solid var(--border);
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-1), #f39c12);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header .subtitle {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }

        .header .badge {
            background: var(--accent-1);
            color: white;
            padding: 0.4rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        /* ── Main Content ── */
        .container {
            position: relative;
            z-index: 1;
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* ── Section ── */
        .section {
            margin-bottom: 2.5rem;
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent-1);
            display: inline-block;
        }

        /* ── Cards ── */
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-3));
        }

        .card:hover {
            transform: translateY(-3px);
            box-shadow: var(--glow);
            border-color: var(--accent-1);
        }

        .card .label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent-1);
        }

        .card .sub {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }

        /* ── Chart Container ── */
        .chart-container {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .chart-container:hover {
            border-color: rgba(233, 69, 96, 0.3);
            box-shadow: var(--glow);
        }

        /* ── Table ── */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-card);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .data-table th {
            background: var(--accent-2);
            color: var(--text-primary);
            padding: 1rem;
            text-align: left;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }

        .data-table td {
            padding: 0.9rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.9rem;
        }

        .data-table tr:hover td {
            background: rgba(233, 69, 96, 0.05);
        }

        /* ── Status Badge ── */
        .status-badge {
            display: inline-block;
            padding: 0.2rem 0.8rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .status-badge.success {
            background: rgba(46, 204, 113, 0.15);
            color: #2ecc71;
        }

        .status-badge.pending {
            background: rgba(243, 156, 18, 0.15);
            color: #f39c12;
        }

        /* ── Results Grid ── */
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }

        .result-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }

        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--glow);
        }

        .result-card h3 {
            font-size: 1rem;
            color: var(--accent-1);
            margin-bottom: 1rem;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .metric-row .metric-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
        }

        .metric-row .metric-value {
            font-weight: 600;
            color: var(--text-primary);
        }

        /* ── Footer ── */
        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
        }

        /* ── Animations ── */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .section {
            animation: fadeInUp 0.6s ease forwards;
        }

        .section:nth-child(2) { animation-delay: 0.1s; }
        .section:nth-child(3) { animation-delay: 0.2s; }
        .section:nth-child(4) { animation-delay: 0.3s; }
        .section:nth-child(5) { animation-delay: 0.4s; }

        /* ── Responsive ── */
        @media (max-width: 768px) {
            .header { flex-direction: column; gap: 1rem; padding: 1.5rem; }
            .container { padding: 1rem; }
            .card-grid { grid-template-columns: repeat(2, 1fr); }
        }

        /* ── No Results Message ── */
        .no-results {
            text-align: center;
            padding: 3rem;
            color: var(--text-secondary);
        }

        .no-results .icon { font-size: 3rem; margin-bottom: 1rem; }
        .no-results p { font-size: 1.1rem; }
        .no-results .hint {
            font-size: 0.85rem;
            margin-top: 1rem;
            color: var(--accent-1);
            font-family: monospace;
            background: var(--bg-card);
            padding: 0.8rem;
            border-radius: 8px;
            display: inline-block;
        }

        /* ── Tabs ── */
        .tab-bar {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }
        .tab-btn {
            padding: 0.6rem 1.2rem;
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.9rem;
            font-family: 'Inter', sans-serif;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
        }
        .tab-btn:hover { color: var(--text-primary); background: rgba(255,255,255,0.03); }
        .tab-btn.active {
            color: var(--accent-1);
            border-bottom: 2px solid var(--accent-1);
            font-weight: 600;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <div>
            <h1>⚡ ABSA Dashboard</h1>
            <div class="subtitle">Aspect-Based Sentiment Analysis — SemEval-2014</div>
        </div>
        <div class="badge">LIVE DATA</div>
    </div>

    <div class="container">

        <!-- ── Dataset Overview ── -->
        <div class="section">
            <div class="section-title">📊 Dataset Overview</div>
            <div class="card-grid">
                {% for row in table_rows %}
                <div class="card">
                    <div class="label">{{ row['Domain'] }}</div>
                    <div class="value">{{ row['Train Samples'] }}</div>
                    <div class="sub">train samples</div>
                </div>
                {% endfor %}
                {% for row in table_rows %}
                <div class="card">
                    <div class="label">{{ row['Domain'] }} Aspects</div>
                    <div class="value">{{ row['Unique Aspects'] }}</div>
                    <div class="sub">unique aspect terms</div>
                </div>
                {% endfor %}
                {% for row in table_rows %}
                <div class="card">
                    <div class="label">{{ row['Domain'] }} Avg Len</div>
                    <div class="value">{{ row['Avg Sentence Len'] }}</div>
                    <div class="sub">words per sentence</div>
                </div>
                {% endfor %}
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>Domain</th>
                        <th>Train Samples</th>
                        <th>Test Samples</th>
                        <th>Unique Aspects</th>
                        <th>Avg Sentence Len</th>
                        <th>Avg Aspect Len</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in table_rows %}
                    <tr>
                        <td><strong>{{ row['Domain'] }}</strong></td>
                        <td>{{ row['Train Samples'] }}</td>
                        <td>{{ row['Test Samples'] }}</td>
                        <td>{{ row['Unique Aspects'] }}</td>
                        <td>{{ row['Avg Sentence Len'] }}</td>
                        <td>{{ row['Avg Aspect Len'] }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <!-- ── Polarity Distribution ── -->
        <div class="section">
            <div class="section-title">🎯 Polarity Distribution</div>
            <div class="chart-container">
                <div id="polarity-chart"></div>
            </div>
        </div>

        <!-- ── Sentence Lengths ── -->
        <div class="section">
            <div class="section-title">📏 Sentence Length Distribution</div>
            <div class="chart-container">
                <div id="length-chart"></div>
            </div>
        </div>

        <!-- ── Model Results ── -->
        <div class="section">
            <div class="section-title">🏆 Model Performance</div>

            {% if comparison_chart %}
            <div class="chart-container">
                <div id="comparison-chart"></div>
            </div>
            {% endif %}

            {% if model_results %}
            <div class="results-grid">
                {% for key, data in model_results.items() %}
                <div class="result-card">
                    <h3>{{ data.get('model', key) }} — {{ data.get('domain', '') | capitalize }}</h3>
                    <div class="metric-row">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">
                            {% if data.get('accuracy') is not none %}
                                {{ "%.2f"|format(data['accuracy'] * 100) }}%
                            {% else %}—{% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Macro-F1</span>
                        <span class="metric-value">
                            {% if data.get('macro_f1') is not none %}
                                {{ "%.2f"|format(data['macro_f1'] * 100) }}%
                            {% else %}—{% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Training Time</span>
                        <span class="metric-value">
                            {% if data.get('training_time_seconds') is not none %}
                                {{ "%.1f"|format(data['training_time_seconds']) }}s
                            {% else %}—{% endif %}
                        </span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Status</span>
                        <span class="status-badge success">Trained</span>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <div class="no-results">
                <div class="icon">🧪</div>
                <p>No model results yet. Train a model first!</p>
                <div class="hint">python main.py --model traditional --domain laptops</div>
            </div>
            {% endif %}
        </div>

        <!-- ── Training Curves ── -->
        {% if training_curves %}
        <div class="section">
            <div class="section-title">📈 Training Curves</div>
            <div class="tab-bar">
                {% for key in training_curves.keys() %}
                <button class="tab-btn {% if loop.first %}active{% endif %}" onclick="switchTab(event, 'curve-{{ key }}')">
                    {{ key.replace('_', ' ').title() }}
                </button>
                {% endfor %}
            </div>
            {% for key, chart in training_curves.items() %}
            <div id="curve-{{ key }}" class="tab-content chart-container {% if loop.first %}active{% endif %}">
                <div id="curve-chart-{{ key }}"></div>
            </div>
            {% endfor %}
        </div>
        {% endif %}

    </div>

    <div class="footer">
        ABSA Dashboard — SemEval-2014 Task 4 • BERT · LSTM+Attention · SVM · Random Forest
    </div>

    <script>
        // Render charts
        var polarityData = {{ polarity_chart | safe }};
        Plotly.newPlot('polarity-chart', polarityData.data, polarityData.layout, {responsive: true});

        var lengthData = {{ length_chart | safe }};
        Plotly.newPlot('length-chart', lengthData.data, lengthData.layout, {responsive: true});

        {% if comparison_chart %}
        var compData = {{ comparison_chart | safe }};
        Plotly.newPlot('comparison-chart', compData.data, compData.layout, {responsive: true});
        {% endif %}

        {% for key, chart in training_curves.items() %}
        var curveData_{{ key|replace('-','_')|replace(' ','_') }} = {{ chart | safe }};
        Plotly.newPlot('curve-chart-{{ key }}', curveData_{{ key|replace('-','_')|replace(' ','_') }}.data, curveData_{{ key|replace('-','_')|replace(' ','_') }}.layout, {responsive: true});
        {% endfor %}

        // Tab switching
        function switchTab(event, tabId) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            // Trigger Plotly resize
            window.dispatchEvent(new Event('resize'));
        }
    </script>
</body>
</html>
"""


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    stats = get_dataset_stats()
    model_results = load_model_results()

    polarity_chart = create_polarity_distribution_chart(stats)
    length_chart = create_sentence_length_chart(stats)
    comparison_chart = create_model_comparison_chart(model_results)
    training_curves = create_training_curves_chart(model_results)
    table_rows = create_dataset_overview_table(stats)

    return render_template_string(
        DASHBOARD_HTML,
        polarity_chart=polarity_chart,
        length_chart=length_chart,
        comparison_chart=comparison_chart,
        training_curves=training_curves,
        model_results=model_results,
        table_rows=table_rows,
    )


@app.route("/api/refresh")
def api_refresh():
    """API endpoint to re-check results."""
    results = load_model_results()
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  ⚡ ABSA Dashboard")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
