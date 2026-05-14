import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tying the Data Knot",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #FDF6F0; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #1a1a2e;
    line-height: 1.1;
    margin: 0;
}

.hero-sub {
    font-size: 1.1rem;
    color: #666;
    margin-top: 0.5rem;
    font-weight: 300;
}

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #f0e8e0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    text-align: center;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #c94f7c;
    margin: 0;
}

.metric-label {
    font-size: 0.8rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

.predict-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid #f0e8e0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

.result-serious {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border: 1px solid #b8dfc4;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-casual {
    background: linear-gradient(135deg, #ffeeba, #ffd98a);
    border: 1px solid #f5c518;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    margin: 0;
}

.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin: 2px;
}

.pill-gay      { background: #e8d5f5; color: #6b21a8; }
.pill-lesbian  { background: #fce7f3; color: #9d174d; }
.pill-bisexual { background: #dbeafe; color: #1e40af; }
.pill-queer    { background: #d1fae5; color: #065f46; }
.pill-pan      { background: #fef3c7; color: #92400e; }
.pill-asexual  { background: #f3f4f6; color: #374151; }
.pill-straight { background: #fee2e2; color: #991b1b; }
.pill-demi     { background: #e0e7ff; color: #3730a3; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1a1a2e;
    border-bottom: 2px solid #c94f7c;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #c94f7c, #e8698a);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(201, 79, 124, 0.35);
}

.sidebar-label {
    font-size: 0.75rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('dating_app_behavior_dataset.csv')

    if 'interest_tags' in df.columns:
        df['interest_count'] = df['interest_tags'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

    core_features = [
        'bio_length', 'likes_received', 'app_usage_time_min',
        'message_sent_count', 'emoji_usage_rate', 'swipe_right_ratio',
        'mutual_matches', 'profile_pics_count', 'sexual_orientation'
    ]

    df_model = df[core_features + ['match_outcome']].copy()

    for col in df_model.columns:
        if df_model[col].dtype == 'object':
            df_model[col] = df_model[col].fillna(df_model[col].mode()[0])
        else:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    orientation_stats = df_model.groupby('sexual_orientation').agg({
        'bio_length': 'mean',
        'message_sent_count': 'mean',
        'emoji_usage_rate': 'mean',
        'swipe_right_ratio': 'mean'
    }).to_dict('index')

    def determine_seriousness(row):
        score = 0
        stats = orientation_stats[row['sexual_orientation']]
        if (stats['bio_length'] - 100) <= row['bio_length'] <= (stats['bio_length'] + 100): score += 1
        if row['swipe_right_ratio'] < stats['swipe_right_ratio']: score += 1
        if row['message_sent_count'] > stats['message_sent_count']: score += 1
        if (stats['emoji_usage_rate'] - 0.15) <= row['emoji_usage_rate'] <= (stats['emoji_usage_rate'] + 0.15): score += 1
        if row['app_usage_time_min'] > 0:
            if (row['message_sent_count'] / row['app_usage_time_min']) > 0.5: score += 1
        if row['swipe_right_ratio'] > 0:
            if (row['mutual_matches'] / row['swipe_right_ratio']) > 5: score += 1
        if row['match_outcome'] in ['Relationship Formed', 'No Action', 'Blocked']: score -= 1
        return 1 if score >= 4 else 0

    df_model['is_serious'] = df_model.apply(determine_seriousness, axis=1)

    X = df_model.drop(columns=['is_serious', 'match_outcome'])
    y = df_model['is_serious']

    numeric_features    = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = ['sexual_orientation']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    X_proc = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42)

    # Train all models
    models = {
        'Random Forest'      : RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
        'KNN'                : KNeighborsClassifier(n_neighbors=5),
        'SVM'                : LinearSVC(dual=False, max_iter=5000, random_state=42),
        'Decision Tree'      : DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Neural Network'     : MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1'      : f1_score(y_test, y_pred),
            'y_pred'  : y_pred,
            'cm'      : confusion_matrix(y_test, y_pred)
        }
        trained_models[name] = model

    return df_model, preprocessor, trained_models, results, y_test, orientation_stats, numeric_features


# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner('Loading models...'):
    df_model, preprocessor, trained_models, results, y_test, orientation_stats, numeric_features = load_and_train()

rf_model    = trained_models['Random Forest']
serious_rate = df_model['is_serious'].mean()
total_users  = len(df_model)
best_f1      = results['Random Forest']['f1']
best_acc     = results['Random Forest']['accuracy']

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💘 Tying the Data Knot")
    st.markdown("*Love, Life & Likes*")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔮 Live Predictor",
        "📊 Model Comparison",
        "🏳️‍🌈 LGBTQ Analysis",
        "ℹ️ About"
    ])
    st.divider()
    st.markdown("<p class='sidebar-label'>Group 15 · WIA1006/WID3006</p>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-label'>Session 2025/2026</p>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("<h1 class='hero-title'>Tying the Data Knot 💘</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Predicting serious relationship intent across LGBTQ+ communities using machine learning</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>{total_users:,}</p>
            <p class='metric-label'>Total Users</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>{serious_rate*100:.1f}%</p>
            <p class='metric-label'>Serious Daters</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>{best_acc*100:.1f}%</p>
            <p class='metric-label'>RF Accuracy</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <p class='metric-value'>{best_f1:.3f}</p>
            <p class='metric-label'>RF F1-Score</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<p class='section-header'>Serious rate by LGBTQ group</p>", unsafe_allow_html=True)
        grp = df_model.groupby('sexual_orientation')['is_serious'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.RdPu(np.linspace(0.4, 0.9, len(grp)))
        bars = ax.barh(grp.index, grp.values * 100, color=colors)
        ax.set_xlabel('Serious rate (%)')
        ax.set_title('Serious dater rate per orientation', fontsize=12)
        for bar, val in zip(bars, grp.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontsize=9)
        ax.set_xlim(0, 60)
        fig.patch.set_facecolor('#FDF6F0')
        ax.set_facecolor('#FDF6F0')
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("<p class='section-header'>Target distribution</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df_model['is_serious'].value_counts()
        ax.pie(counts, labels=['Casual', 'Serious'],
               autopct='%1.1f%%', startangle=90,
               colors=['#f0c4d4', '#c94f7c'],
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Overall serious vs casual split', fontsize=12)
        fig.patch.set_facecolor('#FDF6F0')
        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>What is is_serious?</p>", unsafe_allow_html=True)
    st.info("""
    **The Goldilocks Rule** — A user is labelled *serious* (score ≥ 4/7) if they meet a combination of:

    ✅ Bio length within ±100 of their orientation group average — invested in their profile  
    ✅ Swipe right ratio below group average — selective, not swiping everyone  
    ✅ Message count above group average — genuinely engaging  
    ✅ Emoji usage within ±0.15 of group average — natural communication style  
    ✅ Message density > 0.5 msgs/min — spending time actually talking  
    ✅ Match ROI > 5 — selective swiping that leads to actual matches  
    ⛔ Red flag penalty — outcome is Blocked, No Action, or Relationship Formed with no follow-through
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Predictor":
    st.markdown("<h1 class='hero-title'>Live Predictor 🔮</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Enter a user's profile details to predict their relationship intent</p>", unsafe_allow_html=True)
    st.markdown("---")

    orientations = sorted(df_model['sexual_orientation'].unique().tolist())

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='predict-card'>", unsafe_allow_html=True)
        st.markdown("#### 👤 Profile Details")

        orientation = st.selectbox("Sexual Orientation", orientations)
        bio_length  = st.slider("Bio Length (characters)", 0, 500,
                                int(orientation_stats[orientation]['bio_length']))
        profile_pics = st.slider("Profile Pictures", 0, 6, 3)
        emoji_usage  = st.slider("Emoji Usage Rate", 0.0, 1.0,
                                 float(orientation_stats[orientation]['emoji_usage_rate']), 0.01)

        st.markdown("#### 📱 App Behaviour")
        app_usage    = st.slider("App Usage (min/day)", 0, 300, 60)
        swipe_ratio  = st.slider("Swipe Right Ratio", 0.0, 1.0,
                                 float(orientation_stats[orientation]['swipe_right_ratio']), 0.01)
        msg_count    = st.slider("Messages Sent", 0, 100,
                                 int(orientation_stats[orientation]['message_sent_count']))
        likes        = st.slider("Likes Received", 0, 200, 100)
        matches      = st.slider("Mutual Matches", 0, 50, 10)

        predict_btn = st.button("✨ Predict Relationship Intent")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📋 Prediction Result")

        if predict_btn:
            # Build input row
            input_df = pd.DataFrame([{
                'bio_length'         : bio_length,
                'likes_received'     : likes,
                'app_usage_time_min' : app_usage,
                'message_sent_count' : msg_count,
                'emoji_usage_rate'   : emoji_usage,
                'swipe_right_ratio'  : swipe_ratio,
                'mutual_matches'     : matches,
                'profile_pics_count' : profile_pics,
                'sexual_orientation' : orientation
            }])

            X_input = preprocessor.transform(input_df)
            pred    = rf_model.predict(X_input)[0]

            # Compute score manually for breakdown
            stats = orientation_stats[orientation]
            score_breakdown = {
                'Bio length in range'       : (stats['bio_length']-100) <= bio_length <= (stats['bio_length']+100),
                'Selective swiper'          : swipe_ratio < stats['swipe_right_ratio'],
                'Active messenger'          : msg_count > stats['message_sent_count'],
                'Natural emoji usage'       : (stats['emoji_usage_rate']-0.15) <= emoji_usage <= (stats['emoji_usage_rate']+0.15),
                'High message density'      : (msg_count / app_usage > 0.5) if app_usage > 0 else False,
                'Good match ROI'            : (matches / swipe_ratio > 5) if swipe_ratio > 0 else False,
            }
            score = sum(score_breakdown.values())

            if pred == 1:
                st.markdown(f"""<div class='result-serious'>
                    <p class='result-title'>💚 Serious Dater</p>
                    <p style='margin:0.5rem 0 0'>Score: {score}/6 rules met</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='result-casual'>
                    <p class='result-title'>💛 Casual Dater</p>
                    <p style='margin:0.5rem 0 0'>Score: {score}/6 rules met</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Score Breakdown:**")
            for rule, passed in score_breakdown.items():
                icon = "✅" if passed else "❌"
                st.markdown(f"{icon} {rule}")

            st.markdown("<br>", unsafe_allow_html=True)
            # Show group average comparison
            st.markdown(f"**{orientation} group averages:**")
            st.markdown(f"- Avg bio length: `{stats['bio_length']:.0f}` chars")
            st.markdown(f"- Avg swipe ratio: `{stats['swipe_right_ratio']:.2f}`")
            st.markdown(f"- Avg messages: `{stats['message_sent_count']:.0f}`")
            st.markdown(f"- Avg emoji rate: `{stats['emoji_usage_rate']:.2f}`")
        else:
            st.info("👈 Fill in the profile details and click **Predict** to see results.")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**How it works:**")
            st.markdown("""
            The Random Forest model uses 8 behavioural features to predict relationship intent.
            The Goldilocks scoring compares each user against their orientation group's average behaviour —
            not a one-size-fits-all threshold.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown("<h1 class='hero-title'>Model Comparison 📊</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Performance across all 6 machine learning models</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Leaderboard table
    leaderboard = pd.DataFrame([
        {'Model': name, 'Accuracy': v['accuracy'], 'F1-Score': v['f1']}
        for name, v in results.items()
    ]).sort_values('F1-Score', ascending=False).reset_index(drop=True)
    leaderboard.index += 1
    leaderboard.index.name = 'Rank'
    leaderboard['Accuracy'] = leaderboard['Accuracy'].apply(lambda x: f'{x:.4f}')
    leaderboard['F1-Score'] = leaderboard['F1-Score'].apply(lambda x: f'{x:.4f}')

    st.markdown("<p class='section-header'>🏆 Model Leaderboard</p>", unsafe_allow_html=True)
    st.dataframe(leaderboard, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown("<p class='section-header'>Accuracy vs F1-Score</p>", unsafe_allow_html=True)
    chart_df = pd.DataFrame([
        {'Model': name, 'Metric': 'Accuracy', 'Score': v['accuracy']}
        for name, v in results.items()
    ] + [
        {'Model': name, 'Metric': 'F1-Score', 'Score': v['f1']}
        for name, v in results.items()
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    models_order = leaderboard['Model'].tolist()
    x = np.arange(len(models_order))
    width = 0.35
    acc_vals = [results[m]['accuracy'] for m in models_order]
    f1_vals  = [results[m]['f1'] for m in models_order]

    bars1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy', color='#f0c4d4', edgecolor='white')
    bars2 = ax.bar(x + width/2, f1_vals,  width, label='F1-Score',  color='#c94f7c', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models_order, rotation=20)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontsize=13)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.patch.set_facecolor('#FDF6F0')
    ax.set_facecolor('#FDF6F0')
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Confusion matrices
    st.markdown("<p class='section-header'>Confusion Matrices</p>", unsafe_allow_html=True)
    model_names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    cmaps = ['Blues', 'Purples', 'Greens', 'Oranges', 'YlOrBr', 'Reds']

    for i, (name, cmap) in enumerate(zip(model_names, cmaps)):
        cm = results[name]['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                    xticklabels=['Casual', 'Serious'],
                    yticklabels=['Casual', 'Serious'],
                    ax=axes[i], cbar=False)
        axes[i].set_title(f'{name}\nAcc={results[name]["accuracy"]:.3f} F1={results[name]["f1"]:.3f}',
                          fontsize=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    fig.patch.set_facecolor('#FDF6F0')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LGBTQ ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏳️‍🌈 LGBTQ Analysis":
    st.markdown("<h1 class='hero-title'>LGBTQ+ Analysis 🏳️‍🌈</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>How serious relationship intent varies across sexual orientation communities</p>", unsafe_allow_html=True)
    st.markdown("---")

    orientations = sorted(df_model['sexual_orientation'].unique())
    pill_colors  = {
        'Gay': 'pill-gay', 'Lesbian': 'pill-lesbian', 'Bisexual': 'pill-bisexual',
        'Queer': 'pill-queer', 'Pansexual': 'pill-pan', 'Asexual': 'pill-asexual',
        'Straight': 'pill-straight', 'Demisexual': 'pill-demi'
    }

    # Pills
    pills_html = " ".join([
        f"<span class='pill {pill_colors.get(o, \"pill-gay\")}'>{o}</span>"
        for o in orientations
    ])
    st.markdown(pills_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p class='section-header'>Serious rate per group</p>", unsafe_allow_html=True)
        grp_rate = df_model.groupby('sexual_orientation')['is_serious'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.RdPu(np.linspace(0.3, 0.9, len(grp_rate)))
        bars = ax.bar(grp_rate.index, grp_rate.values * 100, color=colors, edgecolor='white', width=0.6)
        for bar, val in zip(bars, grp_rate.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val*100:.1f}%', ha='center', fontsize=9)
        ax.set_ylabel('Serious rate (%)')
        ax.set_ylim(0, 80)
        ax.tick_params(axis='x', rotation=30)
        ax.set_title('% of serious daters per orientation', fontsize=11)
        fig.patch.set_facecolor('#FDF6F0')
        ax.set_facecolor('#FDF6F0')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<p class='section-header'>Group behaviour averages</p>", unsafe_allow_html=True)
        stats_display = pd.DataFrame(orientation_stats).T.round(2)
        stats_display.columns = ['Avg Bio', 'Avg Messages', 'Avg Emoji Rate', 'Avg Swipe Ratio']
        st.dataframe(stats_display, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("These group averages are used as personalised thresholds in the Goldilocks scoring — each orientation group is compared against their own community average, not a global standard.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Stacked bar — serious vs casual per group
    st.markdown("<p class='section-header'>Serious vs casual split by orientation</p>", unsafe_allow_html=True)
    dist = df_model.groupby('sexual_orientation')['is_serious'].value_counts(normalize=True).unstack().fillna(0) * 100
    dist.columns = ['Casual', 'Serious']

    fig, ax = plt.subplots(figsize=(12, 5))
    dist[['Casual', 'Serious']].plot(kind='bar', stacked=True, ax=ax,
                                     color=['#f0c4d4', '#c94f7c'], edgecolor='white', width=0.6)
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel('Sexual Orientation')
    ax.set_title('Casual vs Serious breakdown per LGBTQ+ group', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 115)
    fig.patch.set_facecolor('#FDF6F0')
    ax.set_facecolor('#FDF6F0')
    st.pyplot(fig)
    plt.close()

    # Filter by orientation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='section-header'>Explore a specific group</p>", unsafe_allow_html=True)
    selected = st.selectbox("Select orientation group", orientations)
    grp_df   = df_model[df_model['sexual_orientation'] == selected]

    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Total users",    f"{len(grp_df):,}")
    with g2:
        st.metric("Serious daters", f"{grp_df['is_serious'].sum():,}")
    with g3:
        st.metric("Serious rate",   f"{grp_df['is_serious'].mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(grp_df[grp_df['is_serious']==1]['message_sent_count'],
                 alpha=0.7, color='#c94f7c', label='Serious', bins=20)
    axes[0].hist(grp_df[grp_df['is_serious']==0]['message_sent_count'],
                 alpha=0.7, color='#f0c4d4', label='Casual',  bins=20)
    axes[0].set_title(f'{selected} — Message count distribution')
    axes[0].set_xlabel('Messages sent')
    axes[0].legend()

    axes[1].hist(grp_df[grp_df['is_serious']==1]['swipe_right_ratio'],
                 alpha=0.7, color='#c94f7c', label='Serious', bins=20)
    axes[1].hist(grp_df[grp_df['is_serious']==0]['swipe_right_ratio'],
                 alpha=0.7, color='#f0c4d4', label='Casual',  bins=20)
    axes[1].set_title(f'{selected} — Swipe ratio distribution')
    axes[1].set_xlabel('Swipe right ratio')
    axes[1].legend()

    for ax in axes:
        ax.set_facecolor('#FDF6F0')
    fig.patch.set_facecolor('#FDF6F0')
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("<h1 class='hero-title'>About This Project ℹ️</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### 🎓 WIA1006/WID3006 Machine Learning
    **Session 2025/2026 · Semester 2 · Group 15**

    ---

    ### 📌 Problem Statement
    *Can we predict whether a user on a dating app is a serious dater,
    and does this prediction differ across LGBTQ+ sexual orientation groups?*

    ---

    ### 🎯 Target Variable — `is_serious`
    An engineered label using the **Goldilocks Rule** (score ≥ 4 out of 7):
    - Bio length within ±100 of orientation group average
    - Swipe ratio below group average (selective)
    - Message count above group average (engaged)
    - Emoji usage within ±0.15 of group average
    - Message density > 0.5 msgs/min
    - Match ROI > 5 (selective + successful)
    - Red flag penalty for No Action / Blocked outcomes

    ---

    ### 🤖 Models Trained
    | Model | Description |
    |---|---|
    | **Random Forest** ⭐ | Ensemble of decision trees — best performer |
    | KNN | Distance-based — sensitive to imbalance |
    | SVM | Linear kernel — fast and stable |
    | Decision Tree | Single tree — most interpretable |
    | Logistic Regression | Statistical baseline |
    | Neural Network | MLP with 2 hidden layers |

    ---

    ### 📦 Dataset
    Dating App Behavior Dataset — 50,000 synthetic user records  
    19 features covering demographics, app usage, swipe behaviour, and match outcomes  
    Source: [Kaggle](https://www.kaggle.com/datasets/keyushnisar/dating-app-behavior-dataset)

    ---

    ### 🔑 Key Features Used
    `bio_length` · `message_sent_count` · `swipe_right_ratio` · `emoji_usage_rate`  
    `app_usage_time_min` · `mutual_matches` · `likes_received` · `profile_pics_count`  
    `sexual_orientation` (one-hot encoded)
    """)
