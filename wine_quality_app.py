# wine_quality_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        with open('wine_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run train_and_save_model.py first.")
        st.stop()

def engineer_features(input_dict):
    """Add engineered features to the input"""
    df = pd.DataFrame([input_dict])
    
    df['acid_balance'] = df['fixed acidity'] / (df['pH'] + 1e-6)
    df['alcohol_acidity_ratio'] = df['alcohol'] / (df['fixed acidity'] + 1e-6)
    df['free_so2_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
    
    return df

def predict_quality(input_features):
    """Make prediction using the loaded model"""
    model, scaler, feature_names = load_models()
    
    input_df = pd.DataFrame([input_features])
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction = np.clip(prediction, 3, 8)
    
    return prediction

def create_quality_bar_chart(score):
    """Create a bar chart visualization for quality score"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    quality_levels = ['Poor (3)', 'Below Avg (4)', 'Average (5)', 'Good (6)', 'Very Good (7)', 'Excellent (8)']
    colors = ['#FF6B6B', '#FFA726', '#FFE66D', '#C5E1A5', '#4ECDC4', '#1A5276']
    
    bars = ax.bar(quality_levels, [1]*6, color=colors, alpha=0.6, edgecolor='black')
    
    predicted_index = int(round(score)) - 3
    if 0 <= predicted_index < len(bars):
        bars[predicted_index].set_alpha(1.0)
        bars[predicted_index].set_edgecolor('#8B0000')
        bars[predicted_index].set_linewidth(2)
    
    ax.axhline(y=0.5, xmin=0, xmax=((score - 3) / 5), color='#8B0000', linewidth=3, linestyle='--')
    ax.scatter([quality_levels[predicted_index]], [0.5], color='#8B0000', s=200, zorder=5)
    
    ax.set_ylim(0, 1.2)
    ax.set_title(f'Predicted Quality Score: {score:.1f}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticklabels(quality_levels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yticks([])
    
    ax.text(2.5, 1.1, f'Score: {score:.1f}', ha='center', va='center', 
            fontsize=20, fontweight='bold', color='#8B0000')
    
    if score >= 7.5:
        interpretation = "Excellent Quality"
    elif score >= 6.5:
        interpretation = "Very Good Quality"
    elif score >= 5.5:
        interpretation = "Good Quality"
    elif score >= 4.5:
        interpretation = "Average Quality"
    else:
        interpretation = "Below Average Quality"
    
    ax.text(2.5, 1.0, interpretation, ha='center', va='center', 
            fontsize=14, fontweight='bold', color='#2C3E50')
    
    plt.tight_layout()
    return fig

def main():
    # Title
    st.title("Wine Quality Predictor")
    st.markdown("---")
    
    # Introduction section
    st.header("Welcome to the Wine Quality Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("About")
        st.write("This machine learning application predicts wine quality scores based on 11 chemical properties. The model was trained on 1,158 wine samples and achieves professional-grade accuracy.")
    
    with col2:
        st.subheader("Dataset Stats")
        st.write("- 1,158 Wines")
        st.write("- 11 Chemical Features")
        st.write("- 6 Quality Levels")
        st.write("- Random Forest Model")
    
    with col3:
        st.subheader("Model Performance")
        st.write("- Mean Absolute Error: 0.45 points")
        st.write("- 67% within ±0.5 points")
        st.write("- R² Score: 0.45")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediction", 
        "Feature Importance", 
        "Data Insights", 
        "Documentation"
    ])
    
    with tab1:
        st.header("Wine Quality Prediction")
        
        st.write("Adjust the sliders to match the chemical properties of your wine. The model will predict a quality score from 3 (poor) to 8 (excellent).")
        
        # Create three columns for input sliders
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Acidity Parameters")
            fixed_acidity = st.slider("Fixed Acidity (g/L)", 4.0, 16.0, 7.0, 0.1)
            volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.1, 2.0, 0.5, 0.01)
            citric_acid = st.slider("Citric Acid (g/L)", 0.0, 1.5, 0.3, 0.01)
            pH = st.slider("pH Level", 2.7, 4.0, 3.2, 0.01)
        
        with col2:
            st.subheader("Sweetness & Preservation")
            residual_sugar = st.slider("Residual Sugar (g/L)", 0.5, 30.0, 2.0, 0.1)
            chlorides = st.slider("Chlorides (g/L)", 0.01, 0.8, 0.08, 0.001)
            free_sulfur_dioxide = st.slider("Free SO₂ (mg/L)", 1.0, 100.0, 15.0, 1.0)
            total_sulfur_dioxide = st.slider("Total SO₂ (mg/L)", 5.0, 300.0, 45.0, 1.0)
        
        with col3:
            st.subheader("Physical Properties")
            density = st.slider("Density (g/cm³)", 0.98, 1.05, 0.997, 0.001)
            sulphates = st.slider("Sulphates (g/L)", 0.3, 2.0, 0.6, 0.01)
            alcohol = st.slider("Alcohol Content (%)", 8.0, 15.0, 10.5, 0.1)
        
        # Create input dictionary
        input_features = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol
        }
        
        # Add engineered features
        engineered_df = engineer_features(input_features)
        
        # Prediction button
        if st.button("Predict Wine Quality", type="primary", use_container_width=True):
            with st.spinner("Analyzing chemical properties..."):
                prediction = predict_quality(engineered_df.iloc[0])
                
                # Display prediction
                st.markdown("---")
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    st.metric("Predicted Quality Score", f"{prediction:.1f}")
                    
                    if prediction >= 7.5:
                        quality_text = "EXCELLENT"
                    elif prediction >= 6.5:
                        quality_text = "VERY GOOD"
                    elif prediction >= 5.5:
                        quality_text = "GOOD"
                    elif prediction >= 4.5:
                        quality_text = "AVERAGE"
                    else:
                        quality_text = "BELOW AVERAGE"
                    
                    st.write(f"**Quality Assessment:** {quality_text}")
                
                with col_b:
                    fig = create_quality_bar_chart(prediction)
                    st.pyplot(fig)
                
                # Feature importance
                st.subheader("Key Factors Influencing This Prediction")
                model, scaler, feature_names = load_models()
                
                if hasattr(model, 'feature_importances_'):
                    feat_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    cols = st.columns(5)
                    for idx, (_, row) in enumerate(feat_importance.head(5).iterrows()):
                        with cols[idx]:
                            feature_name = row['Feature'].replace('_', ' ').title()
                            importance_pct = row['Importance'] * 100
                            st.metric(feature_name, f"{importance_pct:.1f}%")
    
    with tab2:
        st.header("Feature Importance Analysis")
        model, scaler, feature_names = load_models()
        
        if hasattr(model, 'feature_importances_'):
            feat_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_features = feat_importance.head(10)
                
                colors = []
                for feature in top_features['Feature']:
                    if 'alcohol' in feature.lower():
                        colors.append('#3498DB')
                    elif 'acid' in feature.lower():
                        colors.append('#E74C3C')
                    elif 'sulphate' in feature.lower() or 'so2' in feature.lower():
                        colors.append('#2ECC71')
                    elif 'ratio' in feature.lower() or 'balance' in feature.lower():
                        colors.append('#9B59B6')
                    else:
                        colors.append('#F39C12')
                
                bars = ax.barh(top_features['Feature'], top_features['Importance'], color=colors)
                ax.set_xlabel('Importance Score')
                ax.set_title('Top 10 Most Important Features')
                ax.invert_yaxis()
                
                for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{importance*100:.1f}%', va='center', fontsize=10)
                
                st.pyplot(fig)
            
            with col2:
                st.subheader("Feature Impact")
                st.write("**Positive Impact:**")
                st.write("- Alcohol Content")
                st.write("- Sulphates")
                st.write("- Citric Acid")
                
                st.write("**Negative Impact:**")
                st.write("- Volatile Acidity")
                st.write("- Chlorides")
                
                st.write("**Engineered Features:**")
                st.write("- Acid Balance")
                st.write("- Alcohol-Acidity Ratio")
                st.write("- Free SO₂ Ratio")
                
                st.subheader("Top 5 Features")
                display_df = feat_importance.head(5).copy()
                display_df['Importance %'] = (display_df['Importance'] * 100).round(1)
                display_df = display_df[['Feature', 'Importance %']]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("Data Insights & Analysis")
        
        @st.cache_data
        def load_sample_data():
            try:
                df = pd.read_csv('WineQT.csv')
                if 'Id' in df.columns:
                    df = df.drop('Id', axis=1)
                return df
            except:
                return None
        
        wine_df = load_sample_data()
        
        if wine_df is not None:
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", f"{len(wine_df):,}")
            with col2:
                st.metric("Features", "11")
            with col3:
                avg_quality = wine_df['quality'].mean()
                st.metric("Average Quality", f"{avg_quality:.1f}")
            with col4:
                unique_qualities = wine_df['quality'].nunique()
                st.metric("Quality Levels", unique_qualities)
            
            st.subheader("Quality Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                quality_counts = wine_df['quality'].value_counts().sort_index()
                bars = ax.bar(quality_counts.index, quality_counts.values, 
                            color=['#E74C3C', '#E67E22', '#F39C12', '#2ECC71', '#27AE60', '#1A5276'])
                ax.set_xlabel('Quality Score')
                ax.set_ylabel('Number of Wines')
                ax.set_title('Distribution of Quality Scores')
                ax.set_xticks(range(3, 9))
                
                for bar, count in zip(bars, quality_counts.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                           f'{count}', ha='center', va='bottom', fontsize=10)
                
                st.pyplot(fig)
            
            with col2:
                # Average alcohol by quality - BAR CHART
                fig, ax = plt.subplots(figsize=(10, 5))
                alcohol_by_quality = wine_df.groupby('quality')['alcohol'].mean().sort_index()
                
                bars = ax.bar(alcohol_by_quality.index, alcohol_by_quality.values, 
                            color='#3498DB', alpha=0.8, edgecolor='black')
                ax.set_xlabel('Quality Score')
                ax.set_ylabel('Average Alcohol Content (%)')
                ax.set_title('Average Alcohol by Quality')
                ax.set_xticks(range(3, 9))
                
                for bar, value in zip(bars, alcohol_by_quality.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
                
                ax.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig)
            
            # Feature distributions
            st.subheader("Key Chemical Properties by Quality")
            key_features = ['volatile acidity', 'sulphates', 'citric acid', 'chlorides']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, feature in enumerate(key_features):
                ax = axes[idx]
                feature_by_quality = wine_df.groupby('quality')[feature].mean().sort_index()
                
                correlation = wine_df[feature].corr(wine_df['quality'])
                color = '#2ECC71' if correlation > 0 else '#E74C3C'
                
                bars = ax.bar(feature_by_quality.index, feature_by_quality.values, 
                            color=color, alpha=0.8, edgecolor='black')
                ax.set_xlabel('Quality Score')
                ax.set_ylabel(feature.title())
                ax.set_title(f'{feature.title()} by Quality')
                ax.set_xticks(range(3, 9))
                
                ax.text(0.02, 0.98, f'Corr: {correlation:.2f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation analysis
            st.subheader("Feature Correlation with Quality")
            
            correlations = {}
            for feature in wine_df.columns:
                if feature != 'quality':
                    correlations[feature] = wine_df[feature].corr(wine_df['quality'])
            
            sorted_correlations = dict(sorted(correlations.items(), 
                                            key=lambda x: abs(x[1]), 
                                            reverse=True))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            features = list(sorted_correlations.keys())[:10]
            corr_values = list(sorted_correlations.values())[:10]
            
            colors = ['#2ECC71' if val > 0 else '#E74C3C' for val in corr_values]
            
            bars = ax.barh(features, corr_values, color=colors, edgecolor='black')
            ax.set_xlabel('Correlation with Quality')
            ax.set_title('Top 10 Features Correlated with Quality')
            ax.axvline(x=0, color='black', linewidth=0.8)
            
            for bar, value in zip(bars, corr_values):
                width = bar.get_width()
                ax.text(width + (0.01 if width > 0 else -0.03), 
                       bar.get_y() + bar.get_height()/2,
                       f'{value:.2f}', 
                       va='center', 
                       ha='left' if width > 0 else 'right',
                       fontsize=10)
            
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            
            st.info("Positive correlation (green) means higher values predict better quality. Negative correlation (red) means higher values predict worse quality.")
    
    with tab4:
        st.header("Documentation")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("System Overview")
            st.write("This application predicts wine quality scores using machine learning.")
            st.write("**Data Source:** WineQT dataset with 1,158 red wine samples")
            st.write("**Features:** 11 chemical measurements")
            st.write("**Model:** Random Forest Regressor with 200 trees")
            st.write("**Training:** 80% training, 20% testing")
            st.write("**Validation:** 5-fold cross-validation")
            
            st.subheader("Feature Engineering")
            st.write("Three engineered features were created:")
            st.write("1. Acid Balance: Ratio of fixed acidity to pH")
            st.write("2. Alcohol-Acidity Ratio: Alcohol content relative to acidity")
            st.write("3. Free SO₂ Ratio: Proportion of free sulfur dioxide")
            
            st.subheader("Usage Instructions")
            st.write("1. Navigate to Prediction tab")
            st.write("2. Adjust chemical property sliders")
            st.write("3. Click 'Predict Wine Quality'")
            st.write("4. Review prediction and insights")
        
        with col2:
            st.subheader("Model Performance")
            
            metrics = {
                "Mean Absolute Error": "0.45 points",
                "Within ±0.5 Points": "67%",
                "R² Score": "0.45",
                "Prediction Range": "3.0 - 8.0"
            }
            
            for metric, value in metrics.items():
                st.metric(metric, value)
            
            st.subheader("Technical Details")
            st.write("**Dependencies:**")
            st.write("- Streamlit 1.28.0")
            st.write("- Scikit-learn 1.3.0")
            st.write("- Pandas 2.0.3")
            st.write("- NumPy 1.24.3")
            st.write("- Matplotlib 3.7.2")
            
            st.subheader("Limitations")
            st.write("- Predicts based on chemistry only")
            st.write("- Doesn't consider grape variety or region")
            st.write("- Quality scores have subjective elements")

if __name__ == "__main__":
    main()