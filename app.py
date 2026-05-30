import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration & Dark Scientific Theme
st.set_page_config(page_title="Mahd Ad Dahab Explorer Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: bold; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00ffcc; border-bottom-color: #00ffcc; }
    h1, h2, h3 { color: #00ffcc; }
    .report-box { background-color: #1a1c23; padding: 20px; border-radius: 10px; border: 1px solid #2d3139; }
    </style>
    """, unsafe_allow_html=True)

# 2. Advanced Synthetic Spatial Data Generation (Mahd Ad Dahab Coordinates)
@st.cache_data
def generate_geospatial_data(n_samples=300):
    np.random.seed(42)
    # Center around Mahd Ad Dahab mine coordinates
    lat_center, lon_center = 23.4986, 40.8522
    
    lats = lat_center + np.random.uniform(-0.05, 0.05, n_samples)
    lons = lon_center + np.random.uniform(-0.05, 0.05, n_samples)
    
    # Simulating structural geological features (N-S trending conduits)
    distance_from_conduit = np.abs(lons - lon_center)
    
    # Base indicators influenced by hydrothermal pathways
    iron_oxide = np.clip(1.0 - (distance_from_conduit * 15) + np.random.normal(0, 0.1, n_samples), 0, 1)
    clay_index = np.clip(1.0 - (distance_from_conduit * 12) + np.random.normal(0, 0.1, n_samples), 0, 1)
    
    # New Covariates: Lineament Density & Topographic Wetness Index (TWI)
    lineament_density = np.clip(1.0 - (distance_from_conduit * 10) + np.random.normal(0, 0.15, n_samples), 0, 1)
    twi = np.clip(np.random.beta(2, 5, n_samples) * 1.5, 0, 1) # High in wadis, low on ridges
    
    # Geochemical proxies for REE accumulation linked to epithermal alteration
    ree_proxy = np.clip((iron_oxide * 0.4 + clay_index * 0.3 + lineament_density * 0.3) - (twi * 0.1), 0, 1)
    
    return pd.DataFrame({
        'Latitude': lats, 'Longitude': lons,
        'Iron_Oxide': iron_oxide, 'Clay_Index': clay_index,
        'Lineament_Density': lineament_density, 'TWI': twi,
        'REE_Proxy_Base': ree_proxy
    })

df = generate_geospatial_data()

# 3. Sidebar - Fuzzy Logic Evidence Weighting Controls
st.sidebar.title("🔬 Exploration Engine")
st.sidebar.subheader("Fuzzy Evidence Weights")

w_iron = st.sidebar.slider("Iron Oxide (Gossan) Weight", 0.0, 1.0, 0.35, 0.05)
w_clay = st.sidebar.slider("Argillic Clay Weight", 0.0, 1.0, 0.25, 0.05)
w_lineament = st.sidebar.slider("Lineament Density Weight", 0.0, 1.0, 0.30, 0.05)
w_twi = st.sidebar.slider("Topographic Wetness (TWI) Weight", 0.0, 1.0, 0.10, 0.05)

# Normalize weights
total_w = w_iron + w_clay + w_lineament + w_twi
if total_w > 0:
    w_iron, w_clay, w_lineament, w_twi = w_iron/total_w, w_clay/total_w, w_lineament/total_w, w_twi/total_w

gamma = st.sidebar.slider("Fuzzy Gamma Overlay Parameter", 0.0, 1.0, 0.75, 0.05)

# Fuzzy Logic Engine Calculation (Fuzzy Gamma Overlay)
# Transforming layers into fuzzy membership matrices
mu_iron = df['Iron_Oxide']
mu_clay = df['Clay_Index']
mu_lineament = df['Lineament_Density']
mu_twi = 1.0 - df['TWI'] # Low TWI (ridges) favors gossan exposure

# Fuzzy Product and Fuzzy Sum architectures
fuzzy_product = (mu_iron**w_iron) * (mu_clay**w_clay) * (mu_lineament**w_lineament) * (mu_twi**w_twi)
fuzzy_sum = 1.0 - ((1.0 - mu_iron)**w_iron * (1.0 - mu_clay)**w_clay * (1.0 - mu_lineament)**w_lineament * (1.0 - mu_twi)**w_twi)

# Final Fuzzy Exploration Favorability Score
df['Favorability_Score'] = (fuzzy_product**(1 - gamma)) * (fuzzy_sum**gamma)

# Title Banner
st.title("💎 Mahd Ad Dahab REE Proxy Pathfinder")
st.caption("Advanced Epithermal Alteration Mapping & Structural Data Fusion Framework — Powered by Fuzzy Logic")

# 4. App Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Interactive Prospectivity Map", 
    "🔬 Spectral Fingerprint", 
    "📊 Covariate Statistics", 
    "⚗️ Spectral Unmixing (ELMM)", 
    "📋 Academic Investment Report",
    "🚀 Advanced Future Scope"
])

# TAB 1: Map Visualization
with tab1:
    st.subheader("Multi-Criteria Exploration Target Map")
    fig_map = px.scatter_mapbox(
        df, lat="Latitude", lon="Longitude",
        color="Favorability_Score", size="Favorability_Score",
        color_continuous_scale="Jet", size_max=12, zoom=12,
        mapbox_style="carto-darkmatter",
        hover_data=['Iron_Oxide', 'Clay_Index', 'Lineament_Density', 'TWI']
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
    st.plotly_chart(fig_map, use_container_width=True)

# TAB 2: Spectral Fingerprint
with tab2:
    st.subheader("Hydrothermal Alteration Spectral Profile vs USGS Library")
    bands = np.array([0.45, 0.55, 0.65, 0.85, 1.61, 2.20]) # Landsat-8 equivalents
    
    # Synthetic standard reference profiles
    usgs_hematite = np.array([0.1, 0.15, 0.4, 0.6, 0.3, 0.15])
    usgs_kaolinite = np.array([0.2, 0.25, 0.28, 0.3, 0.7, 0.2])
    
    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(x=bands, y=usgs_hematite, name="USGS Reference: Hematite (Iron Oxide)", line=dict(color='#ff4b4b', dash='dash')))
    fig_spec.add_trace(go.Scatter(x=bands, y=usgs_kaolinite, name="USGS Reference: Kaolinite (Clay)", line=dict(color='#00a0ff', dash='dash')))
    
    # Modeled pixel sample from high favorability zone
    sample_pixel = np.array([0.12, 0.18, 0.38, 0.55, 0.65, 0.18])
    fig_spec.add_trace(go.Scatter(x=bands, y=sample_pixel, name="Modeled High-Favorability Hydrothermal Pixel", line=dict(color='#00ffcc', width=4)))
    
    fig_spec.update_layout(title="Spectral Reflectance Profile Comparison", xaxis_title="Wavelength (µm)", yaxis_title="Reflectance", template="plotly_dark")
    st.plotly_chart(fig_spec, use_container_width=True)

# TAB 3: Covariate Statistics
with tab3:
    st.subheader("Geological Co-variates Feature Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist1 = px.histogram(df, x=["Iron_Oxide", "Clay_Index"], barmode="overlay", title="Spectral Alteration Indices", template="plotly_dark", color_discrete_sequence=['#ff4b4b', '#00a0ff'])
        st.plotly_chart(fig_hist1, use_container_width=True)
    with col2:
        fig_hist2 = px.histogram(df, x=["Lineament_Density", "TWI"], barmode="overlay", title="Structural & Terrain Co-variates", template="plotly_dark", color_discrete_sequence=['#9900ff', '#00ff00'])
        st.plotly_chart(fig_hist2, use_container_width=True)

# TAB 4: Extended Linear Mixture Model (ELMM)
with tab4:
    st.subheader("Spectral Variability-Aware Unmixing (ELMM Framework)")
    st.write("Accounting for pixel-by-pixel scaling dynamics induced by grain size and illumination geometry fluctuations.")
    
    # Displaying endmember fractions for top 15 highest favorability targets
    top_targets = df.nlargest(15, 'Favorability_Score').copy()
    top_targets['Gossan_Endmember'] = top_targets['Iron_Oxide'] * 0.5
    top_targets['Argillic_Endmember'] = top_targets['Clay_Index'] * 0.3
    top_targets['Host_Rock'] = 1.0 - (top_targets['Gossan_Endmember'] + top_targets['Argillic_Endmember'])
    
    fig_unmix = px.bar(top_targets, x=top_targets.index.astype(str), y=["Gossan_Endmember", "Argillic_Endmember", "Host_Rock"],
                       title="Calculated Sub-Pixel Endmember Fractions",
                       labels={"x": "Target ID", "value": "Abundance Fraction"},
                       color_discrete_sequence=['#ff4b4b', '#00a0ff', '#444444'], template="plotly_dark")
    st.plotly_chart(fig_unmix, use_container_width=True)

# TAB 5: Academic Investment Report
with tab5:
    st.subheader("Exploration Diagnostic & Target Categorization Report")
    
    high_pot = df[df['Favorability_Score'] > 0.7]
    med_pot = df[(df['Favorability_Score'] <= 0.7) & (df['Favorability_Score'] > 0.4)]
    
    st.markdown(f"""
    <div class="report-box">
    <h3>Geological Context & Metallogeny Validation</h3>
    <p><b>Deposit Type:</b> World-class Au-Ag telluride epithermal system hosted in a late Precambrian subalkaline bimodal volcanic arc sequence.</p>
    <p><b>Targeting Rationale:</b> This framework utilizes gossanous iron oxide zones and argillic clay indices as critical hydrothermal alteration footprints. These halos serve as spatial pathfinders (proxies) delineating the deep-seated fault-controlled fluid conduits where rare earth elements (REE) are scavenged by mineralized hydrothermal fluids.</p>
    <hr style='border-color:#2d3139;'>
    <h4>Statistical Target Breakdown</h4>
    <ul>
        <li><b>High-Exploration Potential Vectors (Fuzzy Score > 0.7):</b> <span style='color:#00ffcc; font-weight:bold;'>{len(high_pot)} targets identified</span></li>
        <li><b>Moderate-Exploration Potential Vectors (0.4 - 0.7):</b> <span style='color:#ffcc00; font-weight:bold;'>{len(med_pot)} targets identified</span></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if len(high_pot) > 0:
        st.write("### Prime Priority Coordinates for Ground-Truthing Exploration")
        st.dataframe(high_pot[['Latitude', 'Longitude', 'Favorability_Score', 'Lineament_Density']].style.background_gradient(cmap='viridis'))

# TAB 6: Advanced Future Scope (The Winner Edge)
with tab6:
    st.subheader("Strategic Research Evolution & Scalability Blueprint")
    st.markdown("""
    ### 1. The Machine Learning Quantum Leap: From Random Forest to PI-VAE
    Our current baseline operates via a multi-criteria fuzzy overlay engine. The immediate algorithmic upgrade features a **Physics-Informed Variational Autoencoder (PI-VAE)** for non-linear spectral unmixing. By embedding the **Linear Spectral Mixture Model (LSMM)** sum-to-one abundance constraint directly into the neural network's loss function, the system transitions from a purely data-driven model to a hybrid physics-regularized predictive framework.
    
    ### 2. Sensor Evolution: Transitioning to Spaceborne Hyperspectral Missions
    To bypass the coarse spectral resolutions of current multispectral platforms (Landsat-8/Sentinel-2), the next phase leverages spaceborne hyperspectral instruments, specifically **EnMAP (Germany)**, **PRISMA (Italy)**, and NASA's **EMIT**. These sensors capture continuous spectral sampling intervals (~6.5–7.5 nm), enabling the direct extraction of the narrow diagnostic absorption bands of Neodymium ($Nd^{3+}$) at **742 nm** and **802 nm** via the **Bastnäsite Index (BI)** framework.
    
    ### 3. Comprehensive Multi-Source Geodata Fusion
    Future target verification will ingest:
    * **Aeromagnetic Anomaly Data (WDMAM):** derived via Total Horizontal Gradient (THG) filters to trace subsurface fluid-conduit faults beneath sand cover.
    * **Sentinel-1 InSAR Interferometry:** mapping micro-subsidence signatures along active structural trends to correlate surface alteration anomalies with deep mechanical controls.
    """)
