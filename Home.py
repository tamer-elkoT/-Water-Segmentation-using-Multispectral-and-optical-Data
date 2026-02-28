import streamlit as st

# 1. Page Config
st.set_page_config(page_title="AI Water Sentinel | Home", page_icon="🌊", layout="wide")

st.title("🌊 AI Water Sentinel: Global Water Monitoring")
st.markdown("### Empowering organizations with Deep Learning and Multispectral Satellite Intelligence.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header("🌍 The Global Impact")
    st.markdown("""
    Water is our most critical resource. Climate change, urban expansion, and agricultural demands are rapidly altering the Earth's surface water.
    
    **AI Water Sentinel** leverages the European Space Agency's Sentinel-2 satellite network and a custom **U-Net Neural Network** to track these changes with mathematical precision. 
    """)

with col2:
    st.header("🎯 Target Audience")
    st.markdown("""
    * **🏛️ Government & Urban Planners**
    * **🌾 Agricultural Ministries**
    * **🚁 Disaster Response Teams**
    * **🔬 Climate Scientists**
    """)

st.divider()

# 🚨 THE MAGIC BUTTON THAT SWITCHES FILES
st.markdown("<h2 style='text-align: center;'>Ready to analyze live satellite data?</h2>", unsafe_allow_html=True)

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    if st.button("🚀 Try It Out (Launch AI Engine)", type="primary", use_container_width=True):
        # This tells Streamlit to instantly load the tool file!
        st.switch_page("pages/1_Water_tool.py")

st.divider()

st.markdown("<div style='text-align: center; color: gray;'>", unsafe_allow_html=True)
st.markdown("Architected and Engineered by **Tamer** | AI Engineer")
st.markdown("Built with Python, FastAPI, TensorFlow, and Microsoft Planetary Computer")
st.markdown("</div>", unsafe_allow_html=True)