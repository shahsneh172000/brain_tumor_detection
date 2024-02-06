import streamlit as st

# st.write("This about page.")
# st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)

# st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background: url("https://img2.goodfon.com/wallpaper/nbig/4/26/brain-mozg-plata-impulsy-minimalizm-linii-chernyi-fon.jpg");
#              background-size: auto
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
# )
st.set_page_config(layout="wide")
st.title("Brain Tumor Detection")
page_bg_img = f"""
<style>

[data-testid="stAppViewContainer"] > .main {{
   
background-image: url("https://images.unsplash.com/photo/1250205787");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)





st.title("Abstract")
st.markdown("""<p style="text-align: justify;font-size:24px;border-radius:3%;">Brain is the most important and complex part of our body. Abnormal 
            growth of cell in brain is known as Brain tumor. Some are benign and 
            some are malignant. Brain tumor can start in brain or it can spread from 
            any part of the body and this is one of deadly cancers across all ages. 
            In 2018, brain tumours was ranked as the 10th most common kind of 
            tumour among Indians. The International Association of Cancer 
            Registries (IARC) reported that there are over 28,000 cases of brain 
            tumours reported in India each year and more than 24,000 people 
            reportedly die due to brain tumours annually.Magnetic Resonance 
            Imaging popularly known as MRI is one of the primary scans 
            to visualize the brain tumor. Proposed methodology was found to 
            deliver significant performance exceptional accuracy.</p> 
""",unsafe_allow_html=True)

st.title("Aim")
st.write("""<p class="mystyle">Brain Tumor Detection System will identify the tumor in 
            the MR image using Machine Learning or Deep Learning Model</p>""",unsafe_allow_html=True)


st.title("Objective")
st.markdown("""<p class="mystyle">•To study Machine Learning and Deep Learning Algorithms which are useful in brain tumor detection<br>•To study Image pre-processing and feature extraction algorithms<br>
•To achieve more accuracy than existing models<br>•	Pinpoint the major focus on the research.</p>""",unsafe_allow_html=True)


st.title("Scope")
# st.markdown("""<p class="mystyle">•	To develop an automated system for classification of brain tumors<br/>
# •	The system incorporates image processing, pattern analysis, and computer vision techniques and is expected to improve the efficiency of brain tumor screening
# <br/>•	The primary is to extract meaningful and accurate information from these images with the least error possible
# </p>
# """,unsafe_allow_html=True)
st.markdown("""<p class="mystyle">•	Our Proposed model can easily classify MR image uploaded by the user in malignant or benign.<br/>
•	The MR picture should be taken from upward of the brain. Model is trained on particular type of MR images which are taken from particular angle of the brain
<br/>•	Model can be used after image is taken in machine. <br/>
•	Image should be clean and with less noise.<br/>
•	Model does not provide live detection of tumor in MR scan.
<br/>
</p>
""",unsafe_allow_html=True)



st.markdown("""
<style>
.mystyle {
  # background-color:#D0D3D4;
  # color:#21618C;
  font-size:24px;
  border-radius:3
  text-align: justify;
}
</style>
""", unsafe_allow_html=True)

