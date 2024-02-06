import streamlit as st
import pandas as pd
import numpy as np
# st.title("Result Analysis of Brain Tumor Detection")
st.title("Brain Tumor Detection")
st.title("Splitting Ratio 70:30")
st.markdown("""
<style>
table,th,td {
border:5px solid black;
  # background-color:#D0D3D4;
  # color:#21618C;
  font-size:24px;
  border-radius:3%;
  width:90%;
  text-align:center;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""<table>  
<tr>
    <th>Model</th>
    <th>Kernel</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
</tr>  

<tr>
    <th rowspan="3">DWT+SVM</th>
    <td>Linear</td>
    <td>95.55</td>
    <td>95.97</td>
    <td>96.34</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>96.88</td>
    <td>97.54</td>
    <td>96.75</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>96.44</td>
    <td>96.19</td>
    <td>97.36</td>
  </tr>

<tr>
    <th rowspan="3">Without Feature Extraction+SVM</th>
    <td>Linear</td>
    <td>93.88</td>
    <td>95.32</td>
    <td>92.57</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>92.11</td>
    <td>96.11</td>
    <td>88.23</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>95.44</td>
    <td>96.45</td>
    <td>94.52</td>
  </tr>


<tr>
    <th rowspan="3">PCA+SVM
</th>
    <td>Linear</td>
    <td>94.22</td>
    <td>97.01</td>
    <td>91.54</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>92.66</td>
    <td>96.25</td>
    <td>89.15</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>96.88</td>
    <td>97.37</td>
    <td>96.52</td>
  </tr>

  
<tr>
    <th rowspan="3">Central Tendency
+
SVM
(128x128)

</th>
    <td>Linear</td>
    <td>79.44</td>
    <td>82.21</td>
    <td>79.71</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>82.88</td>
    <td>94.25</td>
    <td>73.22</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>85.11</td>
    <td>87.31</td>
    <td>85.19</td>
  </tr>

  <tr>
    <th rowspan="3">Central Tendency
+
SVM
(256x256)


</th>
    <td>Linear</td>
    <td>88.33</td>
    <td>90.75</td>
    <td>87.62</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>89.88</td>
    <td>96.31</td>
    <td>84.87</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>87.55</td>
    <td>90.96</td>
    <td>85.80</td>
  </tr>

  <tr>
    <th>CNN


</th>
    <td>-</td>
    <td>98.13</td>
    <td>97.33</td>
    <td>98.2</td>
  </tr>
  

</table>  """,unsafe_allow_html=True)

st.title("Splitting Ratio 80:20")

st.markdown("""<table>  
<tr>
    <th>Model</th>
    <th>Kernel</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
</tr>  

<tr>
    <th rowspan="3">DWT+SVM</th>
    <td>Linear</td>
    <td>96.5</td>
    <td>96.8</td>
    <td>97.08</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>98.16</td>
    <td>98.53</td>
    <td>98.25</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>97.33</td>
    <td>97.11</td>
    <td>98.25</td>
  </tr>

<tr>
    <th rowspan="3">Without Feature Extraction+SVM</th>
    <td>Linear</td>
    <td>94.66</td>
    <td>95.75</td>
    <td>93.91</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>93.33</td>
    <td>96.36</td>
    <td>92.65</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>95.66</td>
    <td>96.10</td>
    <td>95.48</td>
  </tr>


<tr>
    <th rowspan="3">PCA+SVM
</th>
    <td>Linear</td>
    <td>95.33</td>
    <td>97.07</td>
    <td>94.02</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>94.33</td>
    <td>96.71</td>
    <td>92.45</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>97.33</td>
    <td>97.40</td>
    <td>97.84</td>
  </tr>

  
<tr>
    <th rowspan="3">Central Tendency
+
SVM
(128x128)

</th>
    <td>Linear</td>
    <td>80.50</td>
    <td>85.31</td>
    <td>79.59</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>89.16</td>
    <td>95.72</td>
    <td>84.83</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>87.50</td>
    <td>89.64</td>
    <td>88.33</td>
  </tr>

  <tr>
    <th rowspan="3">Central Tendency
+
SVM
(256x256)


</th>
    <td>Linear</td>
    <td>89.66</td>
    <td>93.76</td>
    <td>87.75</td>
  </tr>
  <tr>
    <td>Ploynomial</td>
    <td>91.66</td>
    <td>97.41</td>
    <td>87.75</td>
   
  </tr>
  <tr>
    <td>RBF</td>
    <td>89.16</td>
    <td>91.61</td>
    <td>89.21</td>
  </tr>

  <tr>
    <th>CNN


</th>
    <td>-</td>
    <td>99.05</td>
    <td>98.33</td>
    <td>99.1</td>
  </tr>
  

</table>  """,unsafe_allow_html=True)