import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
  
st.title("Brain Tumor Detection")

st.header(" Accuracy ")
labels = ['DWT+SVM','SVM' , 'PCA+SVM' , """Central Tendency 
+ SVM 128x128""",
"""Central Tendency 
  + SVM 256x256""",'CNN']

acc = [98.16,95.66,97.33,89.16,91.66,99.05]
  
fig = plt.figure(figsize=(15, 8))
plt.bar(labels, acc, color=['purple', 'orange', 'green', 'blue', 'cyan' , 'red'], width=0.4)

for i in range(len(labels)):
  plt.text(i,acc[i]//2,acc[i],ha='center')

plt.xlabel("Model")
plt.ylabel("Accuracy")

st.pyplot(fig)

st.header(" Precision ")
labels = ['DWT+SVM','SVM' , 'PCA+SVM' , """Central Tendency 
+ SVM 128x128""",
"""Central Tendency 
  + SVM 256x256""",'CNN']

pre = [98.53,96.36,97.40,95.72,97.41,98.33,]
  
fig = plt.figure(figsize=(15, 8))
plt.bar(labels, pre, color=['purple', 'orange', 'green', 'blue', 'cyan' , 'red'], width=0.4)

for i in range(len(labels)):
  plt.text(i,pre[i]//2,pre[i],ha='center')

plt.xlabel("Model")
plt.ylabel("Precision")

plt.show()
st.pyplot(fig)

st.header(" Recall ")
labels = ['DWT+SVM','SVM' , 'PCA+SVM' , """Central Tendency 
+ SVM 128x128""",
"""Central Tendency 
  + SVM 256x256""",'CNN']

rec = [98.25,95.48,97.84,88.33,89.21,99.10]
  
fig = plt.figure(figsize=(15, 8))
plt.bar(labels, rec, color=['purple', 'orange', 'green', 'blue', 'cyan' , 'red'], width=0.4)

for i in range(len(labels)):
  plt.text(i,rec[i]//2,rec[i],ha='center')

plt.xlabel("Model")
plt.ylabel("Recall")
plt.show()
st.pyplot(fig)