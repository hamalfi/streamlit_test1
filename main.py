import streamlit as st

import xrayutilities as xu

import numpy as np
import xarray as xr
import pandas as pd

import time
# import tqdm
# from tqdm import trange
from PIL import Image

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import matplotlib.colors as colors

st.title('Streamlit 超入門')

"""
### Dataframe


"""

L = 653.1 #@param
pw1 = 0.172 #@param
pw2 = 0.172 #@param
pc1 = 551 #@param
pc2 = 254 #@param

#input detector and sample angles
#angles defines the angle of center position

tt_c = 23.2162+20 #@param
chi_c = 90-(59.5) #@param
th_c = 11.05 #@param
phi = 0 #@param

#input ROI

ri = 0 #@param
rf = 619 #@param
ci = 0 #@param
cf = 487 #@param

#input fine name

N_f= 200 #@param

b = np.arange(ri+1,rf+1,1)
a = np.arange(ci+1,cf+1,1)
COL, ROW = np.meshgrid(a,b)
col = COL.ravel()
row = ROW.ravel()

print (L, pw1,pw2, pc1, pc2)
print (tt_c, chi_c)
print (N_f)

c1 = 0
c2 = L*np.sin(tt_c/180*np.pi)
c3 = L*np.cos(tt_c/180*np.pi)

x=(row-pc1)*pw1+c1
y=-(col-pc2)*pw2*np.cos(tt_c/180*np.pi)+c2
z=(col-pc2)*pw2*np.sin(tt_c/180*np.pi)+c3

r=np.sqrt(x**2+y**2+z**2)

tt=np.arccos(z/r)
chi=np.arcsin(x/r/np.sin(tt))

tt= tt*180/np.pi
chi =chi *180/np.pi+chi_c

tt= np.round(tt, decimals= 3)
chi =np.round(chi, decimals= 3)

u=np.zeros_like(x)
Fn="./Streamlit/Udemy/lecture1/pilatus1.tif"
PATH="./Streamlit/Udemy/lecture1"

t = xu.io.imagereader.get_tiff (Fn)
#[x1:x2,y1:y2] x:row y:column
INT=t[ri:rf,ci:cf]
Int =INT.ravel()
c=np.stack([tt, chi, Int], 1)
#     np.savetxt(Fn+"_"+i_n+".txt", c)
df=pd.DataFrame(c, columns=['2theta', 'Chi', 'Intensity'])
print(df)
# st.dataframe(df.style.highlight_max(axis=0))
st.dataframe(df)

# # カラーマップ
# cm = plt.cm.get_cmap('jet')
# # figureを生成する
# fig = plt.figure()

# # axをfigureに設定する
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# # axに散布図を描画
# mappable = ax.scatter(df['2theta'], df['Chi'], c=np.log(df['Intensity']), s=10, cmap=cm)

# # カラーバーを付加
# fig.colorbar(mappable, ax=ax)

# # グラフを正方形に整形
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

# plt.savefig(directory_path + "/" + path_name_list[n]+ "_RSM.png", format="png", dpi=300)

# 表示
# plt.show()
# st.pyplot(fig)

latest_iteration= st.empty()
bar = st.progress(0)

for i in range(10):
    latest_iteration.text(f'Iteration{i+1}')
    bar.progress(i+1)
    time.sleep(0.1)

'Done!!'


"""
### Image


"""
option=st.selectbox(
    'Select',
    list(range(1,11))
)
'Your choice is', option,'.'

if st.checkbox('Show Image'):
    img=Image.open("moon_waifu2x.png")
    st.image(img,caption='tif',use_column_width=True)
