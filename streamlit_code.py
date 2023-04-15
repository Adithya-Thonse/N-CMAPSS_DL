import pandas as pd
import numpy as np
import streamlit as st
from os.path import join as opj
import matplotlib.pyplot as plt
prediction_csv_dir = "prediction_csv_dir"
filenames = ['N-CMAPSS_DS02-006', 'N-CMAPSS_DS07', 'N-CMAPSS_DS06', 'N-CMAPSS_DS01-005',
             'N-CMAPSS_DS05', 'N-CMAPSS_DS03-012', 'N-CMAPSS_DS08c-008', 'N-CMAPSS_DS08a-009', 'N-CMAPSS_DS04']

def main():
    st.set_page_config(page_title="Model Deployment", page_icon=":sun:", layout="centered", initial_sidebar_state="auto",
                       menu_items=None)
    st.title("Model Deployment")
    filename = st.sidebar.selectbox(label="Choose any dataset to pick engines from:", options=filenames)
    file_devtest_df = pd.read_csv("File_DevUnits_TestUnits.csv")
    units_index_test = np.fromstring(
        file_devtest_df[file_devtest_df.File == filename + '.h5']["Test Units"].values[0][1:-1],
        dtype=np.float, sep=' ').tolist()
    unit = st.sidebar.selectbox(label="Choose any Engine unit to predict RULs from:", options=[int(x) for x in units_index_test])
    rul_df = pd.read_csv(opj(prediction_csv_dir, "{}_Unit_{}.csv".format(filename, int(unit))))

    gt_rul = st.sidebar.slider(label="Choose a ground truth RUL",
                       min_value=int(min(rul_df.RUL.values)), max_value=int(max(rul_df.RUL.values)))
    models = st.sidebar.multiselect(label="Choose Models:", options=["Transformer", "LargestCUDNN", "GRUCNNDC", "DeepGRU"])

    temp_df = rul_df[rul_df.RUL == gt_rul]
    index = temp_df.index.values[0]

    fig = plt.figure(figsize=(5,5))
    plt.step([x*10+10 for x in sorted(rul_df.RUL.values)], rul_df.RUL.values, label="RUL range", alpha=0.5)
    plt.scatter(index * 10 + 5, temp_df.RUL.values[0], label="Ground Truth", marker='*')
    if 'Transformer' in models:
        plt.scatter(index * 10 + 5, temp_df.Transformer.values[0], label="Transformer", marker='o')
    if 'LargestCUDNN' in models:
        plt.scatter(index * 10 + 5, temp_df.LargestCUDNN.values[0], label="LargestCUDNN", marker='x')
    if 'GRUCNNDC' in models:
        plt.scatter(index * 10 + 5, temp_df.GRUCNNDC.values[0], label="GRUCNNDC", marker='D')
    if 'DeepGRU' in models:
        plt.scatter(index * 10 + 5, temp_df.DeepGRU.values[0], label="DeepGRU", marker=',')
    plt.xlabel("Timestamp")
    plt.ylabel("RUL")
    plt.grid(axis='x', color='0.95')
    plt.legend(title='Model:')
    plt.title('Unit: {}'.format(unit))

    # plt.plot()
    st.pyplot(fig)



if __name__ == '__main__':
    main()