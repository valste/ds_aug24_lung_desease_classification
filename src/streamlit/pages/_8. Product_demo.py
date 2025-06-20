# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------#
# 3-rd-party                                                                    #
# -----------------------------------------------------------------------------#
import os
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# -----------------------------------------------------------------------------#
# Project-level imports                                                         #
# -----------------------------------------------------------------------------#
from process_imgs import image_path_to_base64_html, process_images       # moved up
from src.defs import ModelType as mt, PROJECT_DIR
from src.utils.datahelper import DataHelper as dh

# -----------------------------------------------------------------------------#
# Constants                                                                     #
# -----------------------------------------------------------------------------#

DATA_FOR_PRODUCT_DEMO = os.path.join(PROJECT_DIR, "src", "streamlit", "data", "data_for_product_demo", "unlabeled")

# -----------------------------------------------------------------------------#
# Streamlit page config & global CSS                                            #
# -----------------------------------------------------------------------------#
st.set_page_config(page_title="Lung diagnostics", layout="wide")
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            overflow: hidden !important;
            height: 100vh !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🫁 Lung diagnostics")

# -----------------------------------------------------------------------------#
# Session-state scaffolding ★ NEW                                               #
# -----------------------------------------------------------------------------#
for k, v in {
    "overlay": False,              # draw veil this pass?
    "predict_running": False,      # heavy work scheduled?
    "results_df": pd.DataFrame(),  # right-hand grid data
    "selected_rows": [],           # rows picked on the left
    "grid_version": 0,             # bump → force grid rebuild
}.items():
    st.session_state.setdefault(k, v)


# -----------------------------------------------------------------------------#
# Custom JS renderers                                                           #
# -----------------------------------------------------------------------------#
IMAGE_RENDERER = JsCode(
    """
class HtmlCellRenderer {
    init(params) {
        this.eGui = document.createElement('div');
        this.eGui.innerHTML = params.value;
    }
    getGui() {
        return this.eGui;
    }
}
"""
)

AUTO_SIZE_JS = JsCode(
    """
function(params) {
    const allColumnIds = [];
    params.columnApi.getAllColumns().forEach(col => allColumnIds.push(col.getColId()));
    params.columnApi.autoSizeColumns(allColumnIds, false);
}
"""
)

CODE_RENDERER = JsCode("""
class CodeCellRenderer {
  init(params) {
    this.eGui = document.createElement('pre');
    this.eGui.style.margin = '0';
    this.eGui.style.padding = '6px 8px';
    this.eGui.style.background = '#f6f8fa';
    this.eGui.style.fontFamily = 'monospace';
    this.eGui.style.whiteSpace = 'pre';
    this.eGui.textContent = params.value;
  }
  getGui() {
    return this.eGui;
  }
}
""")

# -----------------------------------------------------------------------------#
# Sidebar / control widgets                                                     #
# -----------------------------------------------------------------------------#
cols = st.columns(4)                                                # shrink to 4 (button moves)

with cols[0]:
    dataset = st.selectbox("Select dataset", ["Original-rotated", "Original-images", "Unknown-images"])
    image_dir = os.path.join(DATA_FOR_PRODUCT_DEMO, dataset)
    image_names = dh.get_file_names(image_dir, extensions=(".png", ".jpg", ".jpeg"))
    imgs_base64_html = [
        image_path_to_base64_html(os.path.join(image_dir, fn), max_width=200) for fn in image_names
    ]

with cols[1]:
    orientation_model = st.selectbox("Orientation model", [mt.MOBILENET, mt.RESNET50])

with cols[2]:
    segmentation_model = st.selectbox("Segmentation model", [mt.GAN, mt.UNET])

with cols[3]:
    classifier_model = st.selectbox("Classifier", [mt.CUST_COVID_CNN, mt.CAPSNET])
    
    
# -----------------------------------------------------------------------------#
# Callback: user pressed “Predict” ★ NEW                                        #
# -----------------------------------------------------------------------------#
def start_predict():
    sel = st.session_state.get("latest_selection", [])
    
    # ------ choose ONE of the following lines ------ #
    if sel is None:
        st.warning("Please select at least one image first.")
        return

    st.session_state.selected_rows = sel     # store the list (not DataFrame!)
    st.session_state.overlay = True
    st.session_state.predict_running = True
    
st.button("Predict", use_container_width=True, key="predict_btn", type="primary", on_click=start_predict)

# -----------------------------------------------------------------------------#
# DataFrames for AG-Grid                                                        #
# -----------------------------------------------------------------------------#
df_left = pd.DataFrame({"Preview": imgs_base64_html, "Filename": image_names})

# --------------- Left grid options ------------------------------------------ #
gb_left = GridOptionsBuilder.from_dataframe(df_left)
gb_left.configure_column("Preview", cellRenderer=IMAGE_RENDERER, autoHeight=True)
gb_left.configure_column("Filename", flex=1, checkboxSelection=True)
gb_left.configure_selection(
    selection_mode="multiple",
    use_checkbox=False,
    rowMultiSelectWithClick=True,
    suppressRowClickSelection=False,
)
gb_left.configure_grid_options(onFirstDataRendered=AUTO_SIZE_JS)
grid_options_left = gb_left.build()
grid_options_left["defaultColDef"] = {"resizable": True, "flex": 1, "minWidth": 120}

# --------------- Right grid options ----------------------------------------- #
gb_right = GridOptionsBuilder()
gb_right.configure_column("Preview", cellRenderer=IMAGE_RENDERER, autoHeight=True, flex=1)
gb_right.configure_column("Protocol", cellRenderer=CODE_RENDERER, autoHeight=True, flex=2)
gb_right.configure_grid_options(
    localeText={"noRowsToShow": "Select images on the left and press Predict button"},
    onFirstDataRendered=AUTO_SIZE_JS,
)
grid_options_right = gb_right.build()
grid_options_right["defaultColDef"] = {"resizable": True, "minWidth": 100}

# -----------------------------------------------------------------------------#
# Helper: build “protocol” strings ★ NEW (extracted)                            #
# -----------------------------------------------------------------------------#
def make_protocol(row, class_cols, show_sep=True, show_file=True):
    file_line = f"{row['Filename']}\n" if show_file else ""
    class_line = "class:      " + "  ".join(f"{c:<15}" for c in class_cols)
    conf_line  = "confidence: " + "  ".join(f"{row[c]:<15.2f}" for c in class_cols)
    sep = "\n" + ("_" * max(len(class_line), len(conf_line))) if show_sep else ""
    return f"{file_line}{class_line}\n{conf_line}{sep}"

# -----------------------------------------------------------------------------#
# Layout – two-column main area                                                 #
# -----------------------------------------------------------------------------#
left_col, right_col = st.columns([3, 5])

# ---------------- Left: selection grid -------------------------------------- #
with left_col:
    st.markdown("##### Select for processing")
    grid_response_left = AgGrid(
        df_left,
        key="left_grid",
        gridOptions=grid_options_left,
        update_on=[],                            # no Python callbacks per click
        allow_unsafe_jscode=True,
        columns_auto_size_mode=True,
        height=600,
    )
    # cache selection in session_state so the callback can read it
    st.session_state.latest_selection = grid_response_left["selected_rows"]


# ---------------- Heavy work – only on 2nd pass ----------------------------- #
if st.session_state.predict_running:
    # simulate long task – remove in production
    # time.sleep(5)                                                   # ★ DEMO ONLY
    df_selected_rows = pd.DataFrame(st.session_state.selected_rows)

    selected_models = {
        "orientation": orientation_model,
        "segmentation": segmentation_model,
        "desease_classifier": classifier_model,
    }

    df_ori, df_m, df_d = process_images(
        dataset_dir=image_dir,
        df_selected_rows=df_selected_rows,
        selected_models=selected_models,
    )

    # build “Protocol”
    class_cols = [c for c in df_ori.columns if c != "Filename"]
    df_ori[class_cols] = df_ori[class_cols].apply(pd.to_numeric, errors="coerce")
    ori_p = df_ori.apply(make_protocol, axis=1, args=(class_cols,))

    class_cols = [c for c in df_d.columns if c != "Filename"]
    df_d[class_cols] = df_d[class_cols].apply(pd.to_numeric, errors="coerce")
    dis_p = df_d.apply(make_protocol, axis=1, args=(class_cols, False, False))

    df_m["Protocol"] = ori_p + "\n" + dis_p

    # store + mark “done”
    st.session_state.results_df = df_m
    st.session_state.predict_running = False
    st.session_state.overlay = False
    st.session_state.grid_version += 1          # force new grid
    st.rerun()                     # 3rd pass – veil disappears


# ---------------- Right: results grid --------------------------------------- #
with right_col:
    st.markdown("##### Results")
    AgGrid(
        st.session_state.results_df,
        key=f"results_grid_{st.session_state.grid_version}",
        gridOptions=grid_options_right,
        update_on=[],                            # keep state client-side
        allow_unsafe_jscode=True,
        columns_auto_size_mode=True,
        height=600,
    )

    

# -----------------------------------------------------------------------------#
# Overlay – draw only while heavy job runs ★ NEW                               #
# -----------------------------------------------------------------------------#
if st.session_state.overlay:
    st.markdown(
        """
        <style>
        #overlay{
            position:fixed;top:0;left:0;width:100vw;height:100vh;
            background:rgba(0,0,0,.4);backdrop-filter:blur(2px);
            display:flex;align-items:center;justify-content:center;
            z-index:10000;pointer-events:all;
        }
        #overlay h1{color:white;font-size:2rem;text-align:center;}
        </style>
        <div id="overlay"><h1>Please wait…</h1></div>
        """,
        unsafe_allow_html=True,
    )
