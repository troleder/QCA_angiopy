import os
import os.path
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import pydicom
import glob
import mpld3
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import tifffile
from streamlit_plotly_events import plotly_events
from streamlit_drawable_canvas import st_canvas
from PIL import Image
# from streamlit_image_coordinates import streamlit_image_coordinates
import predict
import angioPyFunctions
import scipy
import cv2

import ssl
import pooch

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AngioPy Segmentation", layout="wide")

if 'stage' not in st.session_state:
    st.session_state.stage = 0


segmentationModelWeights = pooch.retrieve(
    url="doi:10.5281/zenodo.13848135/modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth",
    known_hash="md5:bf893ef57adaf39cfee33b25c7c1d87b",
)


# Make output folder
# os.makedirs(name=outputPath, exist_ok=True)

# arteryDictionary = {
#     'LAD':       {'colour': "#f03b20"},
#     'CX':        {'colour': "#31a354"},
#     'OM':    {'colour' : "#74c476"},
#     'RCA':       {'colour': "#08519c"},
#     'AM':   {'colour' : "#3182bd"},
#     'LM':        {'colour' : "#984ea3"},
# }

# def file_selector(folder_path='.'):
#     fileNames = [file for file in glob.glob(f"{folder_path}/*")]
#     selectedDicom = st.sidebar.selectbox('Select a DICOM file:', fileNames)
#     if selectedDicom is None:
#         return None

#     return selectedDicom

@st.cache_data
def selectSlice(slice_ix, pixelArray, fileName):

    # Save the selected frame 
    tifffile.imwrite(f"{outputPath}/{fileName}", pixelArray[slice_ix, :, :])

    # Set the button as clicked
    st.session_state.btnSelectSlice = True


DicomFolder = "Dicoms/"
# exampleDicoms = {
#     'RCA2' : 'Dicoms/RCA1',
#     'RCA1' : 'Dicoms/RCA4',
#     # 'RCA2' : 'Dicoms/RCA2',
#     # 'RCA3' : 'Dicoms/RCA3',
#     # 'LCA1' : 'Dicoms/LCA1',
#     # 'LCA2' : 'Dicoms/LCA2',
# 
# }
exampleDicoms = {}
files = sorted(glob.glob(DicomFolder+"/*"))
for file in files:
    exampleDicoms[os.path.basename(file)] = file


# Main text
st.markdown("<h1 style='text-align: center;'>AngioPy Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'> Welcome to <b>AngioPy Segmentation</b>, an AI-driven, coronary angiography segmentation tool.</h1>", unsafe_allow_html=True)
st.markdown("")

# Build the sidebar
# Select DICOM file: here eventually we will use the file_uploader widget, but for the demo this is deactivate. Instead we will have a choice of 3 anonymised DICOMs to pick from
# selectedDicom = st.sidebar.file_uploader("Upload DICOM file:",type=["dcm"], accept_multiple_files=False)

# def changeSessionState():

#     # value += 1

#     print("CHANGED!")


DropDownDicom = st.sidebar.selectbox("Select example DICOM file:",
                        options = list(exampleDicoms.keys()),
                        # on_change=changeSessionState(st.session_state.key),
                        key="dicomDropDown"
                    )

selectedDicom = exampleDicoms[DropDownDicom]

stepOne = st.sidebar.expander("STEP ONE", True)
stepTwo = st.sidebar.expander("STEP TWO", True)

# Create tabs 
tab1, tab2 = st.tabs(["Segmentation", "Analysis"])

# Increase tab font size
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:16px;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

# while True:
# Once a file is uploaded, the following annotation sequence is initiated 
if selectedDicom is not None:
        try:
            print(f"Trying to load {selectedDicom}")
            dcm = pydicom.dcmread(selectedDicom, force=True)

            # handAngle = dcm.PositionerPrimaryAngle
            # headAngle = dcm.PositionerSecondaryAngle
            # dcmLabel = f"{'LAO' if handAngle > 0 else 'RAO'} {numpy.abs(handAngle):04.1f}° {'CRA' if headAngle > 0 else 'CAU'} {numpy.abs(headAngle):04.1f}°"

            pixelArray = dcm.pixel_array

            # Just take first channel if it's RGB?
            if len(pixelArray.shape) == 4:
                pixelArray = pixelArray[:,:,:,0]

            n_slices = pixelArray.shape[0]

            slice_ix = 0
        except:
            selectedDicom = None
            # continue

        with tab1:

            with stepOne:
                st.write("Select frame for annotation. Aim for an end-diastolic frame with good visualisation of the artery of interest.")

                slice_ix = st.slider('Frame', 0, n_slices-1, int(n_slices/2), key='sliceSlider')


                predictedMask = numpy.zeros_like(pixelArray[slice_ix, :, :])


            with stepTwo:

                selectedArtery = st.selectbox("Select artery for annotation:",
                        ['LAD', 'CX', 'RCA', 'LM', 'OM', 'AM', 'D'],
                        key="arteryDropMenu"
                    )

                st.write("Beginning with the desired start point and finishing at the desired end point, click along the artery aiming for ~5-10 points.")


                stroke_color = angioPyFunctions.colourTableList[selectedArtery]


            col1, col2 = st.columns((15,15))

            with col1:
                col1a, col1b, col1c = st.columns((1,10,1))

                with col1b:

                    leftImageText = "<p style='text-align: center; color: white;'>Beginning with the desired <u><b>start point</b></u> and finishing at the desired <u><b>end point</b></u>, click along the artery aiming for ~5-10 points. Segmentation is automatic.</p>"

                    st.markdown(f"<h5 style='text-align: center; color: white;'>Selected frame</h5>", unsafe_allow_html=True)

                    st.markdown(leftImageText, unsafe_allow_html=True)

                    selectedFrame = pixelArray[slice_ix, :, :]
                    selectedFrame = cv2.resize(selectedFrame, (512,512))

                    # Create a canvas component
                    annotationCanvas = st_canvas(
                        fill_color="red",  # Fixed fill color with some opacity
                        stroke_width=1,
                        stroke_color="red",
                        background_color='black',
                        background_image= Image.fromarray(selectedFrame),
                        update_streamlit=True,
                        height=512,
                        width=512,
                        drawing_mode="point",
                        point_display_radius=2,
                        key=st.session_state.dicomDropDown,
                    )


                    # Do something interesting with the image data and paths
                    if annotationCanvas.json_data is not None:
                        objects = pd.json_normalize(annotationCanvas.json_data["objects"]) # need to convert obj to str because PyArrow

                        if len(objects) != 0:

                            for col in objects.select_dtypes(include=['object']).columns:
                                objects[col] = objects[col].astype("str")

                            groundTruthPoints = numpy.vstack(
                                (
                                    numpy.array(objects['top']),
                                    numpy.array(objects['left']+3.5)
                                )
                            ).T

                            predictedMask = angioPyFunctions.arterySegmentation(
                                pixelArray[slice_ix],
                                groundTruthPoints,
                                segmentationModelWeights
                            )

            with col2:
                col2a, col2b, col2c = st.columns((1,10,1))

                with col2b:
                    st.markdown(f"<h5 style='text-align: center; color: white;'>Predicted mask</h1>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: white;'>If the predicted mask has errors, restart and select more points to help the segmentation model. </p>", unsafe_allow_html=True)

                    stroke_color = "rgba(255, 255, 255, 255)"

                    maskCanvas = st_canvas(
                        fill_color=angioPyFunctions.colourTableList[selectedArtery],  # Fixed fill color with some opacity
                        stroke_width=0,
                        stroke_color=stroke_color,
                        background_color='black',
                        background_image= Image.fromarray(predictedMask),
                        update_streamlit=True,
                        height=512,
                        width=512,
                        drawing_mode="freedraw",
                        point_display_radius=3,
                        key="maskCanvas",
                    )


                    # Check that the mask array is not blank
                    if numpy.sum(predictedMask) > 0 and len(objects)>4:
                        # add alpha channel to predict mask in order to merge
                        b_channel, g_channel, r_channel = cv2.split(predictedMask)
                        a_channel = numpy.full_like(predictedMask[:,:,0], fill_value=255)

                        predictedMaskRGBA = cv2.merge((predictedMask, a_channel))


                        with tab2:
                            # combinedMask = cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2RGB)

                            # print(combinedMask.shape)
                            # tifffile.imwrite(f"{outputPath}/test.tif", combinedMask)


                            # tab2Col1, tab2Col2, tab2Col3 = st.columns([1,15,1])
                            tab2Col1, tab2Col2 = st.columns([20,10])

                            with tab2Col1:
                                st.markdown(f"<h5 style='text-align: center; color: white;'><br>Artery profile</h5>", unsafe_allow_html=True)

                                # Extract thickness information from mask
                                EDT = scipy.ndimage.distance_transform_edt(cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2GRAY))

                                # Skeletonise, get a list of ordered centreline points, and spline them
                                skel = angioPyFunctions.skeletonise(predictedMaskRGBA)
                                tck = angioPyFunctions.skelSplinerWithThickness(skel=skel, EDT=EDT)

                                # Interogate the spline function over 1000 points
                                splinePointsY, splinePointsX, splineThicknesses = scipy.interpolate.splev(
                                numpy.linspace(
                                    0.0,
                                    1.0,
                                    1000), 
                                    tck)

                                clippingLength = 20

                                vesselThicknesses = splineThicknesses[clippingLength:-clippingLength]*2

                                fig = px.line(x=numpy.arange(1,len(vesselThicknesses)+1),y=vesselThicknesses, labels=dict(x="Centreline point", y="Thickness (pixels)"), width=800)
                                # fig.update_layout(showlegend=False, xaxis={'showgrid': False, 'zeroline': True})
                                fig.update_traces(line_color='rgb(31, 119, 180)', textfont_color="white", line={'width':4})
                                fig.update_xaxes(showline=True, linewidth=2, linecolor='white', showgrid=False,gridcolor='white')
                                fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='white')

                                fig.update_layout(yaxis_range=[0,numpy.max(vesselThicknesses)*1.2])
                                fig.update_layout(font_color="white",title_font_color="white")
                                fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)'})


                                selected_points = plotly_events(fig)



                            with tab2Col2:

                                st.markdown(f"<h5 style='text-align: center; color: white;'><br>Contours</h5>", unsafe_allow_html=True)


                                selectedFrameRGBA = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGBA)

                                contour = angioPyFunctions.maskOutliner(labelledArtery=predictedMaskRGBA[:,:,0], outlineThickness=1)

                                selectedFrameRGBA[contour, :] =    [angioPyFunctions.colourTableList[selectedArtery][2],
                                                                    angioPyFunctions.colourTableList[selectedArtery][1],
                                                                    angioPyFunctions.colourTableList[selectedArtery][0],
                                                                    255]

                                fig2 = px.imshow(selectedFrameRGBA)


                                fig2.update_xaxes(visible=False)
                                fig2.update_yaxes(visible=False)
                                fig2.update_layout(margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0},) #remove margins
                                # fig2.coloraxis(visible=False)

                                fig2.update_traces(dict(
                                    showscale=False, 
                                    coloraxis=None, 
                                    colorscale='gray'), selector={'type':'heatmap'})

                                fig2.add_trace(go.Scatter(x=splinePointsX[clippingLength:-clippingLength], y=splinePointsY[clippingLength:-clippingLength], line=dict(width=1)))

                                st.plotly_chart(fig2, use_container_width=True)
