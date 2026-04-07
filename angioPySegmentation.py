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
import predict
import angioPyFunctions
import scipy
import scipy.signal
import cv2
import io
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="AngioPy Segmentation", layout="wide")

if 'stage' not in st.session_state:
    st.session_state.stage = 0

@st.cache_data
def selectSlice(slice_ix, pixelArray, fileName):
    tifffile.imwrite(f"{outputPath}/{fileName}", pixelArray[slice_ix, :, :])
    st.session_state.btnSelectSlice = True

# ── DICOM file list ───────────────────────────────────────────────────────────
DicomFolder = "Dicom/"
exampleDicoms = {}
for file in sorted(glob.glob(DicomFolder + "/*")):
    exampleDicoms[os.path.basename(file)] = file

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align: center;'>AngioPy Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Welcome to <b>AngioPy Segmentation</b>, an AI-driven, coronary angiography segmentation tool.</h5>", unsafe_allow_html=True)
st.markdown("")

# ── Sidebar ───────────────────────────────────────────────────────────────────
uploadedDicom = st.sidebar.file_uploader("Upload DICOM file", type=["dcm", "dicom", "DCM"], key="dicomUploader")

if uploadedDicom is not None:
    # Save uploaded file to a temp path so pydicom can read it
    import tempfile
    tmpDicomPath = os.path.join(tempfile.gettempdir(), uploadedDicom.name)
    with open(tmpDicomPath, "wb") as f:
        f.write(uploadedDicom.getbuffer())
    selectedDicom = tmpDicomPath
    dicomLabel = uploadedDicom.name
    st.sidebar.success(f"Loaded: {uploadedDicom.name}")
elif exampleDicoms:
    DropDownDicom = st.sidebar.selectbox(
        "Or select example DICOM:",
        options=list(exampleDicoms.keys()),
        key="dicomDropDown"
    )
    selectedDicom = exampleDicoms[DropDownDicom]
    dicomLabel = DropDownDicom
else:
    st.warning("No DICOM files found. Please upload a DICOM file.")
    st.stop()

# key used for canvas reset when file changes
if "dicomDropDown" not in st.session_state:
    st.session_state["dicomDropDown"] = dicomLabel

stepOne = st.sidebar.expander("STEP ONE", True)
stepTwo = st.sidebar.expander("STEP TWO", True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Segmentation", "Analysis"])

st.markdown('''<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size:16px; }
</style>''', unsafe_allow_html=True)

# ── Load DICOM ────────────────────────────────────────────────────────────────
if selectedDicom is not None:
    try:
        dcm = pydicom.dcmread(selectedDicom, force=True)
        pixelArray = dcm.pixel_array
        if len(pixelArray.shape) == 4:
            pixelArray = pixelArray[:, :, :, 0]
        n_slices = pixelArray.shape[0]
        slice_ix = 0
    except Exception as e:
        st.error(f"Could not load DICOM file: {e}")
        st.stop()

    with stepOne:
        st.write("Select frame for annotation. Aim for an end-diastolic frame with good visualisation of the artery of interest.")
        slice_ix = st.slider('Frame', 0, n_slices - 1, int(n_slices / 2), key='sliceSlider')
        predictedMask = numpy.zeros_like(pixelArray[slice_ix, :, :])

    with stepTwo:
        selectedArtery = st.selectbox(
            "Select artery for annotation:",
            ['LAD', 'CX', 'RCA', 'LM', 'OM', 'AM', 'D'],
            key="arteryDropMenu"
        )
        st.write("Beginning with the desired start point and finishing at the desired end point, click along the artery aiming for ~5-10 points.")

    # ── SEGMENTATION TAB ──────────────────────────────────────────────────────
    with tab1:
        objects = pd.DataFrame()

        selectedFrame     = pixelArray[slice_ix, :, :]
        selectedFrame     = cv2.resize(selectedFrame, (512, 512))
        selectedFrameRGB  = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5 style='text-align:center; color:white;'>Selected frame</h5>", unsafe_allow_html=True)

            # ── Mode toggle ───────────────────────────────────────────────────
            canvasMode = st.radio(
                "Canvas mode:",
                ["📏 Calibrate catheter (6F = 1.98 mm)", "📍 Annotate artery"],
                horizontal=True,
                key="canvasMode"
            )

            isCalibMode = canvasMode.startswith("📏")

            if isCalibMode:
                st.caption("Kliknij kilka żółtych punktów wzdłuż środka cewnika. Aplikacja automatycznie wykryje jego szerokość.")

                calibDotCanvas = st_canvas(
                    fill_color="yellow",
                    stroke_width=2,
                    stroke_color="yellow",
                    background_color='black',
                    background_image=Image.fromarray(selectedFrameRGB),
                    update_streamlit=True,
                    height=512,
                    width=512,
                    drawing_mode="point",
                    point_display_radius=5,
                    key="canvas_calib_dots",
                )

                if calibDotCanvas.json_data is not None:
                    dotObjs = pd.json_normalize(calibDotCanvas.json_data["objects"])
                    if len(dotObjs) > 0 and "left" in dotObjs.columns:
                        detectedDiameters = []
                        for _, drow in dotObjs.iterrows():
                            cx = int(float(drow.get("left", 0)) + 5)
                            cy = int(float(drow.get("top",  0)) + 5)
                            cy = max(0, min(511, cy))
                            cx = max(0, min(511, cx))

                            # Scan both horizontal and vertical profiles, take smaller (= across catheter)
                            bestDiam = None
                            for axis in ["horizontal", "vertical"]:
                                margin = 60
                                if axis == "horizontal":
                                    x0p = max(0, cx - margin)
                                    x1p = min(512, cx + margin)
                                    profile = selectedFrame[cy, x0p:x1p].astype(float)
                                else:
                                    y0p = max(0, cy - margin)
                                    y1p = min(512, cy + margin)
                                    profile = selectedFrame[y0p:y1p, cx].astype(float)

                                if len(profile) < 10:
                                    continue

                                # Smooth and compute gradient magnitude
                                smoothed = numpy.convolve(profile, numpy.ones(3)/3, mode='same')
                                grad = numpy.abs(numpy.gradient(smoothed))
                                if grad.max() == 0:
                                    continue

                                # Find peaks in gradient (catheter edges)
                                peaks = scipy.signal.find_peaks(grad, height=grad.max() * 0.25, distance=4)[0]
                                if len(peaks) >= 2:
                                    diam = int(peaks[-1]) - int(peaks[0])
                                    if diam > 3:
                                        if bestDiam is None or diam < bestDiam:
                                            bestDiam = diam

                            if bestDiam is not None:
                                detectedDiameters.append(bestDiam)

                        if detectedDiameters:
                            avgDiam = numpy.mean(detectedDiameters)
                            st.session_state["mmPerPixelCalib"] = 1.98 / avgDiam
                            st.session_state["calibLinePx"]     = avgDiam

                # Show calibration status
                mmPerPixelCalib = st.session_state.get("mmPerPixelCalib", None)
                if mmPerPixelCalib:
                    linePx = st.session_state.get("calibLinePx", 0)
                    st.success(f"✅ Calibration: {linePx:.1f} px = 1.98 mm → **{mmPerPixelCalib:.4f} mm/px**")
                else:
                    try:
                        dicomMmPx = float(dcm.ImagerPixelSpacing[0]) * (float(dcm.DistanceSourceToPatient) / float(dcm.DistanceSourceToDetector))
                        st.info(f"ℹ️ No catheter calibration yet — using DICOM metadata ({dicomMmPx:.4f} mm/px).")
                    except Exception:
                        st.info("ℹ️ No calibration yet.")

            else:
                st.caption("Click along the artery from start to end — aim for 5–10 points.")
                annotationCanvas = st_canvas(
                    fill_color="red",
                    stroke_width=2,
                    stroke_color="red",
                    background_color='black',
                    background_image=Image.fromarray(selectedFrameRGB),
                    update_streamlit=True,
                    height=512,
                    width=512,
                    drawing_mode="point",
                    point_display_radius=2,
                    key="canvas_seg_" + str(dicomLabel),
                )

                # Show calibration status in annotation mode too
                mmPerPixelCalib = st.session_state.get("mmPerPixelCalib", None)
                if mmPerPixelCalib:
                    linePx = st.session_state.get("calibLinePx", 0)
                    st.success(f"✅ Calibration: {linePx:.1f} px = 1.98 mm → **{mmPerPixelCalib:.4f} mm/px**")
                else:
                    try:
                        dicomMmPx = float(dcm.ImagerPixelSpacing[0]) * (float(dcm.DistanceSourceToPatient) / float(dcm.DistanceSourceToDetector))
                        st.info(f"ℹ️ No catheter calibration yet — using DICOM metadata ({dicomMmPx:.4f} mm/px). Switch to 📏 mode to calibrate.")
                    except Exception:
                        st.info("ℹ️ No calibration yet. Switch to 📏 mode to calibrate.")

                # ── Annotation logic ──────────────────────────────────────────
                if annotationCanvas.json_data is not None:
                    objects = pd.json_normalize(annotationCanvas.json_data["objects"])

                    if len(objects) != 0:
                        for c in objects.select_dtypes(include=['object']).columns:
                            objects[c] = objects[c].astype("str")

                        groundTruthPoints = numpy.vstack((
                            numpy.array(objects['top']),
                            numpy.array(objects['left'] + 3.5)
                        )).T

                        with st.spinner(f"Running segmentation on {len(objects)} points (30–60 s on CPU)…"):
                            try:
                                mask = angioPyFunctions.arterySegmentation(
                                    pixelArray[slice_ix],
                                    groundTruthPoints,
                                )
                                predictedMask = predict.CoronaryDataset.mask2image(mask)
                                predictedMask = numpy.asarray(predictedMask)
                            except Exception as segErr:
                                st.error(f"Segmentation error: {segErr}")

        with col2:
            st.markdown("<h5 style='text-align:center; color:white;'>Predicted mask</h5>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:white;'>If the predicted mask has errors, restart and select more points.</p>", unsafe_allow_html=True)

            maskCanvas = st_canvas(
                fill_color=angioPyFunctions.colourTableList[selectedArtery],
                stroke_width=0,
                stroke_color="rgba(255,255,255,255)",
                background_color='black',
                background_image=Image.fromarray(predictedMask),
                update_streamlit=True,
                height=512,
                width=512,
                drawing_mode="freedraw",
                point_display_radius=3,
                key="maskCanvas",
            )

    # ── ANALYSIS TAB ──────────────────────────────────────────────────────────
    if numpy.sum(predictedMask) > 0 and len(objects) > 4:
        b_channel, g_channel, r_channel = cv2.split(predictedMask)
        a_channel = numpy.full_like(predictedMask[:, :, 0], fill_value=255)
        predictedMaskRGBA = cv2.merge((predictedMask, a_channel))

        with tab2:
            tab2Col1, tab2Col2 = st.columns([20, 10])

            with tab2Col1:
                st.markdown("<h5 style='text-align:center; color:white;'><br>Artery profile</h5>", unsafe_allow_html=True)

                EDT  = scipy.ndimage.distance_transform_edt(cv2.cvtColor(predictedMaskRGBA, cv2.COLOR_RGBA2GRAY))
                skel = angioPyFunctions.skeletonise(predictedMaskRGBA)
                tck  = angioPyFunctions.skelSplinerWithThickness(skel=skel, EDT=EDT)

                splinePointsY, splinePointsX, splineThicknesses = scipy.interpolate.splev(
                    numpy.linspace(0.0, 1.0, 1000), tck)

                clippingLength    = 20
                vesselThicknesses = splineThicknesses[clippingLength:-clippingLength] * 2

                fig = px.line(
                    x=numpy.arange(1, len(vesselThicknesses) + 1),
                    y=vesselThicknesses,
                    labels=dict(x="Centreline point", y="Thickness (pixels)"),
                    width=800
                )
                fig.update_traces(line_color='rgb(31, 119, 180)', line={'width': 4})
                fig.update_xaxes(showline=True, linewidth=2, linecolor='white', showgrid=False, gridcolor='white')
                fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='white')
                fig.update_layout(
                    yaxis_range=[0, numpy.max(vesselThicknesses) * 1.2],
                    font_color="white", title_font_color="white",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                )
                selected_points = plotly_events(fig)

            with tab2Col2:
                st.markdown("<h5 style='text-align:center; color:white;'><br>Contours</h5>", unsafe_allow_html=True)

                selectedFrameRGBA = cv2.cvtColor(selectedFrame, cv2.COLOR_GRAY2RGBA)
                contour = angioPyFunctions.maskOutliner(labelledArtery=predictedMaskRGBA[:, :, 0], outlineThickness=1)
                selectedFrameRGBA[contour, :] = [
                    angioPyFunctions.colourTableList[selectedArtery][2],
                    angioPyFunctions.colourTableList[selectedArtery][1],
                    angioPyFunctions.colourTableList[selectedArtery][0],
                    255
                ]

                fig2 = px.imshow(selectedFrameRGBA)
                fig2.update_xaxes(visible=False)
                fig2.update_yaxes(visible=False)
                fig2.update_layout(margin={"t": 0, "b": 0, "r": 0, "l": 0, "pad": 0})
                fig2.update_traces(dict(showscale=False, coloraxis=None, colorscale='gray'), selector={'type': 'heatmap'})
                fig2.add_trace(go.Scatter(
                    x=splinePointsX[clippingLength:-clippingLength],
                    y=splinePointsY[clippingLength:-clippingLength],
                    line=dict(width=1)
                ))
                st.plotly_chart(fig2, use_container_width=True)

            # ── QCA METRICS (outside nested columns) ──────────────────────────
            mmPerPixelCalib = st.session_state.get("mmPerPixelCalib", None)
            if mmPerPixelCalib:
                mmPerPixel  = mmPerPixelCalib
                calibSource = "6F catheter"
            else:
                try:
                    mmPerPixel  = float(dcm.ImagerPixelSpacing[0]) * (float(dcm.DistanceSourceToPatient) / float(dcm.DistanceSourceToDetector))
                    calibSource = "DICOM metadata"
                except Exception:
                    mmPerPixel  = None
                    calibSource = "unknown"

            origH, origW = pixelArray[slice_ix].shape[:2]
            # vesselThicknesses are in 512-scale pixels (mask space)
            # convert to mm directly using the pixel spacing at original image scale
            # (origW/512 accounts for any resize from original to 512)
            # ×10 correction: raw pixel→mm gives values in cm, multiply to get mm
            pxToMm = (origW / 512.0) * mmPerPixel if mmPerPixel else None

            refLen     = max(1, int(len(vesselThicknesses) * 0.20))
            proxDiamMm = numpy.mean(vesselThicknesses[:refLen])  * (pxToMm or 1.0)
            distDiamMm = numpy.mean(vesselThicknesses[-refLen:]) * (pxToMm or 1.0)
            refDiamMm  = (proxDiamMm + distDiamMm) / 2.0
            mldMm      = numpy.min(vesselThicknesses)            * (pxToMm or 1.0)

            pctDiam = (1.0 - mldMm / refDiamMm) * 100.0 if refDiamMm > 0 else 0.0
            pctArea = (1.0 - (mldMm / refDiamMm) ** 2) * 100.0 if refDiamMm > 0 else 0.0

            # arc length along centreline (spline points are in 512-scale px → convert)
            spX    = splinePointsX[clippingLength:-clippingLength] * (origW / 512.0)
            spY    = splinePointsY[clippingLength:-clippingLength] * (origH / 512.0)
            diffs  = numpy.sqrt(numpy.diff(spX) ** 2 + numpy.diff(spY) ** 2)
            cumLen = numpy.concatenate([[0], numpy.cumsum(diffs)])  # in original pixels
            totalLenMm = cumLen[-1] * mmPerPixel if mmPerPixel else cumLen[-1]

            # stenosis length: segment where diameter < 50 % of reference
            vesselThicknessMm = vesselThicknesses * (pxToMm or 1.0)
            stenosisMask = vesselThicknessMm < (refDiamMm * 0.5)
            if numpy.any(stenosisMask):
                idxs = numpy.where(stenosisMask)[0]
                stenosisLenMm = (cumLen[min(idxs[-1], len(cumLen)-1)] - cumLen[idxs[0]]) * (mmPerPixel or 1.0)
            else:
                stenosisLenMm = 0.0

            st.markdown("---")
            st.markdown("<h5 style='color:white;'>📐 QCA Metrics</h5>", unsafe_allow_html=True)
            if mmPerPixelCalib:
                st.success(f"✅ **6F catheter calibration** ({mmPerPixel:.4f} mm/px)")
            else:
                st.warning(f"⚠️ **DICOM metadata** ({mmPerPixel:.4f} mm/px) — switch to 📏 mode in Segmentation for accuracy")

            if mmPerPixel:
                m1, m2, m3 = st.columns(3)
                m1.metric("% Diameter Stenosis", f"{pctDiam:.1f}%")
                m2.metric("% Area Stenosis",     f"{pctArea:.1f}%")
                m3.metric("MLD",                 f"{mldMm:.2f} mm")
                m4, m5, m6 = st.columns(3)
                m4.metric("Proximal Diameter",   f"{proxDiamMm:.2f} mm")
                m5.metric("Distal Diameter",     f"{distDiamMm:.2f} mm")
                m6.metric("Reference Diameter",  f"{refDiamMm:.2f} mm")
                m7, m8 = st.columns(2)
                m7.metric("Lesion Length",       f"{totalLenMm:.1f} mm")
            else:
                m1, m2 = st.columns(2)
                m1.metric("% Diameter Stenosis", f"{pctDiam:.1f}%")
                m2.metric("% Area Stenosis",     f"{pctArea:.1f}%")

            # ── EXPORT ────────────────────────────────────────────────────────
            dicomBaseName = os.path.splitext(os.path.basename(selectedDicom))[0]
            st.markdown("---")
            st.markdown("<h5 style='color:white;'>Export results</h5>", unsafe_allow_html=True)

            maskBuf = io.BytesIO()
            Image.fromarray(predictedMask).save(maskBuf, format="PNG")
            st.download_button("⬇ Download mask (PNG)", data=maskBuf.getvalue(),
                file_name=f"{dicomBaseName}_{selectedArtery}_mask_frame{slice_ix}.png", mime="image/png")

            thicknessDf = pd.DataFrame({
                "centreline_point": numpy.arange(1, len(vesselThicknesses) + 1),
                "thickness_px":     vesselThicknesses,
                "thickness_mm":     vesselThicknessMm,
                "arc_length_mm":    cumLen * (mmPerPixel if mmPerPixel else 1.0),
            })
            summaryRows = pd.DataFrame([
                {"centreline_point": "--- QCA SUMMARY ---"},
                {"centreline_point": "calibration_source",      "thickness_px": calibSource},
                {"centreline_point": "pct_diameter_stenosis_%", "thickness_px": round(pctDiam, 2)},
                {"centreline_point": "pct_area_stenosis_%",     "thickness_px": round(pctArea, 2)},
                {"centreline_point": "MLD_mm",                  "thickness_px": round(mldMm, 3)},
                {"centreline_point": "proximal_diameter_mm",    "thickness_px": round(proxDiamMm, 3)},
                {"centreline_point": "distal_diameter_mm",      "thickness_px": round(distDiamMm, 3)},
                {"centreline_point": "reference_diameter_mm",   "thickness_px": round(refDiamMm, 3)},
                {"centreline_point": "lesion_length_mm",         "thickness_px": round(totalLenMm, 2)},
            ])
            csvBuf = io.StringIO()
            pd.concat([thicknessDf, summaryRows], ignore_index=True).to_csv(csvBuf, index=False)
            st.download_button("⬇ Download QCA (CSV)", data=csvBuf.getvalue(),
                file_name=f"{dicomBaseName}_{selectedArtery}_QCA_frame{slice_ix}.csv", mime="text/csv")

            overlayBuf = io.BytesIO()
            Image.fromarray(selectedFrameRGBA).save(overlayBuf, format="PNG")
            st.download_button("⬇ Download overlay (PNG)", data=overlayBuf.getvalue(),
                file_name=f"{dicomBaseName}_{selectedArtery}_overlay_frame{slice_ix}.png", mime="image/png")
