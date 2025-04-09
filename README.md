# AngioPy Segmentation

## AngioPy paper
Please see [here](https://doi.org/10.1016/j.ijcard.2024.132598) for our paper in the International Journal of Cardiology

## AngioPy in the news
AngioPy segmentation is being used in [this RTS reportage on AI in Cardiology](https://www.rts.ch/play/tv/19h30/video/lia-fait-irruption-en-cardiologie-et-redefinit-le-role-des-medecins?urn=urn:rts:video:15479233) (in French)

## Online Example
Please visit https://imaging.epfl.ch/angiopy-segmentation/ for a live demo of this code on some example DICOM images

![](illustration.mp4)

## Description
This software allows single arteries to be segmented given a few clicks on a single time frame with a PyTorch 2 Deep Learning model.

## Installing and running
 - Install dependencies: ` pip install -r requirements.txt`
 - Launch Streamlit Web Interface: `streamlit run angioPySegmentation.py --server.fileWatcherType none`

 ...a website should pop up in your browser!

 You need to create a /Dicom folder and put some angiography DICOMs in there
