import numpy
import scipy.interpolate
import skimage.filters
import skimage.morphology
import scipy.ndimage
import scipy.optimize
import predict
from PIL import Image
from fil_finder import FilFinder2D
import astropy.units as u
from tqdm import tqdm
import pooch
import utils.dataset
import cv2


colourTableHex = {
                'LAD':       "#f03b20",
                'D':         "#fd8d3c",
                'CX':        "#31a354",
                'OM':        "#74c476",
                'RCA':       "#08519c",
                'AM':        "#3182bd",
                'LM':        "#984ea3",
                }

colourTableList = {}

for item in colourTableHex.keys():
    ### WARNING HACK: The colours go in backwards here for some reason perhaps related to RGBA?
    colourTableList[item] = [int(colourTableHex[item][5:7], 16),
                             int(colourTableHex[item][3:5], 16),
                             int(colourTableHex[item][1:3], 16)]


def skeletonise(maskArray):
    # if len(maskArray.shape) == 3:
    maskArray = cv2.cvtColor(maskArray, cv2.COLOR_BGR2GRAY)

    skeleton = skimage.morphology.skeletonize(maskArray.astype('bool'))

    # Process the skeleton and find the longest path
    fil = FilFinder2D(skeleton.astype('uint8'),
                    distance=250 * u.pc, mask=skeleton, beamwidth=10.0*u.pix)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
                    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=400 * u.pix,
                        skel_thresh=10 * u.pix, prune_criteria='length')

    # add image arrays dictionary
    # tifffile.imwrite(os.path.join(arteryFolder, "skel.tif"), fil.skeleton.astype('<u1')*255)

    skel = fil.skeleton.astype('<u1')*255

    return skel


def skelEndpoints(skel):
    #skel[skel!=0] = 1
    skel = numpy.uint8(skel>0)

    # Apply the convolution.
    kernel = numpy.uint8([[1,  1, 1],
    [1, 10, 1],
    [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # Look through to find the value of 11.
    # This returns a mask of the endpoints, but if you
    # just want the coordinates, you could simply
    # return np.where(filtered==11)
    out = numpy.zeros_like(skel)
    out[numpy.where(filtered==11)] = 1
    endCoords = numpy.where(filtered==11)
    endCoords = list(zip(*endCoords))
    startPoint = endCoords[0]
    endPoint = endCoords[1]

    # print(f"Skel starts at {startPoint} and finishes at {endPoint}")

    return startPoint, endPoint


def skelPointsInOrder(skel, startPoint=None):
    """
    put in a skel image, get the y, x points out in order
    """

    # Lazy!!
    if startPoint is None:
        startPoint, _ = skelEndpoints(skel)

    # get the coordinates of all points in the skeleton
    skelXY = numpy.array(numpy.where(skel))
    skelPoints = list(zip(skelXY[0], skelXY[1]))
    skelLength = len(skelPoints)

    # Loop through the skeleton starting with startPoint, deleting the starting point from the skelPoints list, and finding the closest pixel. This is appended to orderedPoints. startPoint now becomes the last point to be appended.
    startPointCopy = startPoint # copied as we are going to loop and overwrite, but want to also keep the original startPoint
    orderedPoints = []

    while len(skelPoints) > 1:

        skelPoints.remove(startPointCopy)

        # Calculate the point that is closest to the start point
        diffs = numpy.abs(numpy.array(skelPoints)-numpy.array(startPointCopy))
        dists = numpy.sum(diffs,axis=1) #l1-distance
        closest_point_index = numpy.argmin(dists)
        closestPoint = skelPoints[closest_point_index]
        orderedPoints.append(closestPoint)

        startPointCopy = closestPoint

    orderedPoints = numpy.array(orderedPoints)

    # YX points
    return orderedPoints


def skelSplinerWithThickness(skel, EDT, smoothing=50, order=3, decimation=2):
    # NOTE: the coordinate seem to come out with y first, then x
    startPoint, endPoint = skelEndpoints(skel)

    # Impose an order to points
    orderedPoints = skelPointsInOrder(skel, startPoint)

    # unzip ordered points to extract x and y arrays
    x = orderedPoints[:, 1].ravel()
    y = orderedPoints[:, 0].ravel()

    x = x[::decimation]
    y = y[::decimation]

    #NOTE: Should the EDT be median filtered? I wonder in fact if doing so will reduce the accuracy of the model.
    # EDT = skimage.filters.median(EDT)

    t = EDT[y, x]

    x = x[0:-1]
    y = y[0:-1]
    t = t[0:-1]

    print(x.shape, y.shape, t.shape)

    tcko, uo = scipy.interpolate.splprep(
        [y, x, t], s=smoothing, k=order, per=False)

    return tcko


def arterySegmentation(inputImage, groundTruthPoints, segmentationModelWeights=None):
    """
    Segment a single greyscale artery with a UNet model.

    Parameters
    ----------
        inputImage: 2D numpy array
            Ideally this input is normalised 0-255 and 512x512
            If a different size it is rescaled along with groundTruthPoints

        groundTruthPoints: Nx2 numpy array
            Y and X positions of annotated points along the artery,
            Ordering is not important except that start and end points should be top and bottom of the array

        segmentationModelWeights: segmentation model weights (pth), optional
            Segmentation model weights to use.
            If not set the default ones from this paper: https://doi.org/10.1016/j.ijcard.2024.132598

    Returns
    -------
        mask : 512x512 numpy array (int64)
            Mask selecting the selected artery, 0 = background and 1 = artery
    """
    if segmentationModelWeights is None:
        segmentationModelWeights = pooch.retrieve(
            url="doi:10.5281/zenodo.13848135/modelWeights-InternalData-inceptionresnetv2-fold2-e40-b10-a4.pth",
            known_hash="md5:bf893ef57adaf39cfee33b25c7c1d87b",
        )

    if inputImage.shape[0] != 512 and inputImage.shape[1] != 512:
        ratioYX = numpy.array([512./inputImage.shape[0], 512./inputImage.shape[1]])
        print(f"arterySegmentation(): Rescaling image to 512x512 by {ratioYX=}, and also applying this to input points")
        inputImage = scipy.ndimage.zoom(inputImage, ratioYX)
        groundTruthPoints *= ratioYX
        print(inputImage.shape)

    imageSize = inputImage.shape

    n_classes = 2 # binary output

    net = predict.smp.Unet(
        encoder_name='inceptionresnetv2',
        encoder_weights="imagenet",
        in_channels=3,
        classes=n_classes
    )

    net = predict.nn.DataParallel(net)

    device = predict.torch.device('cuda' if predict.torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    net.load_state_dict(
        predict.torch.load(
            segmentationModelWeights,
            map_location=device
        )
    )

    orig_image = Image.fromarray(inputImage)

    image = predict.Image.new('RGB', imageSize, (0, 0, 0))
    image.paste(orig_image, (0, 0))

    imageArray = numpy.array(image).astype('uint8')

    # Clear last channels
    imageArray[:, :, -1] = 0
    imageArray[:, :, -2] = 0

    ## Get endpoints of skeleton
    startPoint = groundTruthPoints[0]
    endPoint = groundTruthPoints[-1]

    # End points on Channel 1
    for y, x in [startPoint, endPoint]:
        y = int(numpy.round(y))
        x = int(numpy.round(x))
        imageArray[y-2:y+2, x-2:x+2, 1] = 255

    # All other points on Channel 2
    for y, x in groundTruthPoints[1:-1]:
        y = int(numpy.round(y))
        x = int(numpy.round(x))
        imageArray[y-2:y+ 2, x-2:x+2, 2] = 255

    image = Image.fromarray(imageArray.astype(numpy.uint8))

    mask = predict.predict_img(
        net=net,
        dataset_class=utils.dataset.CoronaryDataset,
        full_img=image,
        scale_factor=1,
        device=device
    )

    return mask



def maskOutliner(labelledArtery, outlineThickness=3):

    # Compute the boundary of the mask
    contours, _ = cv2.findContours(labelledArtery, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tmp = numpy.zeros_like(labelledArtery)
    boundary = cv2.drawContours(tmp, contours, -1, (255,255,255), outlineThickness)
    boundary = boundary > 0

    return boundary
